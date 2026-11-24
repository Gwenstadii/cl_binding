"""Fixed 40/80/120-step, same-state policy x instruction continuations.

Each branch reconstructs the entire prefix by original executed controls. This
preserves controller state rather than restoring qpos alone. No CA patches.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import stage2p_ca_closedloop as base
from stage2k_entry_handoff import _control_state
from stage2r_handoff_protocol import jobs, summarize_episode


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot_key(ep, h):
    return f"{ep['model']}|{ep['requested_goal']}|i{ep['init_index']}|s{ep['noise_seed']}|h{h}"


def validate(cfg):
    assert cfg['boundaries'] == [40,80,120] and cfg['indices'] == [0,1]
    assert cfg['seeds'] == [2026102801,2026102802]
    assert cfg['horizon'] == 400 and cfg['n_action_steps'] == cfg['flow_steps'] == 10
    assert cfg['noise_replan_stride'] == 1000003 and cfg['expected_episodes'] == 192
    assert cfg['source_models'] == cfg['continuation_models'] == ['R0','FT4']
    assert cfg['prefix_goals'] == cfg['continuation_goals'] == ['left','front']
    assert not cfg['save_videos']


def references(cfg):
    source = Path(cfg['source_reference_root']) / 'evaluation'
    for name, expected in cfg['source_reference_hashes'].items():
        assert base.sha(source/name) == expected, name
    eps = [json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    chosen = [e for e in eps if e['condition'] == 'native' and e['requested_goal'] in ('left','front')
              or e['condition'] == 'full_old_response' and e['window'] == 120]
    traces = {e['episode_id']: [] for e in chosen}
    with gzip.open(source/'events.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['episode_id'] in traces: traces[r['episode_id']].append(r)
    natives = {(e['model'],e['requested_goal'],e['init_index'],e['noise_seed']):(e,traces[e['episode_id']])
               for e in chosen if e['condition']=='native'}
    hybrids = {(e['requested_goal'],e['init_index'],e['noise_seed']):traces[e['episode_id']]
               for e in chosen if e['condition']=='full_old_response'}
    assert len(natives)==16 and len(hybrids)==8
    return natives, hybrids


def physical_row(eid, step, action, obs, flags, np):
    executed=np.clip(action,-1,1)
    return {'episode_id':eid,'step':step,'predicates':flags,'action_unclipped':action.tolist(),
            'action_executed':executed.tolist(),'clipped_dimensions':(action!=executed).tolist(),
            'eef_position':np.asarray(obs['robot0_eef_pos']).tolist(),
            'object_positions':{k:np.asarray(v).tolist() for k,v in obs.items() if k.endswith('_pos') and not k.startswith('robot')}}


def compare_state(row, original, np):
    assert row['predicates']==original['predicates'], (row['episode_id'],row['step'],'predicate')
    error=max(float(np.max(np.abs(np.asarray(row['eef_position'])-original['eef_position']))),
              *(float(np.max(np.abs(np.asarray(v)-original['object_positions'][k]))) for k,v in row['object_positions'].items()))
    assert error<=1e-7, (row['episode_id'],row['step'],'position',error)
    return error


def signature(env,obs,np):
    state=np.asarray(env.get_sim_state(),dtype=np.float64)
    return {'sim_state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),
            'control_sha256':digest(_control_state(env)),
            'policy_observation_sha256':{k:hashlib.sha256(np.ascontiguousarray(obs[k]).tobytes()).hexdigest()
                for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}}


def reset(ep, cfg, rt, evaluation, env, states, xml, root_cfg, fresh):
    rt.np.random.seed(root_cfg['seed']+ep['init_index'])
    evaluation.reset_common(env,states['front'][ep['init_index']],root_cfg['seed']+ep['init_index'],
                            root_cfg['evaluation']['settle_steps'],xml)
    obs=fresh(env)
    initial=hashlib.sha256(rt.np.asarray(env.get_sim_state(),dtype=rt.np.float64).tobytes()).hexdigest()
    assert initial==ep['initial_state_sha256']
    return obs


def replay_step(row,eid,env,evaluation,predicates,fresh,np):
    obs,_,_,_=env.step(row['action_executed'])
    if row['step']%10==0:obs=fresh(env)
    actual=physical_row(eid,row['step'],np.asarray(row['action_unclipped']),obs,evaluation.events(env,predicates),np)
    error=compare_state(actual,row,np)
    return obs,actual,error


def protocol(report):
    report.pop('site',None);report.pop('intervention_steps',None)
    report.update(scope='fixed early handoff: original-prefix replay x continuation policy x instruction; no CA patch or training',
                  boundaries=[40,80,120],driver_sha256=base.sha(__file__))


def engineering(args,cfg,report,rt,evaluation,env,states,xml,instructions,root_cfg,fresh,guard):
    protocol(report);natives,_=references(cfg);checks=[]
    for ep, trace in natives.values():
        obs=reset(ep,cfg,rt,evaluation,env,states,xml,root_cfg,fresh)
        maximum=0.
        for row in trace[1:121]:
            obs,actual,error=replay_step(row,'engineering',env,evaluation,report['predicates'],fresh,rt.np)
            maximum=max(maximum,error)
            h=row['step']
            if h not in cfg['boundaries']:continue
            sign=signature(env,obs,rt.np)
            chunk,_=rt.sample(ep['model'],'native',ep['requested_goal'],obs,instructions,ep['noise_seed'],h//10,False,'engineering')
            controls=rt.physical_chunk(ep['model'],chunk)[:10]
            difference=float(rt.np.max(rt.np.abs(controls-rt.np.asarray([r['action_unclipped'] for r in trace[h+1:h+11]]))))
            assert difference<=1e-5, ('first chunk identity',snapshot_key(ep,h),difference)
            checks.append({'snapshot_id':snapshot_key(ep,h),'signature':sign,'prefix_position_max_abs':maximum,
                           'first_chunk_action_max_abs':difference,'sim_state':rt.np.asarray(env.get_sim_state()).tolist(),
                           'control_state':_control_state(env)})
        guard()
    assert len(checks)==48
    gate={'passed':True,'checks':checks,'config_sha256':report['config_sha256'],
          'driver_sha256':report['driver_sha256'],'model_sha256':report['model_sha256'],'sources':report['sources']}
    base.write(args.output/'engineering.json',gate)
    report['engineering_checks']=checks


def evaluate(args,cfg,report,rt,evaluation,env,states,xml,instructions,root_cfg,fresh,guard):
    protocol(report);natives,hybrids=references(cfg)
    gate_path=args.output.parent/'engineering/engineering.json'
    gate=json.loads(gate_path.read_text());assert gate['passed']
    for k in ('config_sha256','driver_sha256','model_sha256','sources'):assert gate[k]==report[k]
    base.write(args.output/'engineering.json',gate)
    expected_signatures={c['snapshot_id']:c['signature'] for c in gate['checks']}
    schedule=jobs();base.write(args.output/'jobs.json',schedule)
    with gzip.open(args.output/'events.jsonl.gz','wt') as out, gzip.open(args.output/'replans.jsonl.gz','wt') as replans:
        for number,job in enumerate(schedule):
            role,goal,h,receiver,request,index,seed=[job[k] for k in ('source_model','prefix_goal','boundary','continuation_model','requested_goal','init_index','noise_seed')]
            ep,reference=natives[role,goal,index,seed]
            eid=f"j{number:03d}|{role}_{goal}|h{h}|{receiver}_{request}|i{index}|s{seed}"
            report['episode_in_progress']=eid;guard()
            obs=reset(ep,cfg,rt,evaluation,env,states,xml,root_cfg,fresh)
            for model in rt.models.values():model.reset()
            base.dump_line(out,{'episode_id':eid,'step':0,'predicates':evaluation.events(env,report['predicates'])})
            trace=[];prefix_error=0.;identity_error=0.;hybrid_error=0.
            for row in reference[1:h+1]:
                obs,actual,error=replay_step(row,eid,env,evaluation,report['predicates'],fresh,rt.np)
                prefix_error=max(prefix_error,error);trace.append(actual);base.dump_line(out,{**actual,'phase':'replayed_prefix'})
                report['steps_completed']+=1
            sign=signature(env,obs,rt.np);sid=snapshot_key(ep,h)
            assert sign==expected_signatures[sid],('boundary changed',sid)
            for replan in range(h//10,40):
                chunk,meta=rt.sample(receiver,'native',request,obs,instructions,seed,replan,False,eid)
                base.dump_line(replans,{**meta,'effective_model':receiver,'requested_goal':request,'boundary':h,'snapshot_id':sid})
                controls=rt.physical_chunk(receiver,chunk)
                for offset,action in enumerate(controls[:10]):
                    step=replan*10+offset+1
                    obs,_,_,_=env.step(rt.np.clip(action,-1,1))
                    if step%10==0:obs=fresh(env)
                    actual=physical_row(eid,step,action,obs,evaluation.events(env,report['predicates']),rt.np)
                    trace.append(actual);base.dump_line(out,{**actual,'phase':'continuation'})
                    # Same-model/same-request continuations must recreate native trajectories.
                    if receiver==role and request==goal:
                        difference=float(rt.np.max(rt.np.abs(action-reference[step]['action_unclipped'])))
                        identity_error=max(identity_error,difference)
                        assert difference<=1e-5,('native action identity',eid,step,difference)
                        compare_state(actual,reference[step],rt.np)
                    # Exactly recreate the earlier R0-120 -> FT4 positive control too.
                    if role=='R0' and receiver=='FT4' and request==goal and h==120:
                        old=hybrids[goal,index,seed][step]
                        difference=float(rt.np.max(rt.np.abs(action-old['action_unclipped'])))
                        hybrid_error=max(hybrid_error,difference)
                        assert difference<=1e-5,('old hybrid action identity',eid,step,difference)
                        compare_state(actual,old,rt.np)
                    report['steps_completed']+=1
                report['replans_completed']+=1
                out.flush();replans.flush();guard()
            row={**job,'episode_id':eid,'snapshot_id':sid,'boundary_signature':sign,
                 'initial_state_sha256':ep['initial_state_sha256'],'source_episode_id':ep['episode_id'],
                 'prefix_position_max_abs':prefix_error,'native_identity_action_max_abs':identity_error,
                 'hybrid120_identity_action_max_abs':hybrid_error,
                 **summarize_episode(job,trace,cfg['movement_geometry']['sites'])}
            with (args.output/'episodes.jsonl').open('a') as f:base.dump_line(f,row)
            report['episodes_completed']=number+1
            print(json.dumps(row),flush=True);guard()
    assert report['steps_completed']==76800 and report['replans_completed']==6144
    assert rt.counts['integrator_velocity_calls']==rt.counts['recipient_denoise_calls']==61440
    assert rt.counts['donor_denoise_calls']==rt.counts['patch_flows']==0
    report.pop('episode_in_progress',None)


def main(args):
    base.validate_config=validate;base.engineering=engineering;base.evaluation_run=evaluate
    base.main(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--runtime-scripts',type=Path,required=True)
    parser.add_argument('--phase',choices=['engineering','evaluation'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
