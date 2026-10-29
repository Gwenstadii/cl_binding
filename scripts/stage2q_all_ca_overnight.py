"""Fixed all-CA closed-loop screen, using the validated stage2p runtime.

Adapter replaces only the registered protocol handlers. Checkpoints, native
sampling, message hooks, cache isolation and action postprocessing are reused.
"""
import argparse
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path

import stage2p_ca_closedloop as base

SITES=(1,3,5,7,9,11,13,15)
WINDOWS=(120,400)
MODES=('cross_same','cross_wrong','random_delta')
ORIGINAL_ENGINEERING=base.engineering
ORIGINAL_SEED=base.random_seed_for

def matched_seed(anchor,role,goal,site,salt):
    # Both windows use the same perturbation directions until step 120.
    anchor=re.sub(r'^j\d+\|','',anchor)
    anchor=re.sub(r'\|w\d+\|','|',anchor)
    return ORIGINAL_SEED(anchor,role,goal,site,salt)

def jobs():
    cases=[]
    def add(model,mode,site,window,goals):
        for goal in goals:
            for index in (0,1):
                for seed in (2026102801,2026102802):
                    cases.append(dict(model=model,condition=mode,site=site,window=window,
                                      requested_goal=goal,init_index=index,noise_seed=seed))
    add('R0','native',None,0,('left','front'))
    add('FT4','native',None,0,('left','front','right'))
    for window in WINDOWS:add('FT4','full_old_response',None,window,('left','front'))
    for site in SITES:
        for window in WINDOWS:
            for mode in MODES:add('FT4',mode,site,window,('left','front'))
    return cases

def validate(cfg):
    if (cfg['sites']!=list(SITES) or cfg['windows']!=list(WINDOWS)
        or cfg['indices']!=[0,1] or cfg['seeds']!=[2026102801,2026102802]
        or cfg['horizon']!=400 or cfg['n_action_steps']!=10 or cfg['flow_steps']!=10
        or cfg['noise_replan_stride']!=1000003 or cfg['save_videos']
        or cfg['expected_episodes']!=420):raise ValueError('Night protocol changed')

class Runtime(base.ClosedLoopRuntime):
    def __init__(self,cfg,*args):
        local=copy.deepcopy(cfg);local['site']=SITES[0]
        super().__init__(local,*args)
    def set_site(self,site):
        if site not in SITES:raise ValueError('Unknown site')
        for hook in self.hooks.values():hook.close()
        self.hooks={role:base.LayerMessageHook(policy,site,self.torch) for role,policy in self.models.items()}
        self.cfg['site']=site

def protocol(report,cfg):
    report.pop('site',None);report.pop('intervention_steps',None)
    if 'layer_map' in report:report['initial_layer_map']=report.pop('layer_map')
    report.update(scope='registered all-CA M screen, movement primary, no training',
                  sites=list(SITES),windows=list(WINDOWS),driver_sha256=base.sha(__file__))

def engineering(args,cfg,report,rt,*rest):
    protocol(report,cfg);checks=[]
    for site in SITES:
        rt.set_site(site)
        directory=args.output/f'site_{site:02d}';directory.mkdir()
        child=copy.copy(args);child.output=directory
        local={**cfg,'site':site}
        ORIGINAL_ENGINEERING(child,local,report,rt,*rest)
        gate=json.loads((directory/'engineering.json').read_text())
        checks.extend({**row,'site':site} for row in gate['checks'])
    report['engineering_checks']=checks
    base.write(args.output/'engineering.json',dict(passed=True,checks=checks,sites=list(SITES),
        tolerance=1e-5,script_sha256=report['script_sha256'],driver_sha256=report['driver_sha256'],
        config_sha256=report['config_sha256'],model_sha256=report['model_sha256'],sources=report['sources']))

def evaluate(args,cfg,report,rt,evaluation,env,states,xml,instructions,root_cfg,fresh,guard):
    from stage1_protocol import summarize_events
    protocol(report,cfg)
    gate_path=args.output.parent/'engineering/engineering.json';gate=json.loads(gate_path.read_text())
    if not gate['passed'] or any(gate[k]!=report[k] for k in
        ('script_sha256','driver_sha256','config_sha256','model_sha256','sources')):raise ValueError('Engineering provenance changed')
    base.write(args.output/'engineering.json',gate);rows=[]
    schedule=jobs();base.write(args.output/'jobs.json',schedule)
    with gzip.open(args.output/'events.jsonl.gz','wt') as events,gzip.open(args.output/'replans.jsonl.gz','wt') as replans,gzip.open(args.output/'norms.jsonl.gz','wt') as norms:
        for job_index,job in enumerate(schedule):
            role,mode,site,window,goal,index,seed=[job[k] for k in
                ('model','condition','site','window','requested_goal','init_index','noise_seed')]
            eid=f"j{job_index:03d}|{role}|{mode}|site{site}|w{window}|{goal}|i{index}|s{seed}"
            report['episode_in_progress']=eid;guard()
            if site is not None:rt.set_site(site)
            rt.np.random.seed(root_cfg['seed']+index)
            evaluation.reset_common(env,states['front'][index],root_cfg['seed']+index,
                root_cfg['evaluation']['settle_steps'],xml)
            obs=fresh(env);flat=rt.np.asarray(env.get_sim_state(),dtype=rt.np.float64)
            for policy in rt.models.values():policy.reset()
            flags=evaluation.events(env,report['predicates']);trace=[flags]
            if any(flags.values()):raise ValueError('Successful reset')
            base.dump_line(events,dict(episode_id=eid,step=0,predicates=flags))
            patches=0;clipped_steps=0
            for replan in range(40):
                start=replan*10
                effective='R0' if mode=='full_old_response' and start<window else role
                effective_mode='native' if mode=='full_old_response' else mode
                active=mode in MODES and start<window
                chunk,meta=rt.sample(effective,effective_mode,goal,obs,instructions,seed,replan,active,eid,norms)
                meta.update(effective_model=effective,site=site,window=window,condition=mode)
                base.dump_line(replans,meta);patches+=meta['patch_flow_count']
                controls=rt.physical_chunk(effective,chunk)
                for offset,action in enumerate(controls[:10]):
                    executed=rt.np.clip(action,-1,1);mask=action!=executed;clipped_steps+=int(mask.any())
                    obs,_,_,_=env.step(executed);step=start+offset+1
                    if step%10==0:obs=fresh(env)
                    flags=evaluation.events(env,report['predicates']);trace.append(flags)
                    base.dump_line(events,dict(episode_id=eid,step=step,predicates=flags,
                        action_unclipped=action.tolist(),action_executed=executed.tolist(),clipped_dimensions=mask.tolist(),
                        eef_position=rt.np.asarray(obs['robot0_eef_pos']).tolist(),
                        object_positions={k:rt.np.asarray(v).tolist() for k,v in obs.items()
                            if k.endswith('_pos') and not k.startswith('robot')}))
                events.flush();replans.flush();norms.flush()
                report['steps_completed']+=10;report['replans_completed']+=1;guard()
            row={**job,**summarize_events(goal,trace),'episode_id':eid,'steps':400,
                 'initial_state_sha256':hashlib.sha256(flat.tobytes()).hexdigest(),
                 'scene_xml_sha256':report['scene_xml_sha256'],'patch_flows':patches,'clipped_steps':clipped_steps}
            rows.append(row)
            with (args.output/'episodes.jsonl').open('a') as f:base.dump_line(f,row)
            report['episodes_completed']=len(rows);print(json.dumps(row),flush=True);guard()
    report.pop('episode_in_progress',None)
    expected={'recipient_denoise_calls':201280,'donor_denoise_calls':99840,
              'patch_flows':99840,'integrator_velocity_calls':168000,
              'extra_random_recipient_calibration_calls':33280,'prefix_builds':26784}
    if len(rows)!=420 or any(rt.counts[k]!=v for k,v in expected.items()):raise ValueError('Night budget matrix incomplete')

def main(args):
    base.validate_config=validate;base.ClosedLoopRuntime=Runtime
    base.engineering=engineering;base.evaluation_run=evaluate
    base.random_seed_for=matched_seed
    base.main(args)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--runtime-scripts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['engineering','evaluation'],required=True)
    main(p.parse_args())
