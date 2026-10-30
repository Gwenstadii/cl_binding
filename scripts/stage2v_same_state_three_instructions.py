"""Same-instruction trajectories with observation-only controller instrumentation."""
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import stage2p_ca_closedloop as base
from stage2r_early_handoff import physical_row, compare_state, signature
from stage2k_entry_handoff import _snapshot, _restore_snapshot
from stage2r_handoff_protocol import summarize_episode

ORIGINAL_ENGINEERING=base.engineering


def jobs():
    return [dict(mode=m,requested_goal=g,init_index=i,noise_seed=s) for m,g,i,s in itertools.product(
        ('handoff80',),('left','front','right'),(0,1),(2026102801,2026102802))]


def references(cfg):
    selected={}
    for source,ref in cfg['references'].items():
        root=Path(ref['server_root'])/'evaluation'
        for name,h in ref['files'].items():assert base.sha(root/name)==h
        eps=[json.loads(x) for x in (root/'episodes.jsonl').read_text().splitlines()]
        wanted={}
        for e in eps:
            if source=='q' and e['model']=='FT4' and e['condition']=='native':mode='native'
            elif source=='r' and e['source_model']=='R0' and e['continuation_model']=='FT4' and e['boundary']==80 and e['prefix_goal']=='left':mode='handoff80'
            else:continue
            key=(mode,e['requested_goal'],e['init_index'],e['noise_seed'])
            wanted[e['episode_id']]=(key,e)
        traces={eid:[] for eid in wanted}
        with gzip.open(root/'events.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line)
                if r['episode_id'] in wanted:traces[r['episode_id']].append(r)
        for eid,(key,e) in wanted.items():selected[key]=(e,traces[eid])
    assert len(selected)==20
    return selected


class ControlWitness:
    """Observe set_goal's own ee_pos and resulting goal_pos in native world frame."""
    def __init__(self,env,np):
        self.controller=env.robots[0].controller;self.np=np
        self.original=self.controller.set_goal;self.rows=[]
        def wrapped(*args,**kwargs):
            out=self.original(*args,**kwargs);c=self.controller
            assert c.use_delta and c.impedance_mode=='fixed'
            self.rows.append({'eef_at_set_goal':c.ee_pos.copy().tolist(),'goal_pos':c.goal_pos.copy().tolist(),
                'world_delta':(c.goal_pos-c.ee_pos).copy().tolist()})
            return out
        self.controller.set_goal=wrapped
    def close(self):self.controller.set_goal=self.original


def sensor_details(env,obs,np):
    contacts=[]
    for contact in env.sim.data.contact[:env.sim.data.ncon]:
        a=env.sim.model.geom_id2name(contact.geom1) or '';b=env.sim.model.geom_id2name(contact.geom2) or ''
        if 'black_book' in a or 'black_book' in b:
            contacts.append({'geom1':a,'geom2':b,'distance':float(contact.dist)})
    try:
        grasp=bool(env.env._check_grasp(env.robots[0].gripper,env.env.objects_dict['black_book_1'].contact_geoms));error=None
    except (AttributeError,KeyError,TypeError) as exc:grasp=None;error=repr(exc)
    return {'book_contacts':contacts,'native_grasp':grasp,'grasp_api_error':error,
            'eef_quat':np.asarray(obs['robot0_eef_quat']).tolist(),'book_quat':np.asarray(obs['black_book_1_quat']).tolist(),
            'gripper_qpos':np.asarray(obs['robot0_gripper_qpos']).tolist(),
            'relative_book_eef':(np.asarray(obs['black_book_1_pos'])-obs['robot0_eef_pos']).tolist()}


def validate(cfg):
    assert cfg['goals']==['left','front','right'] and cfg['boundaries']==[80]
    assert cfg['indices']==[0,1] and cfg['seeds']==[2026102801,2026102802]
    assert cfg['horizon']==400 and cfg['n_action_steps']==cfg['flow_steps']==10 and cfg['expected_episodes']==12


def protocol(report):
    report.pop('site',None);report.pop('intervention_steps',None)
    report.update(scope='R0 Left80 saved full state, restored into FT4 Left/Front/Right; no training',driver_sha256=base.sha(__file__))


def engineering(args,cfg,report,rt,*rest):
    protocol(report);references(cfg)
    old=base.GOALS;base.GOALS=('left','front','right')
    try:ORIGINAL_ENGINEERING(args,cfg,report,rt,*rest)
    finally:base.GOALS=old


def evaluate(args,cfg,report,rt,evaluation,env,states,xml,instructions,root_cfg,fresh,guard):
    import av
    from PIL import Image,ImageDraw
    protocol(report);refs=references(cfg)
    gate=json.loads((args.output.parent/'engineering/engineering.json').read_text());assert gate['passed']
    for k in ('config_sha256','model_sha256','sources'):assert gate[k]==report[k]
    base.write(args.output/'engineering.json',gate)
    videos=args.output/'videos';videos.mkdir();witness=None
    snapshots=args.output/'snapshots';snapshots.mkdir()
    report['controller_source']={'path':__import__('inspect').getfile(type(env.robots[0].controller)),
        'sha256':base.sha(__import__('inspect').getfile(type(env.robots[0].controller)))}
    report['steps_completed']=0
    try:
        with gzip.open(args.output/'events.jsonl.gz','wt') as out,gzip.open(args.output/'replans.jsonl.gz','wt') as replans:
            for number,job in enumerate(jobs()):
                mode,g,index,seed=(job[k] for k in ('mode','requested_goal','init_index','noise_seed'))
                key=mode,g,index,seed;reference=refs.get(key)
                eid=f"j{number:02d}|{mode}|{g}|i{index}|s{seed}"
                report['episode_in_progress']=eid;guard()
                if witness is not None:witness.close()
                rt.np.random.seed(root_cfg['seed']+index)
                evaluation.reset_common(env,states['front'][index],root_cfg['seed']+index,root_cfg['evaluation']['settle_steps'],xml)
                obs=fresh(env);witness=ControlWitness(env,rt.np)
                initial=hashlib.sha256(rt.np.asarray(env.get_sim_state(),dtype=rt.np.float64).tobytes()).hexdigest()
                if reference:assert initial==reference[0]['initial_state_sha256']
                else:assert initial==refs['native',g,index,seed][0]['initial_state_sha256']
                for p in rt.models.values():p.reset()
                base.dump_line(out,{'episode_id':eid,'step':0,'predicates':evaluation.events(env,report['predicates'])})
                trace=[];maximum=0.;filename=f"{g.upper()}_{'R0_LEFT_001-080_then_FT4_'+g.upper()+'_081-400'}_init{index:02d}_seed{seed}.mp4"
                with av.open(str(videos/filename),'w') as video:
                    stream=video.add_stream('libx264',rate=20);stream.width=768;stream.height=492;stream.pix_fmt='yuv420p';stream.options={'crf':'20','preset':'fast'}
                    for replan in range(40):
                        role='R0' if mode=='handoff80' and replan<8 else 'FT4'
                        if replan==8:
                            sid=f'i{index}_s{seed}'
                            if g=='left':
                                arrays,meta_snapshot=_snapshot(env,obs)
                                rt.np.savez_compressed(snapshots/(sid+'.npz'),**arrays)
                                base.write(snapshots/(sid+'.json'),meta_snapshot)
                            with rt.np.load(snapshots/(sid+'.npz')) as saved:arrays={k:saved[k].copy() for k in saved.files}
                            meta_snapshot=json.loads((snapshots/(sid+'.json')).read_text())
                            witness.close()
                            obs,restore_check=_restore_snapshot(env,xml,arrays,meta_snapshot,root_cfg['seed']+index)
                            restored_signature=signature(env,obs,rt.np)
                            witness=ControlWitness(env,rt.np)
                        if replan<8:
                            source=refs['handoff80','left',index,seed][1]
                            controls=[rt.np.asarray(r['action_unclipped']) for r in source[10*replan+1:10*replan+11]]
                        else:
                            chunk,meta=rt.sample('FT4','native',g,obs,instructions,seed,replan,False,eid)
                            controls=rt.physical_chunk('FT4',chunk)[:10]
                            base.dump_line(replans,{**meta,'effective_model':'FT4','requested_goal':g})
                        for offset,action in enumerate(controls):
                            step=replan*10+offset+1;before=len(witness.rows)
                            obs,_,_,_=env.step(rt.np.clip(action,-1,1))
                            if step%10==0:obs=fresh(env)
                            assert len(witness.rows)==before+1,'Expected one native controller goal per action'
                            row=physical_row(eid,step,action,obs,evaluation.events(env,report['predicates']),rt.np)
                            if reference:maximum=max(maximum,compare_state(row,reference[1][step],rt.np))
                            row.update(controller=witness.rows[-1],**sensor_details(env,obs,rt.np));trace.append(row);base.dump_line(out,row)
                            canvas=Image.new('RGB',(768,492),'#fafafa');draw=ImageDraw.Draw(canvas)
                            draw.text((10,8),f'Instruction: {"LEFT" if step<=80 else g.upper()} | step {step:03d}/400 | controller {role}',fill='black')
                            draw.text((10,28),f'{mode} | init {index} | noise {seed} | '+'SAME SNAPSHOT RESTORED AT STEP 80',fill='black')
                            draw.text((10,48),f'Native In: {[k for k,v in row["predicates"].items() if v]} | native grasp: {row["native_grasp"]}',fill='black')
                            draw.text((10,78),'Agent view                                             Wrist view',fill='black')
                            for col,k in enumerate(('agentview_image','robot0_eye_in_hand_image')):
                                canvas.paste(Image.fromarray(rt.np.ascontiguousarray(obs[k][::-1,::-1])).resize((384,384)),(col*384,108))
                            for packet in stream.encode(av.VideoFrame.from_ndarray(rt.np.asarray(canvas),format='rgb24')):video.mux(packet)
                            report['steps_completed']+=1
                        out.flush();replans.flush();guard()
                    for packet in stream.encode():video.mux(packet)
                result={**job,'episode_id':eid,'initial_state_sha256':initial,'video':filename,'video_sha256':base.sha(videos/filename),
                    'reference_episode_id':reference[0]['episode_id'] if reference else None,'replay_position_max_abs':maximum,'snapshot_id':sid,'restored_signature':restored_signature,'restore_check':restore_check,
                    **summarize_episode({'boundary':80,'requested_goal':g},trace,cfg['movement_geometry']['sites'])}
                with (args.output/'episodes.jsonl').open('a') as f:base.dump_line(f,result)
                report['episodes_completed']=number+1;print(json.dumps(result),flush=True);guard()
        assert report['steps_completed']==4800 and rt.counts['integrator_velocity_calls']==3840 and rt.counts['patch_flows']==0
    finally:
        if witness is not None:witness.close()
    report.pop('episode_in_progress',None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runtime-scripts',type=Path,required=True)
    p.add_argument('--phase',choices=['engineering','evaluation'],required=True);p.add_argument('--output',type=Path,required=True)
    base.validate_config=validate;base.engineering=engineering;base.evaluation_run=evaluate;base.main(p.parse_args())
