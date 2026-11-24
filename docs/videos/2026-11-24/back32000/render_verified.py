"""Replay original recorded native actions, verify every state, then encode videos."""
import sys,json,gzip
from pathlib import Path
from hashlib import file_digest,sha256
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,'/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01')
import whole4_new as c
RUN=Path('/home/lzx/cl_binding/.server/k25_back_stable_20261124_01')
OUT=Path(__file__).resolve().parent

def longest(r):
    best=current=0
    for m in r['trace']['success']:
        current=current+1 if m&8 else 0;best=max(best,current)
    return best

marker=c.read(RUN/'dev_32000_COMPLETE.json')
assert c.digest(RUN/'032000/trainable.safetensors')==marker['weights_sha256']
assert c.digest(RUN/'dev_32000.jsonl.gz')==marker['rows_sha256']
with gzip.open(RUN/'dev_32000.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
assert len(rows)==20 and sum(r['terminal']['back'] for r in rows)==6
failure=max((r for r in rows if r['ever']['back'] and not r['terminal']['back']),key=lambda r:(longest(r),-r['init']))
success=min((r for r in rows if r['terminal']['back']),key=lambda r:(not r['stable_released_final'],r['init']))
cases={r['init']:r for r in c.read(Path('/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01/dev_cases.json'))}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
manifest=dict(checkpoint_step=32000,weights_sha256=marker['weights_sha256'],record_sha256=marker['rows_sha256'],
    selection='failure: longest consecutive success among eventual failures, tie smallest init; success: prefer stable-released final, then smallest init',
    method='replay saved executed actions from original evaluation; no new sampling and no controller modification',videos=[])
lab=c.ref.old.original.k21.Lab(family='scene2',models=False)
try:
    for kind,r in [('ever_yes_final_no',failure),('ever_yes_final_yes',success)]:
        obs=c.ref.old.base.sr.reset(lab.ctx,cases[r['init']]);assert c.ref.old.timing.signature(lab,obs)==r['signatures'][0]
        first=next(i+1 for i,m in enumerate(r['trace']['success']) if m&8)
        last=max(i+1 for i,m in enumerate(r['trace']['success']) if m&8)
        name=f'back32000_init{r["init"]}_{kind}.mp4';path=OUT/name
        assert not path.exists();frames=[];states=[];signatures=[r['signatures'][0]]
        original_step=lab.ctx.step
        def wrapped(action):
            obs,executed=original_step(action)
            states.append(np.asarray(lab.ctx.env.get_sim_state(),dtype=np.float64).copy())
            return obs,executed
        lab.ctx.step=wrapped
        try:
            for block in range(40):
                action=np.array(r['trace']['actions'][block*10:(block+1)*10],dtype=np.float32)
                assert sha256(action.tobytes()).hexdigest()==r['action_hashes'][block]
                obs,events=lab.execute(obs,action,block*10,cases[r['init']]['z0'],None)
                for j,event in enumerate(events):
                    idx=block*10+j
                    np.testing.assert_array_equal(event['obj'],r['trace']['object'][idx],err_msg=f'object step {idx+1}')
                    np.testing.assert_array_equal(event['eef'],r['trace']['eef'][idx],err_msg=f'eef step {idx+1}')
                    assert sum(1<<n for n,g in enumerate(c.ref.old.spec.scoring.SITES) if event['pred'][g])==r['trace']['success'][idx]
                    assert int(event['c1'])+2*int(event['c2'])==r['trace']['contacts'][idx]
                signatures.append(c.ref.old.timing.signature(lab,obs))
        finally:lab.ctx.step=original_step
        assert len(states)==400 and signatures==r['signatures']
        print('REPLAY_EXACT',kind,r['init'],first,last,flush=True)
        # Rendering is a separate pass AFTER dynamics verification; it cannot change the replay.
        for i,state in enumerate(states):
            lab.ctx.env.sim.set_state_from_flattened(state);lab.ctx.env.sim.forward()
            views=[]
            for camera in ['agentview','robot0_eye_in_hand']:
                a=lab.ctx.env.sim.render(width=512,height=512,camera_name=camera,depth=False)
                views.append(np.ascontiguousarray(a[::-1,::-1]))
            flag=bool(r['trace']['success'][i]&8);both=r['trace']['contacts'][i]==3
            im=Image.new('RGB',(1024,668),(17,22,29));im.paste(Image.fromarray(views[0]),(0,94));im.paste(Image.fromarray(views[1]),(512,94));d=ImageDraw.Draw(im)
            d.text((14,8),f'Checkpoint 32000 | native BACK | init {r["init"]} | step {i+1:03d}/400 | {((i+1)/20):.2f}s',font=font,fill='white')
            d.text((14,39),f'Back NOW: {"YES" if flag else "NO"}    Ever: {"YES" if i+1>=first else "NO"}    Both-finger contact: {"YES" if both else "NO"}',font=font,fill=(103,224,160) if flag else (255,181,140))
            d.text((14,69),'Original action/state replay: all400 object/EEF positions, contacts and Back flags verified',font=small,fill=(195,205,216))
            d.text((14,610),f'Agent camera   |   first Back={first}, last Back={last}, final400={r["terminal"]["back"]}',font=small,fill='white')
            for k,m in enumerate(r['trace']['success']):
                x=14+int(k*996/400);d.rectangle((x,640,x+3,650),fill=(44,174,113) if m&8 else (61,69,80))
            x=14+int(i*996/400);d.line((x,636,x,656),fill='white',width=2)
            frames.append(np.array(im))
        imageio.mimwrite(path,frames,fps=20,codec='libx264',quality=8,macro_block_size=None)
        Image.fromarray(frames[first-1]).save(OUT/(path.stem+'_first_success.png'))
        Image.fromarray(frames[-1]).save(OUT/(path.stem+'_final400.png'))
        rec=dict(kind=kind,init=r['init'],file=name,first_success=first,last_success=last,longest_success_run=longest(r),
            terminal=r['terminal']['back'],stable_released_final=r['stable_released_final'],duration_seconds=20,frames=400,
            all_states_exact=True,all_predicates_exact=True,all_action_hashes_exact=True,all41_signatures_exact=signatures==r['signatures'],video_sha256=c.digest(path))
        manifest['videos'].append(rec);c.write(OUT/'videos.json',manifest)
        print('VIDEO_VERIFIED',rec,flush=True)
finally:lab.ctx.env.close()
c.write(OUT/'COMPLETE.json',dict(videos=2,frames=800,verified=True))
