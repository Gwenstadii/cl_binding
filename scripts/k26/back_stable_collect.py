"""Real simulator release/retreat extensions from TRAIN demonstration ends only."""
import sys,json
from pathlib import Path
import h5py,numpy as np,torch
from robosuite.utils.transform_utils import quat2axisangle
sys.path.insert(0,'/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01')
import whole4_new as c
from stage1_smoke import sources
RUN=Path(__file__).resolve().parent
conv=c.read(Path('/home/lzx/cl_binding_stage2a/datasets/stage1_single_back_train_aligned_v1/r0_conversion.json'))
ctx=c.ref.old.original.k21.km.Ctx('scene2');bank=[];audit=[]
def raw(obs,action):
    b={'observation.state':torch.from_numpy(np.concatenate((obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos'])).astype(np.float32)),
       'action':torch.tensor(np.tile(action,(50,1)),dtype=torch.float32),'action_is_pad':torch.zeros(50,dtype=torch.bool),
       'task':'pick up the book and place it in the back compartment of the caddy'}
    for key,img in [('observation.images.image','agentview_image'),('observation.images.image2','robot0_eye_in_hand_image')]:
        b[key]=torch.from_numpy(np.ascontiguousarray(obs[img][::-1,::-1])).permute(2,0,1).float()/255
    return b
try:
    with h5py.File(sources(['back'])['back'],'r') as f:
        for ep in conv['episodes']:
            d=f['data'][ep['demo']];ctx.env.reset();obs=ctx.env.set_init_state(d['states'][-1]);frames=[];actions=[];flags=[];contacts=[]
            for k in range(400):
                a=np.array([0.,0.,.08 if 20<=k<50 else 0.,0.,0.,0.,-1.],dtype=np.float32)
                if k<60 or k%10==0:frames.append((k,raw(obs,a)))
                actions.append(a)
                obs,_,_,_=ctx.env.step(a);flags.append(bool(ctx.flags()['back']))
            keep=all(flags[-100:])
            if keep:
                # Every target chunk is the ACTUAL future command sequence, not a repeated transition command.
                for k,b in frames:
                    seq=actions[k:k+50];seq=seq+[actions[-1]]*(50-len(seq))
                    b['action']=torch.tensor(np.asarray(seq),dtype=torch.float32);bank.append(b)
            audit.append(dict(demo=ep['demo'],frames=ep['frames'],last_gripper=float(d['actions'][-1,-1]),
                held_back_at_end=flags[-1],last100_all_back=keep,ever_after_release=any(flags),accepted=keep))
            print('STABLE_DEMO',audit[-1],'bank',len(bank),flush=True)
finally:ctx.env.close()
assert len(audit)==40
c.write(RUN/'extension_audit.json',dict(training_demos_only=True,records=audit,accepted=sum(r['accepted'] for r in audit),samples=len(bank),
    min_frames=min(r['frames'] for r in audit),max_frames=max(r['frames'] for r in audit),last_gripper_closed=sum(r['last_gripper']>0 for r in audit)))
assert len(bank)>=200,'Insufficient independently stabilized training-demo extensions'
torch.save(bank,RUN/'stable_bank.pt')
print('STABLE_BANK_READY',len(bank),flush=True)
