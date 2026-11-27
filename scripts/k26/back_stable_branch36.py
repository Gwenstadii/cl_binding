"""Continue the actual trained F with validated real release/hold supervision."""
import sys,os,json,time,random,math
from pathlib import Path
import numpy as np,torch
from safetensors.torch import load_file,save_file
sys.path[:0]=['/home/lzx/cl_binding/.server/k19_retention_20261111_01','/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01']
import train_k19 as old
import whole4_new as original_eval
RUN=Path(__file__).resolve().parent
START=Path('/home/lzx/cl_binding/.server/k25_back_stable_20261124_01/036000')
BANK_SOURCE=Path('/home/lzx/cl_binding/.server/k25_back_stable_20261124_01')
INITIAL_STEP=36000
END_STEP=48000
BRANCH_SEED=20261125
def write(name,v):original_eval.write(RUN/name,v)
def sha(p):return original_eval.digest(p)

def evaluate(path,cases,name):
    p=original_eval.setup(24000);p['models']['executor']=str(path)
    lab=original_eval.ref.load(p);lab.alpha=1.;rows=[]
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),original_eval.ref.old.gzip.open(RUN/(name+'.jsonl.gz'),'xt') as f:
            for case in cases:
                r=original_eval.ref.episode(lab,case,'back','native')
                r['stable50']=sum(bool(m&8) for m in r['trace']['success'][-50:])>=45
                r['no_two_finger_hold_last10']=all(m!=3 for m in r['trace']['contacts'][-10:])
                r['stable_released_final']=r['terminal']['back'] and r['stable50'] and r['no_two_finger_hold_last10']
                f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();rows.append(r)
                print('NATIVE_CHECK',name,case['init'],r['terminal']['back'],r['stable_released_final'],flush=True)
    finally:lab.ctx.env.close()
    result=dict(n=20,terminal=sum(r['terminal']['back'] for r in rows),ever=sum(r['ever']['back'] for r in rows),
        stable_released_final=sum(r['stable_released_final'] for r in rows))
    write(name+'_COMPLETE.json',dict(**result,weights_sha256=sha(path),rows_sha256=sha(RUN/(name+'.jsonl.gz'))))
    return result

def main():
    torch.set_num_threads(2)
    bank_path=BANK_SOURCE/'stable_bank.pt'
    while not bank_path.exists():
        if (BANK_SOURCE/'collect_FAILED.json').exists():raise RuntimeError('terminal extension collection failed')
        time.sleep(5)
    bank=torch.load(bank_path,map_location='cpu',weights_only=False)
    # Skip k360..390: retain only chunks whose whole50 target commands were actually simulated.
    assert len(bank)%94==0
    bank=[r for i,r in enumerate(bank) if i%94<90]
    audit=original_eval.read(BANK_SOURCE/'extension_audit.json');assert audit['accepted']>=3 and len(bank)>=200
    assert sha(START/'trainable.safetensors')=='dad0a035dba1e7a0127a5f8c087f49928eb5b3158a2dba28135f075fd8ae2a00'
    assert original_eval.read(START/'READY.json')['step']==INITIAL_STEP
    policy,pre=old.setup(BRANCH_SEED);named={n:p for n,p in policy.named_parameters() if p.requires_grad};saved=load_file(str(START/'trainable.safetensors'))
    assert len(named)==155 and named.keys()==saved.keys()
    with torch.no_grad():
        for n,p in named.items():p.copy_(saved[n])
    del saved
    frozen=old.frozen_hash(policy)
    state=torch.load(START/'training_state.pt',map_location='cpu',weights_only=False)
    opt=torch.optim.AdamW(list(named.values()),lr=2e-5,betas=policy.config.optimizer_betas,eps=policy.config.optimizer_eps,weight_decay=policy.config.optimizer_weight_decay)
    opt.load_state_dict(state['optimizer'])
    for vals in opt.state.values():
        for k,v in vals.items():
            if isinstance(v,torch.Tensor) and k!='step':vals[k]=v.cuda()
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    del state
    # Fresh future minibatches and flow noise; optimizer moments are retained.
    torch.manual_seed(BRANCH_SEED);torch.cuda.manual_seed_all(BRANCH_SEED);np.random.seed(BRANCH_SEED);random.seed(BRANCH_SEED)
    data=old.open_data(Path('/home/lzx/cl_binding_stage2a/datasets/stage1_single_back_train_aligned_v1'))
    keys=list(bank[0]);dev=original_eval.read(Path('/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01/dev_cases.json'))
    ctx=original_eval.ref.old.original.k21.km.Ctx('scene2');confirm=[]
    try:
        for i in range(20):
            seed=202611270+i;np.random.seed(seed);ctx.env.seed(seed);ctx.env.reset()
            c=dict(init=700+i,state=np.asarray(ctx.env.get_sim_state()).tolist(),reset_seed=seed,seed=17970041+107*i)
            obs=original_eval.ref.old.base.sr.reset(ctx,c);assert not any(ctx.flags().values())
            c.update(signature=ctx.signature(obs),z0=float(obs[ctx.obj][2]));confirm.append(c)
    finally:ctx.env.close()
    write('confirmation_cases.json',confirm)
    write('manifest.json',dict(start=str(START),start_sha256=sha(START/'trainable.safetensors'),bank_sha256=sha(bank_path),
        samples=len(bank),supervision='3 original training frames +3 real successful release/retreat/hold frames per batch',
        optimizer_resumed=True,frozen_hash=frozen,block=4000,initial_step=INITIAL_STEP,end_step=END_STEP,branch_seed=BRANCH_SEED,optimizer_state_sha256=sha(START/'training_state.pt'),future_rng_reset=True,gate='both20-state native reset400 sets >=16 terminal and >=16 stable released finals; no inference stopping/teacher',
        training_demos_only=True,source_sha256=sha(Path(__file__))))
    step=INITIAL_STEP;attempts=[];start=time.monotonic()
    log=(RUN/'train.jsonl').open('x')
    qualified=False
    while step<END_STEP:
        policy.train()
        for j in range(4000):
            step+=1;rng=random.Random(BRANCH_SEED*1000003+step)
            ids=[rng.randrange(len(data)) for _ in range(3)];stable_ids=[rng.randrange(len(bank)) for _ in range(3)]
            examples=[{k:b[k] for k in keys} for b in data.__getitems__(ids)]+[bank[i] for i in stable_ids]
            batch=pre(torch.utils.data.default_collate(examples))
            lr=2e-5
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,_=policy(batch,reduction='none');loss=loss.mean()
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(named.values()),10.,error_if_nonfinite=True);opt.step()
            log.write(json.dumps(dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),indices=ids,stable_indices=stable_ids,lr=lr))+'\n')
            if step%25==0:
                log.flush();write('status.json',dict(status='training',step=step,loss=float(loss.detach()),elapsed=time.monotonic()-start));print('STABLE_TRAIN',step,float(loss.detach()),flush=True)
        dest=RUN/f'{step:06d}';dest.mkdir();assert old.frozen_hash(policy)==frozen
        path=dest/'trainable.safetensors';save_file({n:p.detach().cpu().contiguous() for n,p in named.items()},str(path))
        torch.save(dict(step=step,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),dest/'training_state.pt')
        write(f'{step:06d}/READY.json',dict(step=step,count=155,sha256=sha(path),frozen_unchanged=True,parent_sha256='f8bd197dc20886ee14873429c90a7b51ea00ccc2656c28fc542b625b1261772c'))
        write('status.json',dict(status='native_evaluation',step=step))
        result=evaluate(path,dev,f'dev_{step}');attempt=dict(step=step,development=result)
        if result['terminal']>=16 and result['stable_released_final']>=16:
            confirmation=evaluate(path,confirm,f'confirm_{step}');attempt['confirmation']=confirmation
            if confirmation['terminal']>=16 and confirmation['stable_released_final']>=16:
                attempts.append(attempt);write('QUALIFIED.json',dict(step=step,checkpoint=str(path),sha256=sha(path),attempts=attempts));write('status.json',dict(status='qualified',step=step));print('QUALIFIED',step,flush=True);qualified=True;break
        attempts.append(attempt);write('attempts.json',attempts)
    log.close()
    if not qualified:
        write('COMPLETE.json',dict(status='budget_complete_not_qualified',step=step,additional_steps=step-INITIAL_STEP,attempts=attempts,qualified=False))
        write('status.json',dict(status='budget_complete_not_qualified',step=step,qualified=False))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write('FAILED.json',dict(error=str(e),traceback=traceback.format_exc()));raise
