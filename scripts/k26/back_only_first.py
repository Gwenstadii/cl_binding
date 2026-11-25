"""B0-Back: public initialization, first project task is Back, matched controls."""
import sys,os,json,random,time,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
from safetensors.torch import save_file
sys.path[:0]=['/home/lzx/cl_binding/.server/k19_retention_20261111_01','/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01','/home/lzx/cl_binding/.server/k25_back_retrain_20261124_01']
import train_k19 as old
import whole4_new as audit
from back_training_protocol import learning_rate,phase_ranges,sample_indices
from stage1_smoke import load_policy,find_model
from lerobot.policies import make_pre_post_processors
RUN=Path(__file__).resolve().parent
def write(name,v):audit.write(RUN/name,v)
def sha(path):return audit.digest(path)
def now():return datetime.now(timezone.utc).isoformat()

def evaluate(path,cases,name):
    # Only this runtime generates actions. R1 supplies unchanged frozen weights/processors;
    # ALL155 task-trainable tensors are overwritten by this public-init Back-only checkpoint.
    lab=audit.ref.old.original.k21.Lab(family='scene2',models=False)
    rt=audit.ref.old.original.k21.km.make_runtime(lab.ctx,{'FT4':str(old.PARENT)},['F'])
    audit.ref.old.base.sr.assign(type('Holder',(),{'rt':rt})(),'F',audit.ref.old.base.selected_weights(path))
    lab.rt=audit.ref.old.DualRuntime(rt,rt);lab.text={g:lab.ctx.all_texts[g] for g in audit.ref.ALL};lab.alpha=1.
    rows=[]
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),audit.ref.old.gzip.open(RUN/(name+'.jsonl.gz'),'xt') as log:
            for case in cases:
                r=audit.ref.episode(lab,case,'back','native')
                r['stable_released_final']=r['terminal']['back'] and sum(bool(m&8) for m in r['trace']['success'][-50:])>=45 and all(m!=3 for m in r['trace']['contacts'][-10:])
                log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();rows.append(r)
                print('B0_NATIVE',name,case['init'],r['terminal']['back'],r['stable_released_final'],flush=True)
    finally:lab.ctx.env.close()
    result=dict(n=20,terminal=sum(r['terminal']['back'] for r in rows),ever=sum(r['ever']['back'] for r in rows),stable_released_final=sum(r['stable_released_final'] for r in rows))
    write(name+'_COMPLETE.json',dict(**result,rows_sha256=sha(RUN/(name+'.jsonl.gz')),weights_sha256=sha(path)))
    return result

def main():
    assert not (RUN/'manifest.json').exists()
    torch.set_num_threads(2);seed=20261124;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    base=Path(find_model('lerobot/smolvla_base'));assert base.name=='smolvla_base'
    policy=load_policy() # No R1/K19 task-trained initialization is loaded for training.
    named={n:p for n,p in policy.named_parameters() if p.requires_grad};assert len(named)==155
    frozen=old.frozen_hash(policy);assert frozen=='0c0f563fdbc566d8f9d7dd542776678eabd0ed9eb112ee4a8e3e9f7437dc4f5e'
    pre,_=make_pre_post_processors(policy.config,pretrained_path=old.PARENT)
    dataset_path=Path('/home/lzx/cl_binding_stage2a/datasets/stage1_single_back_train_aligned_v1')
    conversion=old.read(dataset_path/'r0_conversion.json');assert conversion['spec']['goals']==['back'] and conversion['spec']['split']=='train'
    assert conversion['spec']['split_sha256']=='a55e1c1c8d0796329fc21b196b4d0a0a9348a2a63efebe86679a3a2a0e1b54f0'
    data=old.open_data(dataset_path);ranges,n=phase_ranges(conversion['episodes'],.30);assert n==len(data)==5849
    dev=audit.read(audit.RUN/'dev_cases.json')
    # audit.RUN is original archived whole4-new location, fixed on import.
    confirm=audit.read(Path('/home/lzx/cl_binding/.server/k25_back_stable_20261124_01/confirmation_cases.json'))
    normalizers={p.name:sha(p) for p in old.PARENT.iterdir() if p.is_file() and 'processor' in p.name}
    write('manifest.json',dict(code='B0-Back',started_at_utc=now(),base=str(base),base_sha256=sha(base/'model.safetensors'),
        source_sha256=sha(Path(__file__)),trainable_names=list(named),frozen_hash=frozen,
        project_training_goals=['back'],previous_project_task_checkpoints_loaded=False,
        normalization='same R1 processors/statistics for matched comparison; no R1 trainable weights used',normalizer_files=normalizers,
        seed=seed,training_budget=24000,batch=6,sampler='same3 whole-demo +3 final30%, episode-uniform',
        peak_lr=5e-5,floor_lr=5e-6,warmup=300,pretraining_limit='public VLA pretraining history not claimed Back-free',
        gate='native400; each dev and confirmation >=16/20 terminal and >=16/20 stable/no-continuous-two-finger-hold'))
    (RUN/'000000').mkdir();save_file({n:p.detach().cpu().contiguous() for n,p in named.items()},str(RUN/'000000/trainable.safetensors'))
    opt=torch.optim.AdamW(list(named.values()),lr=5e-5,betas=policy.config.optimizer_betas,eps=policy.config.optimizer_eps,weight_decay=policy.config.optimizer_weight_decay)
    start=time.monotonic();attempts=[];qualified=False
    with (RUN/'train.jsonl').open('x') as log:
        for k in range(24000):
            step=k+1;policy.train();ids=sample_indices(ranges,seed,k,6)
            batch=pre(torch.utils.data.default_collate(data.__getitems__(ids)))
            lr=learning_rate(step,24000,300,5e-5,5e-6)
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=policy(batch,reduction='none');loss=loss.mean()
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(named.values()),10.,error_if_nonfinite=True);opt.step()
            row=dict(step=step,loss=float(loss.detach()),indices=ids,lr=lr,grad_norm=float(norm),elapsed=time.monotonic()-start)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if step==1 or step%25==0:
                log.flush();write('status.json',dict(status='training',code='B0-Back',updated_at_utc=now(),**row));print('B0_TRAIN',row,flush=True)
            if step in [1000,2000,3000] or step%4000==0:
                folder=RUN/f'{step:06d}';folder.mkdir();assert old.frozen_hash(policy)==frozen
                weights=folder/'trainable.safetensors';save_file({n:p.detach().cpu().contiguous() for n,p in named.items()},str(weights))
                torch.save(dict(step=step,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),folder/'training_state.pt')
                write(f'{step:06d}/READY.json',dict(step=step,count=155,sha256=sha(weights),frozen_unchanged=True,base=str(base),base_sha256=sha(base/'model.safetensors'),project_training_goals=['back']))
                write('status.json',dict(status='native_evaluation',code='B0-Back',step=step))
                result=evaluate(weights,dev,f'dev_{step}');attempt=dict(step=step,development=result)
                if result['terminal']>=16 and result['stable_released_final']>=16:
                    second=evaluate(weights,confirm,f'confirm_{step}');attempt['confirmation']=second
                    if second['terminal']>=16 and second['stable_released_final']>=16:
                        attempts.append(attempt);write('QUALIFIED.json',dict(code=f'B0-Back-{step:06d}',step=step,weights=str(weights),sha256=sha(weights),attempts=attempts));qualified=True;break
                attempts.append(attempt);write('attempts.json',attempts)
    write('COMPLETE.json',dict(code='B0-Back',steps=step,qualified=qualified,attempts=attempts,finished_at_utc=now()))
    write('status.json',dict(status='qualified' if qualified else 'budget_complete_not_qualified',code='B0-Back',step=step,qualified=qualified))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        write('FAILED.json',dict(error=str(e),traceback=traceback.format_exc()));raise
