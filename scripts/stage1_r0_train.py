"""Balanced old-task adaptation with complete, verified checkpoint state."""

import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from stage1_protocol import balanced_indices
from stage1_smoke import ROOT, CAMERAS, dump, load_policy


def frozen_hash(policy):
    import torch
    digest=hashlib.sha256()
    for name,param in policy.named_parameters():
        if not param.requires_grad:
            digest.update(name.encode())
            digest.update(param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def train(args,config):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies import make_pre_post_processors

    torch.set_num_threads(4)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    run=args.run_dir
    if run.exists() and not args.resume:raise FileExistsError(run)
    if args.resume and args.resume.resolve().parent.parent!=run.resolve():raise ValueError('Resume must belong to this run')
    run.mkdir(parents=True,exist_ok=bool(args.resume))
    data=ROOT/config['data']['train_output']
    conversion=json.loads((data/'r0_conversion.json').read_text())
    assert conversion['spec']['goals']==config['old_goals'] and conversion['spec']['split']=='train'
    policy=load_policy(args.resume)
    policy.config.n_action_steps=config['model']['n_action_steps']
    policy.config.num_steps=config['model']['flow_steps']
    dataset=LeRobotDataset('local/cl_binding_r0_train',root=data,video_backend='pyav',
            delta_timestamps={'action':[i/config['data']['fps'] for i in range(policy.config.chunk_size)]})
    if args.resume:
        pre,post=make_pre_post_processors(policy.config,pretrained_path=args.resume)
    else:
        pre,post=make_pre_post_processors(policy.config,dataset_stats=dataset.meta.stats)
    groups={g:[] for g in config['old_goals']};offset=0
    for episode in conversion['episodes']:
        groups[episode['goal']].extend(range(offset,offset+episode['frames']));offset+=episode['frames']
    assert offset==len(dataset)
    names=[name for name,p in policy.named_parameters() if p.requires_grad]
    assert not any('.vlm.' in name for name in names), 'VLM must be frozen'
    dump(run/'trainable_parameters.json',{'names':names,'count':sum(p.numel() for p in policy.parameters() if p.requires_grad)})
    original_frozen=frozen_hash(policy)
    params=[p for p in policy.parameters() if p.requires_grad]
    c=config['training']
    optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],betas=policy.config.optimizer_betas,
            eps=policy.config.optimizer_eps,weight_decay=policy.config.optimizer_weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.0,(step+1)/c['warmup_steps']))
    start_step=0
    if args.resume:
        saved=torch.load(args.resume/'training_state.pt',map_location='cpu',weights_only=False)
        assert saved['config']==config and saved['split_sha256']==conversion['spec']['split_sha256']
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        start_step=saved['step']
        assert original_frozen==saved['frozen_hash']
        dump(run/f'resume_{start_step:06d}.json',{'status':'pass','restored_step':start_step,
                'optimizer_state_entries':len(optimizer.state),'scheduler_last_epoch':scheduler.last_epoch})
        print('COMPLETE_TRAINING_STATE_RESTORED',start_step,flush=True)
    else:
        dump(run/'manifest.json',{'config':config,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'split_sha256':conversion['spec']['split_sha256'],'normalization_source':str(data),
              'frozen_hash':original_frozen,'purpose':'R0_development_not_qualified_until_holdout_passes'})
    maximum=args.max_steps or c['max_steps']
    if maximum<=start_step:raise ValueError('Target step must exceed resume step')
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic();last_step=start_step

    def checkpoint(step, raw_batch):
        policy.eval();policy.reset()
        frozen_now=frozen_hash(policy)
        assert frozen_now==original_frozen,'Frozen parameter changed'
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        np_rng=np.random.get_state();python_rng=random.getstate()
        folder=run/'checkpoints'/f'{step:06d}'
        if folder.exists():raise FileExistsError(folder)
        noise=torch.randn(len(raw_batch['task']),policy.config.chunk_size,policy.config.max_action_dim,
                generator=torch.Generator(device='cuda').manual_seed(config['seed']+step),device='cuda')
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            reference=policy.predict_action_chunk(pre(copy.deepcopy(raw_batch)),noise=noise).float().cpu()
        policy.save_pretrained(folder);pre.save_pretrained(folder);post.save_pretrained(folder)
        torch.save({'step':step,'config':config,'split_sha256':conversion['spec']['split_sha256'],
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'torch_rng':cpu_rng,'cuda_rng':cuda_rng,'numpy_rng':np_rng,'python_rng':python_rng,
            'frozen_hash':frozen_now},folder/'training_state.pt')
        restored=load_policy(folder);restored.eval()
        pre2,post2=make_pre_post_processors(restored.config,pretrained_path=folder)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            actual=restored.predict_action_chunk(pre2(copy.deepcopy(raw_batch)),noise=noise).float().cpu()
            physical=post2(actual.to('cuda')).float().cpu()
        assert torch.allclose(reference,actual,atol=1e-3,rtol=1e-3),'Checkpoint reload differs'
        assert physical.shape[-1]==7 and torch.isfinite(physical).all()
        delta=(reference-actual).abs().max().item()
        del restored,pre2,post2
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng);random.setstate(python_rng)
        dump(folder/'CHECKPOINT_COMPLETE.json',{'status':'pass','step':step,'reload_max_difference':delta,
             'frozen_hash_unchanged':True,'optimizer_and_rng_saved':True})
        policy.train();torch.cuda.empty_cache()
        print(json.dumps({'checkpoint_complete':str(folder),'step':step,'reload_difference':delta}),flush=True)

    for step in range(start_step,maximum):
        ids=balanced_indices(groups,c['samples_per_goal'],config['seed'],step)
        samples=dataset.__getitems__(ids)
        raw=torch.utils.data.default_collate(samples)
        batch=pre(copy.deepcopy(raw));policy.train();optimizer.zero_grad(set_to_none=True)
        tick=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            per_record,_=policy(batch,reduction='none');loss=per_record.mean()
        assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,policy.config.optimizer_grad_clip_norm,error_if_nonfinite=True)
        assert norm.item()>0
        optimizer.step();scheduler.step();last_step=step+1
        row={'step':last_step,'loss':loss.item(),'gradient_norm':norm.item(),'lr':optimizer.param_groups[0]['lr'],
             'step_seconds':time.monotonic()-tick,'batch_indices':ids,'samples_per_goal':c['samples_per_goal']}
        with (run/'train_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if last_step<=3 or last_step%c['log_every']==0:print(json.dumps(row),flush=True)
        stop=(run/'STOP_REQUESTED.json').exists() or time.monotonic()-started>c['walltime_hours']*3600
        if last_step%c['save_every']==0 or last_step==maximum or stop:checkpoint(last_step,raw)
        if stop:break
    dump(run/f'training_finished_{last_step:06d}.json',{'status':'complete','step':last_step,'target':maximum,
         'seconds':time.monotonic()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
         'frozen_parameters_unchanged':frozen_hash(policy)==original_frozen})
    print('TRAINING_PROCESS_COMPLETE',last_step,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/stage1_r0.json')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--max-steps',type=int)
    a=p.parse_args();train(a,json.loads(a.config.read_text()))
