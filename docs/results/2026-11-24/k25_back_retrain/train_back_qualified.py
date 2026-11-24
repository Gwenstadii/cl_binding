"""Fresh Back continuation from the frozen three-task parent; qualification pending."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import file_digest
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import sys
from time import monotonic
import traceback

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from back_training_protocol import learning_rate, phase_ranges, sample_indices

RUN=Path(__file__).resolve().parent
K19=RUN.parent/'k19_retention_20261111_01'
sys.path.insert(0,str(K19))
import train_k19 as original


def sha(path):
    with Path(path).open('rb') as f:return file_digest(f,'sha256').hexdigest()


def now():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def train(config_path):
    c=json.loads(config_path.read_text(encoding='utf-8'))
    out=RUN/'training';out.mkdir()
    started=monotonic()
    write(RUN/'status.json',dict(status='loading',pid=os.getpid(),started_at_utc=now(),config_sha256=sha(config_path)))
    assert c['parent_sha256']=='f8bd197dc20886ee14873429c90a7b51ea00ccc2656c28fc542b625b1261772c'
    parent=Path(c['parent']);weights=parent/'trainable.safetensors'
    assert sha(weights)==c['parent_sha256']==original.read(parent/'READY.json')['sha256']
    policy,pre=original.setup(c['seed'])
    named={n:p for n,p in policy.named_parameters() if p.requires_grad}
    saved=load_file(str(weights));assert named.keys()==saved.keys() and len(named)==155
    with torch.no_grad():
        for name,p in named.items():
            assert p.dtype==saved[name].dtype
            p.copy_(saved[name])
    del saved
    frozen=original.frozen_hash(policy)
    assert frozen==c['frozen_hash']
    data_path=Path(c['dataset']);conversion=original.read(data_path/'r0_conversion.json')
    assert conversion['spec']['split']=='train' and conversion['spec']['goals']==['back']
    assert conversion['spec']['split_sha256']==c['dataset_split_sha256']
    assert conversion['spec']['action_alignment']=='post_action_obs_to_next_action'
    assert len(conversion['episodes'])==40
    ranges,frames=phase_ranges(conversion['episodes'],c['tail_fraction'])
    data=original.open_data(data_path);assert len(data)==frames
    assert c['batch']==6 and c['steps']==24000 and c['old_replay'] is False
    assert c['normalization']=='inherit R1 via K19 unchanged'
    files=[Path(__file__),RUN/'back_training_protocol.py',Path(original.__file__),
           Path(inspect.getfile(original.setup)),Path(inspect.getfile(original.frozen_hash)),
           Path(inspect.getfile(original.load_policy)),Path(inspect.getfile(policy.__class__))]
    normalizers={p.name:sha(p) for p in original.PARENT.iterdir() if p.is_file() and ('processor' in p.name or 'config' in p.name)}
    dtypes=dict(Counter(str(p.dtype) for p in named.values()))
    manifest=dict(created_at_utc=now(),source_parent_commit=c['source_parent_commit'],config=c,
        source_files={str(p):sha(p) for p in files},config_sha256=sha(config_path),
        normalizer_files=normalizers,trainable_names=list(named),trainable_count=len(named),
        trainable_parameters=sum(p.numel() for p in named.values()),trainable_dtypes=dtypes,
        dataset_conversion_sha256=sha(data_path/'r0_conversion.json'),episodes=40,frames=frames,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','safetensors','lerobot']},
        python=sys.version,executable=sys.executable,gpu=torch.cuda.get_device_name(),
        qualification='not assessed by loss or training completion; native400-control Back final placement requires independent rollout acceptance')
    write(out/'manifest.json',manifest)
    print('BACK_RETRAIN_READY',dict(frames=frames,dtypes=dtypes,parameters=manifest['trainable_parameters']),flush=True)
    params=list(named.values());pc=policy.config
    opt=torch.optim.AdamW(params,lr=c['peak_lr'],betas=pc.optimizer_betas,eps=pc.optimizer_eps,weight_decay=pc.optimizer_weight_decay)
    policy.train();compute_start=monotonic();saves=set(c['save_steps'])
    with (out/'train.jsonl').open('x',encoding='utf-8') as log:
        for k in range(c['steps']):
            step=k+1
            ids=sample_indices(ranges,c['seed'],k,c['batch'])
            lr=learning_rate(step,c['steps'],c['warmup'],c['peak_lr'],c['floor_lr'])
            for group in opt.param_groups:group['lr']=lr
            batch=pre(torch.utils.data.default_collate(data.__getitems__(ids)))
            assert batch['action'].shape[0]==c['batch']
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                per_sample,_=policy(batch,reduction='none')
                assert per_sample.shape==(c['batch'],)
                loss=per_sample.mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
            opt.step()
            elapsed=monotonic()-compute_start
            row=dict(step=step,loss=float(loss.detach()),uniform_loss=float(per_sample[:3].mean().detach()),
                terminal_phase_loss=float(per_sample[3:].mean().detach()),grad_norm=float(norm),lr=lr,indices=ids,
                train_elapsed_seconds=elapsed,seconds_per_step=elapsed/step)
            log.write(json.dumps(row,allow_nan=False)+'\n')
            if step==1 or step%25==0:
                log.flush()
                write(RUN/'status.json',dict(status='training',pid=os.getpid(),updated_at_utc=now(),**row,
                    target_steps=c['steps'],estimated_remaining_seconds=(c['steps']-step)*elapsed/step))
                print('BACK_RETRAIN',row,flush=True)
            if step in saves:
                destination=out/f'{step:06d}';destination.mkdir()
                assert original.frozen_hash(policy)==frozen
                path=destination/'trainable.safetensors'
                tensors={n:p.detach().cpu().contiguous() for n,p in named.items()}
                save_file(tensors,str(path))
                restored=load_file(str(path));assert all(torch.equal(restored[n],v) for n,v in tensors.items())
                del restored,tensors
                state_path=destination/'training_state.pt'
                torch.save(dict(step=step,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                    config=c,config_sha256=sha(config_path)),state_path)
                write(destination/'READY.json',dict(step=step,count=len(named),sha256=sha(path),
                    training_state_sha256=sha(state_path),parent_sha256=c['parent_sha256'],frozen_unchanged=True,
                    config_sha256=sha(config_path),qualified=False))
                print('BACK_RETRAIN_CHECKPOINT',step,flush=True)
    assert original.frozen_hash(policy)==frozen
    for name,value in manifest['source_files'].items():assert sha(name)==value,name
    for name,value in normalizers.items():assert sha(original.PARENT/name)==value,name
    complete=dict(status='training_complete_qualification_pending',steps=c['steps'],finished_at_utc=now(),
        wall_seconds=monotonic()-started,compute_seconds=monotonic()-compute_start,
        frozen_unchanged=True,qualified=False,saved_steps=sorted(saves),config_sha256=sha(config_path))
    write(out/'COMPLETE.json',complete);write(RUN/'status.json',complete)
    print('BACK_RETRAIN_COMPLETE',complete,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=RUN/'config.json');a=parser.parse_args()
    try:train(a.config)
    except BaseException as error:
        write(RUN/'FAILED.json',dict(failed_at_utc=now(),type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        traceback.print_exc();raise
