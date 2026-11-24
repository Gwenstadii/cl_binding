"""Read-only model diagnosis on a fixed sample; no optimizer or qualification rollouts."""
import copy
import gc
import json
import random
import sys
from pathlib import Path

ROOT=Path('/home/lzx/cl_binding')
sys.path.insert(0,str(ROOT/'scripts'))
from stage1_smoke import load_policy, dump
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors

torch.set_num_threads(4)
seed=20261021
goals=['front','left','right']
output=ROOT/'.server/r0_aligned_v2_20261022_01/offline_action_diagnostic_20261022.json'
if output.exists():raise FileExistsError(output)
datasets={};batches={};selection={}
for split in ('train','validation'):
    root=ROOT/f'datasets/stage1_r0_{split}_aligned_v2'
    ds=LeRobotDataset(f'local/cl_binding_r0_{split}',root=root,video_backend='pyav',
                     delta_timestamps={'action':[i/20 for i in range(50)]})
    datasets[split]=ds
    conversion=json.loads((root/'r0_conversion.json').read_text())
    groups={g:[] for g in goals};offset=0
    for episode in conversion['episodes']:
        groups[episode['goal']].extend(range(offset,offset+episode['frames']));offset+=episode['frames']
    batches[split]={};selection[split]={}
    for goal in goals:
        ids=random.Random(seed+goals.index(goal)).sample(groups[goal],24)
        selection[split][goal]=ids
        batches[split][goal]=[torch.utils.data.default_collate(ds.__getitems__(ids[i:i+6])) for i in range(0,24,6)]
report={'status':'running','purpose':'exploratory_offline_diagnosis_not_R0_qualification',
        'selection':selection,'rows':[],'samples_per_goal_per_split':24,'dataset_alignment':'post_action_obs_to_next_action',
        'source_script':'docs/handoff/2026-10-21/证据/r0_20261021_174358/offline_action_diagnostic.py',
        'metric_scope':'single fixed noise seed; physical action MAE on nonpadded targets; not rollout success'}
for label,checkpoint in [('base',None),('003000',ROOT/'outputs/r0/r0_aligned_v2_20261022_01/checkpoints/003000'),
                         ('006000',ROOT/'outputs/r0/r0_aligned_v2_20261022_01/checkpoints/006000')]:
    policy=load_policy(checkpoint);policy.eval();policy.config.num_steps=10
    if checkpoint:
        pre,post=make_pre_post_processors(policy.config,pretrained_path=checkpoint)
    else:
        pre,post=make_pre_post_processors(policy.config,dataset_stats=datasets['train'].meta.stats)
    for split in batches:
        for goal in goals:
            sums={k:np.zeros(7,dtype=np.float64) for k in ('mae_first1','mae_first10','clip_fraction_first10')}
            counts={k:0 for k in sums};losses=[]
            for index,raw in enumerate(batches[split][goal]):
                torch.manual_seed(seed+index)
                batch=pre(copy.deepcopy(raw))
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,_=policy(batch,reduction='none')
                    noise=torch.randn(6,50,policy.config.max_action_dim,device='cuda',
                        generator=torch.Generator(device='cuda').manual_seed(seed+index))
                    predicted=post(policy.predict_action_chunk(pre(copy.deepcopy(raw)),noise=noise)).float().cpu()
                losses.extend(loss.float().cpu().tolist())
                target=raw['action'].cpu();valid=~raw['action_is_pad'].cpu()
                for horizon,key in ((1,'mae_first1'),(10,'mae_first10')):
                    mask=valid[:,:horizon]
                    error=(predicted[:,:horizon]-target[:,:horizon]).abs()
                    sums[key]+=error[mask].double().sum(0).numpy();counts[key]+=int(mask.sum())
                mask=valid[:,:10]
                sums['clip_fraction_first10']+=(predicted[:,:10].abs()>1)[mask].double().sum(0).numpy()
                counts['clip_fraction_first10']+=int(mask.sum())
            row={'model':label,'split':split,'goal':goal,'flow_matching_loss_mean':float(np.mean(losses)),
                 **{key:(sums[key]/counts[key]).tolist() for key in sums}}
            report['rows'].append(row);dump(output,report);print(json.dumps(row),flush=True)
    del policy,pre,post;gc.collect();torch.cuda.empty_cache()
report['status']='complete';dump(output,report)
print('OFFLINE_ACTION_DIAGNOSTIC_COMPLETE',flush=True)
