"""Train on Front with one-sided old-command transport margins."""
import argparse
import copy
import json
import random
import time

import torch
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.utils import constants
from stage2m_cross_attention import make_prefix
from train_weak import RUN, ROOT, ORIG, setup, read, write, open_data, frozen_hash, snapshot, balanced_indices

PERIOD=3
LAMBDA=1.0
BACK_STEPS=2


def generated(policy,batch,noise):
    model=policy.model
    with torch.no_grad():
        images,masks=policy.prepare_images(batch)
        pad,cache,_=make_prefix(policy,(images,masks,batch[constants.OBS_LANGUAGE_TOKENS],
                            batch[constants.OBS_LANGUAGE_ATTENTION_MASK],policy.prepare_state(batch)),
                            make_att_2d_masks)
    x=noise
    for k in range(10):
        with torch.set_grad_enabled(torch.is_grad_enabled() and k>=10-BACK_STEPS):
            t=torch.full((x.shape[0],),1.-.1*k,device=x.device,dtype=torch.float32)
            v=model.denoise_step(pad,copy.deepcopy(cache),x.clone(),t)
            x=x+(-.1)*v
    return x[:,:10,:7]


def progress_score(x,item,offset,scale,coef):
    cmd=(x[0,:,:2]*scale+offset).clamp(-1,1).sum(0)
    p=torch.tensor(item["pos"],device=x.device)
    c=torch.tensor(item["centers"],device=x.device)
    y=cmd@coef
    delta=torch.linalg.vector_norm(c-p,dim=-1)-torch.linalg.vector_norm(c-p-y,dim=-1)
    j=item["target"]
    others=torch.arange(len(delta),device=x.device)!=j
    return torch.stack([delta[j],delta[j]-delta[others].max()])


def train(seed,steps):
    policy,pre=setup(seed)
    c=read(RUN/"config.json")
    ds=open_data(ROOT/c["data"]["train_output"])
    groups={"front":list(range(len(ds)))}
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    assert [n for n,_ in named]==read(ORIG/"trainable_parameters.json")["names"]
    params=[p for _,p in named]
    frozen=frozen_hash(policy)
    spec=c["training"]
    optimizer=torch.optim.AdamW(params,lr=spec["learning_rate"],betas=policy.config.optimizer_betas,
                               eps=policy.config.optimizer_eps,weight_decay=policy.config.optimizer_weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda k:min(1.,(k+1)/spec["warmup_steps"]))
    bank=torch.load(RUN/"route_bank.pt",weights_only=True,map_location="cpu")
    pools={q:[x for x in bank if x["q"]==q] for q in ("A","B")}
    ref=read(RUN/"calibration.json")
    tensor=lambda x:torch.tensor(x,device="cuda",dtype=torch.float32)
    transform=read(RUN/"action_affine_0.json")
    offset,scale=tensor(transform["offset"][:2]),tensor(transform["scale"][:2])
    coef=tensor(ref["report"]["coef"])
    a,b=(pools[q][0] for q in "AB")
    policy.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for p in (a,b):
            fields={k:v.cuda() for k,v in p["batch"].items()}
            x=generated(policy,fields,p["noise"].cuda())
            scores=progress_score(x,p,offset,scale,coef)
            assert (scores-tensor(p["floor"])/.55).abs().max()<.0015, (p["init"],p["q"],scores,p["floor"])
    print("PARENT_PROXY_GATE_PASS",flush=True)
    directory=RUN/"runs"/f"{seed}_route"
    directory.mkdir(parents=True)
    started=time.monotonic()
    gen=random.Random(seed+8352)
    history=[]
    with (directory/"train_steps.jsonl").open("x") as handle:
        for k in range(steps):
            ids=balanced_indices(groups,spec["samples_per_goal"],seed,k)
            batch=pre(copy.deepcopy(torch.utils.data.default_collate(ds.__getitems__(ids))))
            optimizer.zero_grad(set_to_none=True)
            policy.train()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                v,_=policy(batch,reduction="none")
                front=v.mean()
            front.backward()
            route=None
            active=0
            if k%PERIOD==0:
                item=gen.choice(pools["AB"[(k//PERIOD)%2]])
                policy.eval()
                fields={j:v.cuda() for j,v in item["batch"].items()}
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    x=generated(policy,fields,item["noise"].cuda())
                score=progress_score(x,item,offset,scale,coef)
                gaps=(tensor(item["floor"])-score).relu()
                active=int((gaps>0).sum())
                route=float(gaps.detach().sum()/.01)
                if active:
                    (LAMBDA*gaps.sum()/.01).backward()
            norm=float(torch.nn.utils.clip_grad_norm_(params,policy.config.optimizer_grad_clip_norm,error_if_nonfinite=True))
            lr=optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            row={"step":k+1,"indices":ids,"front_loss":float(front.detach()),"route_loss":route,
                 "active_terms":active,"lr":lr,"grad_norm":norm,"seconds":time.monotonic()-started}
            history.append(row)
            handle.write(json.dumps(row,allow_nan=False)+"\n")
            handle.flush()
            if k+1 in (30,100,300,1000) or k+1==steps:
                snapshot(directory/f"{k+1:06d}",named,k+1,history,optimizer,scheduler)
            if k<3 or (k+1)%50==0:
                print("TRAIN",seed,k+1,row["front_loss"],route,flush=True)
    assert frozen_hash(policy)==frozen
    write(directory/"COMPLETE.json",{"seed":seed,"arm":"route","step":steps,"lambda":LAMBDA,
          "every":PERIOD,"backprop_euler_steps":BACK_STEPS,"frozen_unchanged":True})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=20261021)
    p.add_argument("--steps",type=int,default=1000)
    a=p.parse_args()
    train(a.seed,a.steps)
