"""Train K19 retention controls from the verified R1 parent."""
import argparse
import copy
import hashlib
import json
import random
import re
import time
from pathlib import Path
import sys

from stage1_smoke import ROOT, load_policy
from stage1_protocol import balanced_indices
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from safetensors.torch import save_file, load_file

RUN = Path(__file__).resolve().parent
D = RUN.parent/"k18_training_controls_20261111_01"
sys.path.append(str(D))
from train_controls import PARENT, ORIG, read, write, digest, frozen_hash, snapshot

TEXT = {g:f"pick up the book and place it in the {g} compartment of the caddy" for g in ("left","right","front")}
ARMS=("freeze_kv","mu","d","both","replay","both_replay","replay_full","replay_soft")
LAMBDA=0.5


def open_data(path):
    return LeRobotDataset("local/k19",root=path,video_backend="pyav",
                          delta_timestamps={"action":[i/20 for i in range(50)]})


def setup(seed):
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    assert digest(PARENT/"model.safetensors")=="225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003"
    policy=load_policy(PARENT)
    pre,_=make_pre_post_processors(policy.config,pretrained_path=PARENT)
    return policy,pre


def capture(policy,batch,noise,t):
    recorded=[]
    handle=policy.model.action_out_proj.register_forward_hook(lambda module,args,out:recorded.append(out))
    try:
        loss,_=policy(batch,noise=noise,time=t,reduction="none")
    finally:
        handle.remove()
    assert len(recorded)==1
    return recorded[0][...,:7].float(),loss


def pre_pair(ds,pre,index):
    raw=torch.utils.data.default_collate(ds.__getitems__([index,index]))
    raw["task"]=[TEXT["left"],TEXT["right"]]
    return pre(raw)


def teacher():
    policy,pre=setup(20261111)
    policy.eval()
    path=ROOT/read(RUN/"config.json")["data"]["normalization_source"]
    conversion=read(path/"r0_conversion.json")
    assert set(conversion["spec"]["goals"])=={"left","right"} and conversion["spec"]["split"]=="train"
    ds=open_data(path)
    rows,offset=[],0
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for ep in conversion["episodes"]:
            for i,frac in enumerate((.2,.5,.8)):
                index=offset+int(ep["frames"]*frac)
                batch=pre_pair(ds,pre,index)
                g=torch.Generator(device="cpu").manual_seed(20261111+index)
                noise=torch.randn((1,50,32),generator=g).repeat(2,1,1).cuda()
                t=torch.full((2,),(.85,.55,.20)[i],device="cuda")
                val,_=capture(policy,batch,noise,t)
                assert torch.equal(batch["action"][0],batch["action"][1])
                rows.append({"index":index,"demo":ep["demo"],"goal":ep["goal"],"matched":int(ep["goal"]=="right"),
                             "batch":{k:v.detach().cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)},
                             "noise":noise.cpu(),"time":t.cpu(),"target":val.cpu()})
            offset+=ep["frames"]
            if len(rows)%30==0:
                print("TEACHER",len(rows),flush=True)
    assert len(rows)==240 and offset==len(ds)
    file=(RUN.parent/"k19_retention_20261111_01")/"teacher_bank.pt"
    temp=file.with_suffix(".tmp")
    torch.save(rows,temp)
    temp.replace(file)
    write((RUN.parent/"k19_retention_20261111_01")/"teacher_READY.json",{"pairs":len(rows),"sha256":digest(file),"source":str(path),
          "split_sha256":conversion["spec"]["split_sha256"],
          "left":sum(x["goal"]=="left" for x in rows),"right":sum(x["goal"]=="right" for x in rows)})


def cross_kv(name):
    m=re.fullmatch(r"model\.vlm_with_expert\.lm_expert\.layers\.(\d+)\.self_attn\.[kv]_proj\.weight",name)
    return bool(m and int(m[1])%2==1)


def penalty(response,target,valid):
    diff=response-target
    mu=(diff[0]+diff[1])*.5
    d=(diff[0]-diff[1])*.5
    count=valid.sum().clamp_min(1)*7
    return (mu.square()*valid[:,None]).sum()/count,(d.square()*valid[:,None]).sum()/count


def train(seed,arm,steps):
    policy,pre=setup(seed)
    config=read(RUN/"config.json")
    c=config["training"]
    directory=RUN/"runs"/f"{seed}_{arm}"
    directory.mkdir(parents=True)
    dataset=open_data(ROOT/config["data"]["train_output"])
    groups={"front":list(range(len(dataset)))}
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    assert [n for n,_ in named]==read(ORIG/"trainable_parameters.json")["names"]
    assert sum(cross_kv(n) for n,_ in named)==16
    if arm=="freeze_kv":
        for n,p in named:
            if cross_kv(n):
                p.requires_grad_(False)
    params=[p for _,p in named if p.requires_grad]
    frozen=frozen_hash(policy)
    optimizer=torch.optim.AdamW(params,lr=c["learning_rate"],betas=policy.config.optimizer_betas,
                               eps=policy.config.optimizer_eps,weight_decay=policy.config.optimizer_weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/c["warmup_steps"]))
    old=None
    schedule=None
    live_old=None
    live_groups=None
    live_labels=None
    if arm in ("replay_full","replay_soft"):
        path=ROOT/config["data"]["normalization_source"]
        live_old=open_data(path)
        live_groups={"left":[],"right":[]}
        start=0
        for episode in read(path/"r0_conversion.json")["episodes"]:
            end=start+episode["frames"]
            live_groups[episode["goal"]].extend(range(start,end))
            start=end
        assert start==len(live_old)
        live_labels={i:g for g,values in live_groups.items() for i in values}
    elif arm!="freeze_kv":
        marker=read((RUN.parent/"k19_retention_20261111_01")/"teacher_READY.json")
        assert digest((RUN.parent/"k19_retention_20261111_01")/"teacher_bank.pt")==marker["sha256"]
        old=torch.load((RUN.parent/"k19_retention_20261111_01")/"teacher_bank.pt",map_location="cpu",weights_only=True)
        left=[i for i,r in enumerate(old) if r["goal"]=="left"]
        right=[i for i,r in enumerate(old) if r["goal"]=="right"]
        assert len(left)==len(right)==120
        gen=random.Random(seed+7103)
        # Alternate true demonstrations from the two old goals.
        schedule=[gen.choice(left if k%2==0 else right) for k in range(steps)]
    policy.train()
    started=time.monotonic()
    history=[]
    with (directory/"train_steps.jsonl").open("x") as log:
        for k in range(steps):
            ids=balanced_indices(groups,c["samples_per_goal"],seed,k)
            raw=torch.utils.data.default_collate(dataset.__getitems__(ids))
            batch=pre(copy.deepcopy(raw))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                front,_=policy(batch,reduction="none")
                front=front.mean()
            front.backward()
            mu=d=bc=None
            live_ids=None
            if live_old is not None:
                live_ids=balanced_indices(live_groups,1,seed+7103,k)
                raw_old=torch.utils.data.default_collate(live_old.__getitems__(live_ids))
                # Keep labels paired with the data, regardless of sampler shuffle.
                raw_old["task"]=[TEXT[live_labels[i]] for i in live_ids]
                b=pre(raw_old)
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed*10+300000+k)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        per_old,_=policy(b,reduction="none")
                        bc=per_old.mean()
                (bc*(.35 if arm=="replay_soft" else 1.)).backward()
            elif old is not None:
                item=old[schedule[k]]
                old_batch={j:v.cuda() for j,v in item["batch"].items()}
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    v,loss=capture(policy,old_batch,item["noise"].cuda(),item["time"].cuda())
                mu,d=penalty(v,item["target"].cuda(),~old_batch["action_is_pad"][0])
                bc=loss[item["matched"]]
                if arm=="mu":
                    extra=LAMBDA*mu
                elif arm=="d":
                    extra=LAMBDA*d
                elif arm=="both":
                    extra=LAMBDA*(mu+d)
                elif arm=="replay":
                    extra=bc
                else:
                    extra=bc+LAMBDA*(mu+d)
                extra.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,policy.config.optimizer_grad_clip_norm,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            row={"step":k+1,"indices":ids,"front_loss":float(front.detach()),"grad_norm":float(norm),"lr":lr,
                 "mu_loss":None if mu is None else float(mu.detach()),"d_loss":None if d is None else float(d.detach()),
                 "old_supervised_loss":None if bc is None else float(bc.detach()),
                 "old_index":None if old is None else old[schedule[k]]["index"],
                 "live_old_indices":live_ids,
                 "seconds":time.monotonic()-started}
            history.append(row)
            log.write(json.dumps(row,allow_nan=False)+"\n")
            log.flush()
            if k+1 in (30,100,300,1000) or k+1==steps:
                snapshot(directory/f"{k+1:06d}",named,k+1,history,optimizer,scheduler)
            if k<3 or (k+1)%50==0:
                print("TRAIN",seed,arm,k+1,round(row["front_loss"],5),
                      row["mu_loss"],row["d_loss"],flush=True)
    assert frozen_hash(policy)==frozen
    if arm=="freeze_kv":
        with torch.no_grad():
            from safetensors import safe_open
            with safe_open(PARENT/"model.safetensors",framework="pt") as reference:
                assert all(torch.equal(p.detach().cpu(),reference.get_tensor(n)) for n,p in named if cross_kv(n))
    write(directory/"COMPLETE.json",{"seed":seed,"arm":arm,"step":steps,"seconds":time.monotonic()-started,
          "frozen_unchanged":True,"lambda":LAMBDA,"dynamic_old_weight":.35 if arm=="replay_soft" else
          (1. if arm=="replay_full" else None)})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("teacher","train"))
    p.add_argument("--seed",type=int,default=20261021)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--steps",type=int,default=1000)
    a=p.parse_args()
    if a.mode=="teacher":
        teacher()
    else:
        assert a.arm
        train(a.seed,a.arm,a.steps)

