"""K18-D matched Front training, with actual BF16-aware step norm control."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

from stage1_smoke import ROOT, load_policy
from stage1_protocol import balanced_indices
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from safetensors.torch import load_file, save_file

RUN = Path(__file__).resolve().parent
K18 = RUN.parent / "k18_training_cause_20261111_01"
PARENT = RUN.parent / "r1_left_right_20261104_01/train_r1/checkpoints/024000"
ORIG = RUN.parent / "r1_then_front_v1/train"
SEEDS = (20261021, 20261121, 20261122)
ARMS = ("full", "no_rot", "random", "norm")
SAVE = (30, 100, 300, 1000)
MAX_REL = .015


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as h:
        return hashlib.file_digest(h,"sha256").hexdigest()


def frozen_hash(policy):
    h = hashlib.sha256()
    for name,p in policy.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def group_tensors(params):
    return [tuple(p for p in params if p.dtype == dtype) for dtype in (torch.float32,torch.bfloat16)]


def flatten(params):
    return torch.cat([p.detach().view(-1) for p in params])


def sqnorm(a,b):
    return (a.float()-b.float()).square().sum()


@torch.no_grad()
def actual_norm(old, proposed):
    return float((sqnorm(old[0],proposed[0])+sqnorm(old[1],proposed[1])).sqrt())


@torch.no_grad()
def matched_update(old, proposed, target, mask_seed):
    raw_norm = actual_norm(old, proposed)
    assert target > 0 and raw_norm > 0
    if abs(raw_norm/target-1) <= MAX_REL/2:
        return proposed,raw_norm,raw_norm,1.,1.
    f32, bf16 = old[0], old[1].float()
    d32, d16 = proposed[0]-f32, proposed[1].float()-bf16
    rng = torch.Generator(device="cuda").manual_seed(mask_seed)
    u = torch.rand(bf16.shape,generator=rng,device="cuda")
    pos, neg = torch.full_like(old[1],float("inf")), torch.full_like(old[1],-float("inf"))
    def candidate(scale):
        x = f32 + scale*d32
        desired = bf16 + scale*d16
        rounded = desired.to(torch.bfloat16)
        residue = desired - rounded.float()
        neighbour = torch.nextafter(rounded,torch.where(residue>0,pos,neg))
        jump = (neighbour.float()-rounded.float()).abs()
        y = torch.where(u*jump<residue.abs(),neighbour,rounded)
        return (x,y),actual_norm(old,(x,y))
    lo, hi = 0., max(2.,2*target/raw_norm)
    scale = target/raw_norm
    best = None
    for _ in range(12):
        pair,n = candidate(scale)
        err = abs(n/target-1)
        if best is None or err < best[0]:
            best = err,pair,n,scale
        if err <= MAX_REL/2:
            break
        if n < target:
            lo = scale
        else:
            hi = scale
        scale = (lo+hi)/2
    err,pair,n,scale = best
    # BF16 rounding is discrete. Retain the closest realized norm and log misses.
    dot = ((pair[0]-old[0])*d32).sum()+((pair[1].float()-bf16)*d16).sum()
    return pair,n,raw_norm,scale,float(dot)/(n*raw_norm)


@torch.no_grad()
def set_weights(groups, values):
    for params,flat in zip(groups,values,strict=True):
        parts = flat.split([p.numel() for p in params])
        torch._foreach_copy_(params,[a.view_as(p) for a,p in zip(parts,params,strict=True)])


def random_mask(shape,seed):
    rng=torch.Generator(device="cuda").manual_seed(seed)
    order=torch.rand((*shape,7),device="cuda",generator=rng).topk(3,dim=-1).indices
    keep=torch.ones((*shape,7),device="cuda",dtype=torch.bool)
    return keep.scatter_(-1,order,False)


def snapshot(folder,named,step,history,optimizer,scheduler):
    folder.mkdir(parents=True)
    state={name:p.detach().cpu().contiguous() for name,p in named}
    file=folder/"trainable.safetensors"
    save_file(state,str(file))
    saved=load_file(str(file))
    assert saved.keys()==state.keys() and all(torch.equal(p,saved[n]) for n,p in state.items())
    if step == 1000:
        torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                    "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                    "step":step},folder/"optimizer.pt")
    write(folder/"READY.json",{"step":step,"sha256":digest(file),"count":len(state),
                              "training_rows":len(history)})


def train(seed,arm,maximum):
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    directory=RUN/"runs"/f"{seed}_{arm}"
    directory.mkdir(parents=True)
    config=read(RUN/"config.json")
    spec=config["training"]
    data=ROOT/config["data"]["train_output"]
    conversion=read(data/"r0_conversion.json")
    assert conversion["spec"]["goals"]==["front"] and conversion["spec"]["action_offset"]==1
    assert digest(PARENT/"model.safetensors")=="225322ba28d0ad152fcced19950c43e3e40a435f1835cf64a664226da7830003"
    policy=load_policy(PARENT)
    policy.config.n_action_steps=config["model"]["n_action_steps"]
    policy.config.num_steps=config["model"]["flow_steps"]
    dataset=LeRobotDataset("local/cl_binding_r0_train",root=data,video_backend="pyav",
                           delta_timestamps={"action":[i/config["data"]["fps"] for i in range(50)]})
    pre,_=make_pre_post_processors(policy.config,pretrained_path=PARENT)
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    assert [n for n,_ in named]==read(ORIG/"trainable_parameters.json")["names"]
    params=[p for _,p in named]
    groups=group_tensors(params)
    assert sum(map(len,groups))==len(params)==155
    expected_frozen=frozen_hash(policy)
    optimizer=torch.optim.AdamW(params,lr=spec["learning_rate"],betas=policy.config.optimizer_betas,
                               eps=policy.config.optimizer_eps,weight_decay=policy.config.optimizer_weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/spec["warmup_steps"]))
    available={"front":list(range(len(dataset)))}
    targets=None
    if arm=="norm":
        target_dir=RUN/"runs"/f"{seed}_no_rot"
        assert (target_dir/"COMPLETE.json").exists()
        targets=[json.loads(line) for line in (target_dir/"train_steps.jsonl").read_text().splitlines()]
        assert len(targets)==maximum
    history=[]
    policy.train()
    started=time.monotonic()
    with (directory/"train_steps.jsonl").open("x") as log:
        for step in range(1,maximum+1):
            ids=balanced_indices(available,spec["samples_per_goal"],seed,step-1)
            raw=torch.utils.data.default_collate(dataset.__getitems__(ids))
            batch=pre(copy.deepcopy(raw))
            keep=None
            if arm=="no_rot":
                keep=torch.ones((*batch["action_is_pad"].shape,7),device="cuda",dtype=torch.bool)
                keep[...,3:6]=False
            elif arm=="random":
                keep=random_mask(batch["action_is_pad"].shape,seed*100000+step)
            valid_positions=int((~batch["action_is_pad"]).sum())
            if keep is not None:
                assert int((keep & ~batch["action_is_pad"][...,None]).sum())==4*valid_positions
            old=tuple(map(flatten,groups))
            optimizer.zero_grad(set_to_none=True)
            original=policy.model.forward
            if keep is not None:
                def masked(*a,**kw):
                    losses=original(*a,**kw)
                    assert losses.shape[-1]==32
                    return torch.cat((losses[...,:7]*keep,losses[...,7:]),dim=-1)
                policy.model.forward=masked
            try:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    per_record,_=policy(batch,reduction="none")
                    loss=per_record.mean()
            finally:
                if keep is not None:
                    policy.model.forward=original
            assert torch.isfinite(loss)
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,policy.config.optimizer_grad_clip_norm,error_if_nonfinite=True)
            lr=optimizer.param_groups[0]["lr"]
            optimizer.step()
            proposed=tuple(map(flatten,groups))
            if targets is not None:
                t=targets[step-1]
                assert t["step"]==step and t["indices"]==ids and t["valid_positions"]==valid_positions
                target=t["update_l2"]
                matched,n,raw_norm,scale,cosine=matched_update(old,proposed,target,seed*100000+50000+step)
                set_weights(groups,matched)
            else:
                n=raw_norm=actual_norm(old,proposed)
                target,scale,cosine=None,1.,1.
            scheduler.step()
            row={"step":step,"indices":ids,"valid_positions":valid_positions,"loss":float(loss.detach()),
                 "grad_norm":float(grad),"lr":lr,"update_l2":n,"proposal_l2":raw_norm,"target_l2":target,
                 "scale":scale,"cosine":cosine,"seconds":time.monotonic()-started}
            history.append(row)
            log.write(json.dumps(row,allow_nan=False)+"\n")
            log.flush()
            del old,proposed,loss,per_record
            if targets is not None:
                del matched
            if step in SAVE or step==maximum:
                snapshot(directory/f"{step:06d}",named,step,history,optimizer,scheduler)
            if step<=3 or step%50==0:
                print("TRAIN",seed,arm,step,round(row["loss"],5),round(n,6),target,flush=True)
    assert frozen_hash(policy)==expected_frozen
    write(directory/"COMPLETE.json",{"seed":seed,"arm":arm,"step":maximum,"frozen_unchanged":True,
          "seconds":time.monotonic()-started})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--steps",type=int,default=1000)
    a=p.parse_args()
    train(a.seed,a.arm,a.steps)
