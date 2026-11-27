"""Compare raw gradient conflict with realized AdamW steps at the same anchors."""
import argparse
import copy
import json
import random

import torch
from safetensors.torch import load_file
from train_k19 import RUN,D,ORIG,PARENT,ROOT,setup,read,open_data,balanced_indices,capture,penalty,cross_kv

K18=RUN.parent/"k18_training_cause_20261111_01"
ANCHORS=(0,30,100,300)
OFFSETS=(0,13,41,79,127,193,271,389)


def family(n):
    if cross_kv(n):
        return "cross_kv"
    if ".self_attn." in n:
        return "other_attention"
    if ".lm_expert." in n:
        return "mlp_norm"
    if n.startswith("model.state_proj."):
        return "state"
    if n.startswith("model.action_out_proj."):
        return "output"
    return "action_time"


def opt_for(policy,params,lr):
    c=policy.config
    return torch.optim.AdamW(params,lr=lr,betas=c.optimizer_betas,eps=c.optimizer_eps,
                             weight_decay=c.optimizer_weight_decay)


def old_losses(policy,items):
    metrics=[]
    for item in items:
        b={k:v.cuda() for k,v in item["batch"].items()}
        with torch.autocast("cuda",dtype=torch.bfloat16):
            response,bc=capture(policy,b,item["noise"].cuda(),item["time"].cuda())
        mu,d=penalty(response,item["target"].cuda(),~b["action_is_pad"][0])
        metrics.append((bc[item["matched"]],mu,d))
    return {k:(x+y)*.5 for k,x,y in zip(("old_bc","mu","d"),*metrics,strict=True)}


@torch.no_grad()
def geometry(first,second,families):
    data={}
    for k in ("all",*dict.fromkeys(families)):
        vals=[(a,b) for a,b,g in zip(first,second,families,strict=True) if k=="all" or g==k]
        a2,b2,dot=[sum(float(term(x.float(),y.float())) for x,y in vals) for term in (
            lambda x,y:x.square().sum(),lambda x,y:y.square().sum(),lambda x,y:(x*y).sum())]
        denom=(a2*b2)**.5
        data[k]={"dot":dot,"cosine":dot/denom if denom>1e-16 else None,
                 "first_l2":a2**.5,"second_l2":b2**.5}
    return data


def run(steps):
    config=read(RUN/"config.json")
    seed=config["seed"]
    policy,pre=setup(seed)
    initial_rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    policy.train()
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    assert [n for n,_ in named]==read(ORIG/"trainable_parameters.json")["names"]
    params=[p for _,p in named]
    groups=[family(n) for n,_ in named]
    ds=open_data(ROOT/config["data"]["train_output"])
    ids={"front":list(range(len(ds)))}
    old=torch.load(RUN/"teacher_bank.pt",map_location="cpu",weights_only=True)
    pool={g:[r for r in old if r["goal"]==g] for g in ("left","right")}
    historic=[json.loads(l) for l in (ORIG/"train_steps.jsonl").read_text().splitlines()]
    for step in steps:
        ckpt=PARENT if step==0 else K18/"train/checkpoints"/f"{step:06d}"
        policy.load_state_dict(load_file(str(ckpt/"model.safetensors")),strict=True)
        baseline=[p.detach().clone() for p in params]
        if step:
            state=torch.load(ckpt/"training_state.pt",map_location="cpu",weights_only=False)
            history=state["optimizer"]
            cpu,cuda=state["torch_rng"],state["cuda_rng"]
            lr=history["param_groups"][0]["lr"]
        else:
            history=None
            cpu,cuda=initial_rng
            lr=config["training"]["learning_rate"]/config["training"]["warmup_steps"]
        with (RUN/f"gradient_{step:06d}.jsonl").open("x") as out:
            for offset in OFFSETS:
                with torch.no_grad():
                    for p,w in zip(params,baseline,strict=True):
                        p.copy_(w)
                if offset==0:
                    torch.set_rng_state(cpu)
                    torch.cuda.set_rng_state_all(cuda)
                else:
                    torch.manual_seed(seed+1000*step+offset)
                batch_ids=balanced_indices(ids,6,seed,step+offset)
                batch=pre(copy.deepcopy(torch.utils.data.default_collate(ds.__getitems__(batch_ids))))
                policy.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    vec,_=policy(batch,reduction="none")
                    loss=vec.mean()
                loss.backward()
                full_loss=float(loss.detach())
                front=[p.grad.detach().clone() for p in params]
                norm=float(torch.nn.utils.clip_grad_norm_(params,policy.config.optimizer_grad_clip_norm,error_if_nonfinite=True))
                clipped=[p.grad.detach().clone() for p in params]
                if offset==0:
                    r=historic[step]
                    assert r["batch_indices"]==batch_ids
                    assert abs(full_loss-r["loss"])<1e-6 and abs(norm-r["gradient_norm"])<1e-6,(step,full_loss,norm,r)
                gen=random.Random(328093+step*100+offset)
                pair=[gen.choice(pool[g]) for g in ("left","right")]
                previous=old_losses(policy,pair)
                pre_val={k:float(v.detach()) for k,v in previous.items()}
                align={}
                for k in ("mu","d"):
                    g=torch.autograd.grad(previous[k],params,retain_graph=True)
                    align[k]=geometry(front,g,groups)
                    del g
                old_gradient=torch.autograd.grad(previous["old_bc"],params)
                align["old_bc"]=geometry(front,old_gradient,groups)
                del previous
                updates={}
                for variant in ("actual","zero","fresh"):
                    with torch.no_grad():
                        for p,w in zip(params,baseline,strict=True):
                            p.copy_(w)
                    optimizer=opt_for(policy,params,lr)
                    if variant!="fresh" and history is not None:
                        optimizer.load_state_dict(copy.deepcopy(history))
                    for p,g in zip(params,clipped,strict=True):
                        p.grad=torch.zeros_like(p) if variant=="zero" else g
                    optimizer.step()
                    delta=[p.detach().float()-w.float() for p,w in zip(params,baseline,strict=True)]
                    motion=geometry(delta,old_gradient,groups)
                    del delta
                    with torch.no_grad():
                        values=old_losses(policy,pair)
                    updates[variant]={"old_bc_linear":motion,
                                      "actual_loss_delta":{k:float(v)-pre_val[k] for k,v in values.items()}}
                    del optimizer
                out.write(json.dumps({"step":step,"offset":offset,"front_ids":batch_ids,
                                     "old_indices":[r["index"] for r in pair],
                                     "front_loss":full_loss,"grad_norm":norm,"pre_old_losses":pre_val,
                                     "raw_front_alignment":align,"updates":updates},allow_nan=False)+"\n")
                out.flush()
                print("GRADIENT",step,offset,align["old_bc"]["all"]["cosine"],flush=True)
                del front,clipped,old_gradient


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--steps",nargs="+",type=int,default=ANCHORS)
    run(p.parse_args().steps)
