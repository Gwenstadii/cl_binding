"""K24 continuous 30→130 matched Front training window."""
import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file, save_file

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / "k24_update_feasibility_20261117_01"))
from update_probe import K18, K20, dot, pairs, project, response, set_weights
from train_weak import ORIG, ROOT, balanced_indices, cross_kv, open_data, read, setup, write

SEED, START, END = 20261021, 30, 130
ARMS = ("full", "protect", "matched", "freeze")
TOL = .0001


def frozen_block(name):
    return cross_kv(name) or (".mlp." in name and any(f".layers.{i}." in name for i in (4,5,6,7)))


def delta_norm(d):
    return float(dot(d,d).sqrt())


def run(arm):
    torch.set_num_threads(2)
    policy, pre = setup(SEED)
    config = read(K20/"config.json")
    ds = open_data(ROOT/config["data"]["train_output"])
    groups = {"front":list(range(len(ds)))}
    spec = config["training"]
    ckpt = K18/"train/checkpoints"/f"{START:06d}"
    policy.load_state_dict(load_file(str(ckpt/"model.safetensors")), strict=True)
    state = torch.load(ckpt/"training_state.pt", map_location="cpu", weights_only=False)
    named = [(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    assert [n for n,_ in named] == read(ORIG/"trainable_parameters.json")["names"]
    params = [p for _,p in named]
    blocked = [p for n,p in named if frozen_block(n)] if arm=="freeze" else []
    assert arm!="freeze" or len(blocked)==28
    opt = torch.optim.AdamW(params,lr=spec["learning_rate"])
    opt.load_state_dict(state["optimizer"])
    log = [json.loads(s) for s in (ORIG/"train_steps.jsonl").read_text().splitlines()]
    bank = pairs()
    ids = list(bank)
    affine = read(K20/"action_affine_0.json")
    off, scale = [torch.tensor(affine[k][:2],device="cuda") for k in ("offset","scale")]
    coef = torch.tensor(read(K20/"calibration.json")["report"]["coef"],device="cuda")
    targets = {}
    if arm=="matched":
        assert (RUN/"protect_COMPLETE.json").exists()
        targets = {r["step"]:r for r in (json.loads(x) for x in (RUN/"protect.jsonl").read_text().splitlines())}
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    total = time.monotonic()
    with (RUN/f"{arm}.jsonl").open("x",encoding="utf-8") as file:
        for k in range(START,END):
            batch_ids = balanced_indices(groups,6,SEED,k)
            assert batch_ids == log[k]["batch_indices"]
            batch = pre(copy.deepcopy(torch.utils.data.default_collate(ds.__getitems__(batch_ids))))
            initial = [p.detach().clone() for p in params]
            policy.train()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss = policy(batch,reduction="none")[0].mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params,policy.config.optimizer_grad_clip_norm,
                                                error_if_nonfinite=True)
            if arm=="full" or k==START:
                assert abs(float(loss.detach())-log[k]["loss"])<1e-6 and abs(float(gn)-log[k]["gradient_norm"])<1e-6
            for p in blocked:
                p.grad = None
            lr = opt.param_groups[0]["lr"]
            opt.step()
            delta = [p.detach().float()-w.float() for p,w in zip(params,initial,strict=True)]
            raw_norm = delta_norm(delta)
            fraction, accepted, dual, controlled = 1., True, [], None
            before, after, floors = [], [], []
            if arm=="protect" and k%3==0:
                controlled = ids[((k-START)//3)%12]
                pair = bank[controlled]
                policy.eval()
                set_weights(params,initial)
                f = response(policy,pair,off,scale,coef)[4:]
                old = f.detach()
                floor = torch.minimum(old, old.new_tensor(pair[0]["floor"]+pair[1]["floor"]))
                grads = []
                for j in range(4):
                    g = torch.autograd.grad(f[j],params,retain_graph=j<3,allow_unused=True)
                    grads.append(tuple(t.detach().float() if t is not None else torch.zeros_like(p,dtype=torch.float32)
                                       for t,p in zip(g,params,strict=True)))
                corrected, dual = project(delta,grads,old,floor)
                before, floors = old.tolist(), floor.tolist()
                accepted = False
                for fraction in (1.,.5,.25,.125):
                    set_weights(params,initial,[x*fraction for x in corrected])
                    with torch.no_grad():
                        checked = response(policy,pair,off,scale,coef)[4:]
                    if bool((checked>=floor-TOL).all()):
                        after,accepted = checked.tolist(),True
                        break
                if not accepted:
                    fraction, after = 0., before
                    set_weights(params,initial)
                del grads,corrected,g,f
            elif arm=="matched":
                target = targets[k+1]["actual_norm"]
                fraction = target/max(raw_norm,1e-15)
                set_weights(params,initial,[x*fraction for x in delta])
            actual_norm = delta_norm([p.detach().float()-w.float() for p,w in zip(params,initial,strict=True)])
            # Keep the AdamW moments of the actual Front gradient in every arm. Only
            # the applied weight step is modified; the next step re-evaluates it.
            opt.param_groups[0]["lr"] = spec["learning_rate"]*min(1.,(k+2)/spec["warmup_steps"])
            row = {"step":k+1,"indices":batch_ids,"lr":lr,"front_loss":float(loss.detach()),
                   "raw_norm":raw_norm,"actual_norm":actual_norm,"fraction":fraction,
                   "controlled":controlled,"accepted":accepted,"dual":dual,
                   "before":before,"after":after,"floors":floors,"seconds":time.monotonic()-total}
            file.write(json.dumps(row,allow_nan=False)+"\n")
            file.flush()
            if (k+1)%10==0:
                print(arm,row["step"],row["front_loss"],row["fraction"],flush=True)
            if k+1 in (100,END):
                path=RUN/f"w_{arm}_{k+1}.safetensors"
                save_file({n:p.detach().cpu().contiguous() for n,p in named},str(path))
    write(RUN/f"{arm}_COMPLETE.json",{"arm":arm,"start":START,"end":END,
          "sha256":sha256((RUN/f"w_{arm}_{END}.safetensors").read_bytes()).hexdigest()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=ARMS)
    run(p.parse_args().arm)
