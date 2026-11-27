"""K24 continuous-window parameter and action-response counterfactuals."""
import argparse
from collections import defaultdict
import copy
from hashlib import file_digest
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

RUN = Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/"k20_route_margin_20261112_01"))
from eval_k20 import INSTR, K19, Lab5, episode, make_bank, read, reset, responses, write
from train_weak import cross_kv

TRAIN = ("full","protect","matched","freeze")
HYBRIDS = ("anchor","only_kv","only_cross","only_mlp","only_other","remove_kv",
           "remove_cross","remove_mlp","remove_other")
CORE = (*TRAIN, "anchor","remove_cross","remove_mlp")
PATCHES = {
    "repair_mu":("F","R","F","F"),
    "repair_d":("F","F","R","R"),
    "repair_mu_d":("F","R","R","R"),
    "reverse_mu":("R","F","R","R"),
    "reverse_d":("R","R","F","F"),
    "repair_parallel":("F","F","R","F"),
    "reverse_parallel":("R","R","F","R"),
}
TIME_ARMS = ("remove_time_out","repair_mu","reverse_mu")
WINDOW = 6


def group(name):
    if cross_kv(name):
        return "kv"
    if ".lm_expert.layers." in name and ".self_attn." in name:
        i=int(name.split(".layers.")[1].split(".")[0])
        return "cross_other" if i%2 else "self_attention"
    if ".lm_expert.layers." in name and ".mlp." in name:
        return "mlp"
    if ".lm_expert.layers." in name:
        return "expert_norm"
    if name.startswith("model.action_out_proj."):
        return "action_output"
    if name.startswith("model.state_proj."):
        return "state"
    return "action_time"


SOURCE = {"only_kv":{"kv"},"only_cross":{"kv","cross_other"},"only_mlp":{"mlp"},
          "only_other":{"self_attention","expert_norm","action_output","state","action_time"}}


def load_weights(arm, step=130):
    path = RUN/f"w_{arm}_{step}.safetensors"
    if arm in TRAIN and step==130:
        with path.open("rb") as handle:
            assert file_digest(handle,"sha256").hexdigest()==read(RUN/f"{arm}_COMPLETE.json")["sha256"]
    return load_file(str(path))


def assign(lab, role, values):
    selected = {n:p for n,p in lab.rt.models[role].named_parameters() if p.requires_grad}
    assert len(selected)==155 and values.keys()==selected.keys()
    with torch.no_grad():
        for n,p in selected.items():
            assert values[n].dtype==p.dtype
            p.copy_(values[n])
    lab.rt.models[role].eval()


def prepare():
    whole = load_weights("full")
    source = load_file(str(RUN.parent/"k18_training_cause_20261111_01/train/checkpoints/000030/model.safetensors"))
    anchor = {n:source[n] for n in whole}
    save_file(anchor,str(RUN/"w_anchor_130.safetensors"))
    for only, groups in SOURCE.items():
        for arm, match in ((only,True),(only.replace("only_","remove_"),False)):
            save_file({n:(whole[n] if (group(n) in groups)==match else anchor[n]) for n in whole},
                      str(RUN/f"w_{arm}_130.safetensors"))
    norms = defaultdict(list)
    for n in whole:
        norms[group(n)].append(float((whole[n].float()-anchor[n].float()).square().sum()))
    write(RUN/"blocks.json",{k:{"n":len(v),"delta_norm":sum(v)**.5} for k,v in norms.items()})


def probes():
    lab=Lab5()
    try:
        bank=make_bank(lab,range(20,40))
        for arm in (*TRAIN,*HYBRIDS):
            assign(lab,"F",load_weights(arm))
            write(RUN/f"probe_{arm}.json",responses(lab,bank,True))
            print("PROBE",arm,flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN/"probes_COMPLETE.json",{"arms":list((*TRAIN,*HYBRIDS))})


def measure_episode(lab, rec, q):
    start = 0 if q=="N" else 10*len(rec["controls"])
    captured = {}
    run=lab.execute

    def track(obs, u, offset, z0):
        pos=np.asarray(obs[lab.ctx.obj])[:2].copy()
        out, rows=run(obs,u,offset,z0)
        if offset==start:
            captured["position_xy_m"]=pos.tolist()
            captured["displacement_xy_m"]=(np.asarray(rows[-1]["obj"])[:2]-pos).tolist()
            captured["centers_xy_m"]={k:list(v["centre"][:2]) for k,v in lab.ctx.geometry.items()}
        return out,rows

    lab.execute=track
    try:
        result=episode(lab,rec,q,"confirm")
    finally:
        lab.execute=run
    assert captured
    return {**result,**captured}


def mix(lab, obs, q, seed, chunk, spec):
    rt = lab.rt
    base, mu_src, p_src, r_src = spec
    prompts = {k:rt.prefix(k[0],obs,INSTR[k[1]]) for k in ((role, g) for role in "FRO" for g in "AB")
               if k[0] in set(spec) or p_src!=r_src}
    sign = 1. if q=="A" else -1.

    def vel(x,t):
        values = {k:rt.query(k[0],v,x,t) for k,v in prompts.items()}
        a,b = (values[mu_src,g][:,:,:7].float() for g in "AB")
        d = {r:(values[r,"A"][:,:,:7].float()-values[r,"B"][:,:,:7].float())*.5
             for r in {p_src,r_src}}
        if p_src!=r_src:
            axis = (values["O","A"][:,:,:7].float()-values["O","B"][:,:,:7].float())*.5
            delta = d[p_src]-d[r_src]
            used = d[r_src] + (delta*axis).sum()/axis.square().sum().clamp_min(1e-12)*axis
        else:
            used = d[p_src]
        out=values[base,q].clone()
        out[:,:,:7]=(a+b)*.5+sign*used
        return out
    return rt.integrate(vel,rt.noise(seed,chunk),10)[...,:7]


def native(lane,lanes,patch):
    bank=read(K19/"holdout.json")["selected"]
    lab=Lab5()
    tag="patch" if patch else "native"
    try:
        if patch:
            # Preserve the verified parent already loaded into R, then borrow R for the protected endpoint.
            lab.rt.models["O"]=copy.deepcopy(lab.rt.models["R"])
            lab.rt.pre["O"]=lab.rt.pre["R"]
            assign(lab,"R",load_weights("protect"))
            assign(lab,"F",load_weights("full"))
            rec=bank[lane]
            obs=reset(lab.ctx,rec)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                for role in "FR":
                    for q in "AB":
                        a=lab.native(role,INSTR[q],obs,rec["seed"],0)
                        b=mix(lab,obs,q,rec["seed"],0,(role,)*4)
                        assert torch.equal(a,b), (role,q,float((a-b).abs().max()))
            print("PATCH_IDENTITY_PASS",lane,flush=True)
        with (RUN/f"{tag}_{lane}.jsonl").open("x",encoding="utf-8") as h:
            for arm in (PATCHES if patch else CORE):
                if not patch:
                    assign(lab,"F",load_weights(arm))
                for rec in bank[lane::lanes]:
                    for q in ("B" if patch else "ABN" if arm in TRAIN or arm=="anchor" else "AB"):
                        if patch:
                            orig=lab.native
                            spec=PATCHES[arm]
                            start=len(rec["controls"])

                            def patched(role,text,obs,seed,chunk):
                                if chunk<start+WINDOW:
                                    return mix(lab,obs,q,seed,chunk,spec)
                                return orig(spec[0],text,obs,seed,chunk)

                            lab.native=patched
                        try:
                            row={"arm":arm,**measure_episode(lab,rec,q)}
                        finally:
                            if patch:
                                lab.native=orig
                        h.write(json.dumps(row,allow_nan=False)+"\n")
                        h.flush()
                        print(tag,lane,arm,rec["init"],q,row["first"],flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN/f"{tag}_{lane}_COMPLETE.json",{"lane":lane,"lanes":lanes})


def targeted_native(lane,lanes,kind):
    bank=read(K19/"holdout.json")["selected"]
    full=load_weights("full")
    if kind=="crossnative":
        restored=load_weights("remove_cross")
        arms=("repair_d","reverse_d")
    else:
        initial=load_weights("anchor")
        restored={n:(initial[n] if n.startswith("model.action_time_mlp_out.") else w)
                  for n,w in full.items()}
        arms=TIME_ARMS
    lab=Lab5()
    try:
        assign(lab,"R",restored)
        assign(lab,"F",full)
        with (RUN/f"{kind}_{lane}.jsonl").open("x",encoding="utf-8") as h:
            for arm in arms:
                for rec in bank[lane::lanes]:
                    base="F" if arm.startswith("repair") else "R"
                    start=len(rec["controls"])
                    orig=lab.native

                    def patched(role,text,obs,seed,chunk):
                        if arm!="remove_time_out" and chunk<start+WINDOW:
                            return mix(lab,obs,"B",seed,chunk,PATCHES[arm])
                        return orig(base,text,obs,seed,chunk)

                    lab.native=patched
                    try:
                        row={"arm":arm,**measure_episode(lab,rec,"B")}
                    finally:
                        lab.native=orig
                    h.write(json.dumps(row,allow_nan=False)+"\n")
                    h.flush()
                    print(kind,lane,arm,rec["init"],row["first"],flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN/f"{kind}_{lane}_COMPLETE.json",{"lane":lane,"lanes":lanes})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","probe","native","patch","inputnative","crossnative"))
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=4)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.mode=="prepare":
        prepare()
    elif a.mode=="probe":
        probes()
    elif a.mode in ("inputnative","crossnative"):
        targeted_native(a.lane,a.lanes,a.mode)
    else:
        native(a.lane,a.lanes,a.mode=="patch")
