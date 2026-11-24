"""Evaluate fixed input fields, held-out demos, and locked new resets for K18-D."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys
import time

from train_controls import RUN, K18, SEEDS, ARMS, SAVE, PARENT, read, write, digest
from holdout import reset

sys.path.insert(0,str(K18))
from measurement import Lab5, INSTR, load, make_bank, responses
from demo_probe import make_prefix
from stage1_smoke import ROOT

import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from safetensors import safe_open
from safetensors.torch import load_file

ORDER=((SEEDS[0],"full"),(SEEDS[0],"no_rot"),(SEEDS[1],"full"),(SEEDS[1],"no_rot"),
       (SEEDS[2],"no_rot"),(SEEDS[1],"random"),(SEEDS[0],"random"),(SEEDS[0],"norm"),
       (SEEDS[2],"full"),(SEEDS[1],"norm"),(SEEDS[2],"random"),(SEEDS[2],"norm"))
CHANNELS={"H":slice(0,2),"Z":slice(2,3),"R":slice(3,6),"G":slice(6,7)}


def await_file(path):
    while not path.exists():
        time.sleep(5)


def load_branch(lab,seed,arm,step,check=False):
    folder=RUN/"runs"/f"{seed}_{arm}"/f"{step:06d}"
    await_file(folder/"READY.json")
    marker=read(folder/"READY.json")
    file=folder/"trainable.safetensors"
    if check:
        assert digest(file)==marker["sha256"]
    state=load_file(str(file))
    named={n:p for n,p in lab.rt.models["F"].named_parameters() if p.requires_grad}
    assert state.keys()==named.keys() and len(state)==marker["count"]==155
    with torch.no_grad():
        for n,p in named.items():
            assert p.dtype==state[n].dtype
            p.copy_(state[n])
    lab.rt.models["F"].eval()
    return state


def episode(lab,rec,goal):
    obs=reset(lab.ctx,rec)
    assert lab.ctx.signature(obs)==rec["signature"]
    rows=[]
    if goal in "AB":
        for r,c in enumerate(rec["controls"]):
            obs,events=lab.execute(obs,lab.np.asarray(c,dtype=lab.np.float32),r*10,rec["z0"])
            rows.extend(events)
        assert len(rows)==rec["boundary"] and lab.ctx.signature(obs)==rec["handoff_signature"]
        assert not any(row["hits"] or any(row["pred"].values()) for row in rows)
    b=len(rows)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for r in range(b//10,40):
            chunk=lab.native("F",INSTR[goal],obs,rec["seed"],r)
            obs,events=lab.execute(obs,lab.rt.physical(chunk),r*10,rec["z0"])
            rows.extend(events)
    assert len(rows)==400
    move=lab.ctx.geo.movement("scene2",lab.ctx.geometry,rows,rec["z0"],None)
    return {"init":rec["init"],"goal":goal,"boundary":b,"first":move["first_targets"],
            "first_step":move["first_step"],
            "ever":{g:any(x["pred"][g] for x in rows) for g in lab.ctx.geo.GOALS},
            "terminal":rows[-1]["pred"]}


def native(lab,lane,lanes):
    await_file(RUN/"holdout.json")
    records=read(RUN/"holdout.json")["selected"][lane::lanes]
    output=RUN/f"native_w{lane}.jsonl"
    seen={(r["seed"],r["arm"],r["init"],r["goal"]) for r in map(json.loads,output.read_text().splitlines())} if output.exists() else set()
    for seed,arm in ((0,"parent"),*ORDER):
        expected={(seed,arm,r["init"],g) for r in records for g in "ABN"}
        if expected<=seen:
            continue
        if arm=="parent":
            load(lab,PARENT)
        else:
            load_branch(lab,seed,arm,1000)
        with output.open("a") as handle:
            for rec in records:
                for goal in "ABN":
                    key=(seed,arm,rec["init"],goal)
                    if key in seen:
                        continue
                    row={"seed":seed,"arm":arm,**episode(lab,rec,goal)}
                    handle.write(json.dumps(row,allow_nan=False)+"\n")
                    handle.flush()
                    print("NATIVE",lane,seed,arm,rec["init"],goal,row["first"],flush=True)
    write(RUN/f"native_w{lane}_COMPLETE.json",{"lane":lane,"lanes":lanes})


def validation_bank(lab):
    path=ROOT/read(RUN/"config.json")["data"]["validation_output"]
    conversion=read(path/"r0_conversion.json")
    await_file(RUN/"front_validation_phases.json")
    ph=read(RUN/"front_validation_phases.json")
    assert ph["split_sha256"]==conversion["spec"]["split_sha256"]
    train_path=ROOT/read(RUN/"config.json")["data"]["train_output"]
    train=read(train_path/"r0_conversion.json")
    assert not {x["demo"] for x in conversion["episodes"]} & {x["demo"] for x in train["episodes"]}
    ds=LeRobotDataset("local/k18_front_validation",root=path,video_backend="pyav",
                     delta_timestamps={"action":[i/20 for i in range(50)]})
    labels=ph["labels"]
    assert len(ds)==len(labels)
    bank=[]
    start=0
    with torch.no_grad():
        for rec in conversion["episodes"]:
            end=start+rec["frames"]
            for portion in (.20,.50,.80):
                index=start+min(rec["frames"]-1,int(rec["frames"]*portion))
                raw=torch.utils.data.default_collate(ds.__getitems__([index]))
                batch=lab.rt.pre["R"](copy.deepcopy(raw))
                actions=lab.rt.models["R"].prepare_action(batch)
                valid=~batch["action_is_pad"][0]
                offsets=torch.arange(50,device="cuda")+index
                assert not bool((valid & (offsets>=end)).any())
                phase=torch.tensor(labels[index:min(index+50,end)]+[3]*max(0,index+50-end),device="cuda")
                assert phase.shape==valid.shape
                rng=torch.Generator().manual_seed(202612100+index)
                noise=torch.randn(actions.shape,generator=rng).cuda()
                drafts=[]
                for t in (.9,.5,.1):
                    time_tensor=torch.tensor([t],device="cuda")
                    x,target,_=make_flow_matching_inputs(actions,noise,time_tensor,convention=FlowConvention.NOISE_AT_ONE)
                    drafts.append((x,target,time_tensor,t))
                bank.append((rec["demo"],index,raw,valid,phase,drafts))
            start=end
    assert start==len(ds) and len(bank)==30
    return bank,ph["names"]


def validation_fields(lab,bank,phases):
    rows=[]
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for demo,index,raw,valid,phase,drafts in bank:
            prefix=make_prefix(lab,"F",raw,INSTR["N"])
            for x,target,t,time_value in drafts:
                err=(lab.rt.query("F",prefix,x,t).float()-target)[0,:,:7].square()
                row={"demo":demo,"index":index,"time":time_value,"sq_error":{}}
                for j,name in enumerate(phases):
                    selected=valid & (phase==j)
                    count=int(selected.sum())
                    row["sq_error"][name]={k:{"sum":float(err[selected,v].sum()),
                                           "count":count*(v.stop-v.start)} for k,v in CHANNELS.items()}
                rows.append(row)
    return rows


def key_value_reference():
    a=K18/"train/checkpoints/000030/model.safetensors"
    b=K18/"train/checkpoints/000100/model.safetensors"
    with safe_open(a,framework="pt") as sa,safe_open(b,framework="pt") as sb:
        names=[]
        for n in sa.keys():
            m=re.fullmatch(r"model\.vlm_with_expert\.lm_expert\.layers\.(\d+)\.self_attn\.[kv]_proj\.weight",n)
            if m and int(m[1])%2==1:
                names.append(n)
        assert len(names)==16
        return {n:sb.get_tensor(n).float()-sa.get_tensor(n).float() for n in names}


def parameter_projection(ref,state,initial):
    inner,delta2,ref2=0.,0.,0.
    for n,d in ref.items():
        delta=state[n].float()-initial[n].float()
        inner+=float((d*delta).sum())
        delta2+=float(delta.square().sum())
        ref2+=float(d.square().sum())
    return {"gain":inner/ref2,"cosine":inner/max((delta2*ref2)**.5,1e-20),"l2":delta2**.5}


def probe(lab):
    fixed=make_bank(lab,range(20,40))
    demos,phases=validation_bank(lab)
    ref=key_value_reference()
    load(lab,PARENT)
    write(RUN/"probe_parent.json",{"fixed":responses(lab,fixed,True),
          "front_demo":validation_fields(lab,demos,phases)})
    for seed,arm in ORDER:
        base=load_branch(lab,seed,arm,30,check=True)
        for step in SAVE:
            output=RUN/f"probe_{seed}_{arm}_{step:06d}.json"
            if output.exists():
                continue
            state=load_branch(lab,seed,arm,step,check=True)
            result={"seed":seed,"arm":arm,"step":step,"fixed":responses(lab,fixed,True),
                    "front_demo":validation_fields(lab,demos,phases),"kv_update_from_30":parameter_projection(ref,state,base)}
            write(output,result)
            print("PROBE",seed,arm,step,flush=True)
            del state
        del base
    write(RUN/"probe_COMPLETE.json",{"status":"pass"})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("native","probe"))
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=6)
    a=p.parse_args()
    torch.set_num_threads(2)
    lab=Lab5()
    try:
        if a.mode=="native":
            native(lab,a.lane,a.lanes)
        else:
            probe(lab)
    finally:
        lab.ctx.env.close()
