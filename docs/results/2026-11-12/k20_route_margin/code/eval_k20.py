"""K20 native development/confirmation evaluations and fixed-input probes."""
import argparse
import json
import sys

from train_weak import RUN, read, write, digest
import numpy as np
import torch
from safetensors.torch import load_file

K19=RUN.parent/"k19_retention_20261111_01"
sys.path.insert(0,str(K19))
from eval_k19 import Lab5, INSTR, wait, weights as ref_weights, validation_bank, validation_fields, make_bank, responses, snapshots, rebuild
from holdout import reset


def load_arm(lab,seed,arm,step):
    if arm in ("full","replay_full"):
        ref_weights(lab,seed,arm,step,True)
        return
    name="both" if arm=="weak" else arm
    path=RUN/"runs"/f"{seed}_{name}"/f"{step:06d}"
    wait(path/"READY.json")
    file=path/"trainable.safetensors"
    assert digest(file)==read(path/"READY.json")["sha256"]
    states=load_file(str(file))
    params={n:p for n,p in lab.rt.models["F"].named_parameters() if p.requires_grad}
    assert params.keys()==states.keys() and len(states)==155
    with torch.no_grad():
        for n,p in params.items():
            assert states[n].dtype==p.dtype
            p.copy_(states[n])
    lab.rt.models["F"].eval()


def episode(lab,rec,goal,suite):
    if suite=="dev" and goal!="N":
        obs,rows=rebuild(lab,rec)
        z0=rec["z0"]
    else:
        obs=lab.ctx.reset(rec["init"])[0] if suite=="dev" else reset(lab.ctx,rec)
        if suite=="confirm":
            assert lab.ctx.signature(obs)==rec["signature"]
        z0,rows=float(obs[lab.ctx.obj][2]),[]
        if goal!="N":
            for k,u in enumerate(rec["controls"]):
                obs,part=lab.execute(obs,np.asarray(u,dtype=np.float32),k*10,z0)
                rows.extend(part)
            assert lab.ctx.signature(obs)==rec["handoff_signature"]
    b=len(rows)
    assert not any(r["hits"] or any(r["pred"].values()) for r in rows)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for k in range(b//10,40):
            u=lab.native("F",INSTR[goal],obs,rec["seed"],k)
            obs,part=lab.execute(obs,lab.rt.physical(u),k*10,z0)
            rows.extend(part)
    assert len(rows)==400
    hit=lab.ctx.geo.movement("scene2",lab.ctx.geometry,rows,z0,None)
    stable=lab.ctx.geo.first_run([r["step"] for r in rows if r["obj"][2]-z0>.03 and r["c1"] and r["c2"]])
    return {"init":rec["init"],"goal":goal,"boundary":b,"first":hit["first_targets"],"first_step":hit["first_step"],
            "first_stable_grasp":stable,"reached":hit["times"],
            "ever":{k:any(r["pred"][k] for r in rows) for k in lab.ctx.geo.GOALS},
            "terminal":rows[-1]["pred"]}


def native(lab,suite,seed,lane,lanes,arms):
    if suite=="dev":
        bank=[snapshots()[i] for i in range(20,40)]
    else:
        bank=read(K19/"holdout.json")["selected"]
    with (RUN/f"{suite}_{seed}_w{lane}.jsonl").open("x") as h:
        for arm in arms:
            load_arm(lab,seed,arm,1000)
            goals="N" if arm in ("full","replay_full") else "ABN"
            for rec in bank[lane::lanes]:
                for q in goals:
                    row={"seed":seed,"arm":arm,**episode(lab,rec,q,suite)}
                    h.write(json.dumps(row,allow_nan=False)+"\n")
                    h.flush()
                    print("NATIVE",suite,seed,arm,rec["init"],q,row["first"],flush=True)
    write(RUN/f"{suite}_{seed}_w{lane}_COMPLETE.json",{"arms":arms,"lane":lane})


def probe(lab,seed,arms):
    bank=make_bank(lab,range(20,40))
    val,phases=validation_bank(lab)
    for arm in arms:
        for step in (30,100,300,1000):
            load_arm(lab,seed,arm,step)
            fields=responses(lab,bank,True)
            demo=validation_fields(lab,val,phases)
            write(RUN/f"probe_{seed}_{arm}_{step:06d}.json",{"fixed":fields,"front_demo":demo})
            print("PROBE",seed,arm,step,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("dev","confirm","probe"))
    p.add_argument("--seed",type=int,default=20261021)
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=4)
    p.add_argument("--arms",nargs="+",default=("weak","full","replay_full","route"))
    a=p.parse_args()
    torch.set_num_threads(2)
    lab=Lab5()
    try:
        if a.mode=="probe":
            probe(lab,a.seed,a.arms)
        else:
            native(lab,a.mode,a.seed,a.lane,a.lanes,a.arms)
    finally:
        lab.ctx.env.close()
