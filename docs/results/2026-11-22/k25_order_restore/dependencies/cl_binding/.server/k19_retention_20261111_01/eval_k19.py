"""Native K19 development and independent confirmation, plus fixed-input probes."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file
from train_k19 import RUN, D, ARMS, PARENT, read, write, digest

sys.path.insert(0,str(D))
from evaluate import episode, validation_bank, validation_fields
from measurement import Lab5, INSTR, load, make_bank, responses, rebuild, snapshots

STEPS=(30,100,300,1000)


def wait(path):
    while not path.exists():
        time.sleep(5)


def weights(lab,seed,arm,step,check=False):
    if arm=="parent":
        load(lab,PARENT)
        return
    root=D if arm=="full" else RUN
    directory=root/"runs"/f"{seed}_{arm}"/f"{step:06d}"
    wait(directory/"READY.json")
    file=directory/"trainable.safetensors"
    if check:
        assert digest(file)==read(directory/"READY.json")["sha256"]
    state=load_file(str(file))
    named={n:p for n,p in lab.rt.models["F"].named_parameters() if p.requires_grad}
    assert state.keys()==named.keys() and len(state)==155
    with torch.no_grad():
        for n,p in named.items():
            assert p.dtype==state[n].dtype
            p.copy_(state[n])
    lab.rt.models["F"].eval()


def dev_episode(lab,snap,goal):
    if goal=="N":
        obs,_=lab.ctx.reset(snap["init"])
        z0,rows=float(obs[lab.ctx.obj][2]),[]
    else:
        obs,rows=rebuild(lab,snap)
        z0=snap["z0"]
        assert not any(r["hits"] or any(r["pred"].values()) for r in rows)
    boundary=len(rows)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for k in range(boundary//10,40):
            u=lab.native("F",INSTR[goal],obs,snap["seed"],k)
            obs,part=lab.execute(obs,lab.rt.physical(u),k*10,z0)
            rows.extend(part)
    assert len(rows)==400
    m=lab.ctx.geo.movement("scene2",lab.ctx.geometry,rows,z0,None)
    return {"init":snap["init"],"goal":goal,"boundary":boundary,"first":m["first_targets"],
            "first_step":m["first_step"],"ever":{g:any(x["pred"][g] for x in rows) for g in lab.ctx.geo.GOALS},
            "terminal":rows[-1]["pred"]}


def native(lab,mode,lane,lanes,arms):
    if mode=="dev":
        bank=[snapshots()[i] for i in range(20,40)][lane::lanes]
        conditions=[(20261021,a) for a in (arms or ARMS)]
        step_fn=dev_episode
    else:
        wait(RUN/"holdout.json")
        bank=read(RUN/"holdout.json")["selected"][lane::lanes]
        assert arms and len(arms)==1
        candidate=arms[0]
        conditions=[(0,"parent"),*((s,a) for s in (20261121,20261122) for a in ("full",candidate))]
        step_fn=episode
    output=RUN/f"{mode}_w{lane}.jsonl"
    done={(r["seed"],r["arm"],r["init"],r["goal"]) for r in map(json.loads,output.read_text().splitlines())} if output.exists() else set()
    with output.open("a") as handle:
        for seed,arm in conditions:
            if {(seed,arm,r["init"],g) for r in bank for g in "ABN"}<=done:
                continue
            weights(lab,seed,arm,1000)
            for rec in bank:
                for goal in "ABN":
                    if (seed,arm,rec["init"],goal) in done:
                        continue
                    row={"seed":seed,"arm":arm,**step_fn(lab,rec,goal)}
                    handle.write(json.dumps(row,allow_nan=False)+"\n")
                    handle.flush()
                    print("NATIVE",mode,lane,seed,arm,rec["init"],goal,row["first"],flush=True)
    write(RUN/f"{mode}_w{lane}_COMPLETE.json",{"lane":lane,"conditions":conditions})


def probe(lab,arms,seed):
    bank=make_bank(lab,range(20,40))
    val,phases=validation_bank(lab)
    for a in arms:
        for step in STEPS:
            file=RUN/f"probe_{seed}_{a}_{step:06d}.json"
            if file.exists():
                continue
            weights(lab,seed,a,step,True)
            fields=responses(lab,bank,True)
            demos=validation_fields(lab,val,phases)
            write(file,{"fixed":fields,"front_demo":demos,"seed":seed,"arm":a,"step":step})
            print("PROBE",seed,a,step,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("dev","confirm","probe"))
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=6)
    p.add_argument("--arms",nargs="+")
    p.add_argument("--seed",type=int,default=20261021)
    a=p.parse_args()
    torch.set_num_threads(2)
    lab=Lab5()
    try:
        if a.mode=="probe":
            probe(lab,a.arms or ARMS,a.seed)
        else:
            native(lab,a.mode,a.lane,a.lanes,a.arms)
    finally:
        lab.ctx.env.close()
