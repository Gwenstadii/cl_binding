"""Sample and lock new physical resets before examining any K18-D model."""
import argparse
import hashlib
from pathlib import Path
import sys

from train_k19 import RUN, read, write
K18=RUN.parent/"k18_training_cause_20261111_01"

sys.path.insert(0,str(K18))
from measurement import Lab5, INSTR
from k14 import km
from stage1_smoke import sources

import h5py
import numpy as np
import torch

RESET_BASE=990411
NOISE_BASE=10920031
N_CANDIDATES=48
NEEDED=20


def reset(ctx,rec):
    ctx.evaluation.reset_common(ctx.env,np.asarray(rec["state"],dtype=np.float64),rec["reset_seed"],
                                ctx.root["evaluation"]["settle_steps"],ctx.xml)
    return ctx.fresh(ctx.env)


def generate():
    torch.set_num_threads(2)
    ctx=km.Ctx("scene2")
    refs=[np.asarray(x,dtype=np.float64)[1:] for group in ctx.states.values() for x in group]
    for file in sources(("left","right","front")).values():
        with h5py.File(file,"r") as handle:
            refs.extend(np.asarray(d["states"][0],dtype=np.float64)[1:] for d in handle["data"].values())
    refs.extend(np.asarray(x["state"],dtype=np.float64)[1:] for x in read(RUN.parent/"k18_training_controls_20261111_01/candidates.json")["records"])
    refs=np.stack(refs)
    rows=[]
    try:
        for i in range(N_CANDIDATES):
            seed=RESET_BASE+i
            np.random.seed(seed)
            ctx.env.seed(seed)
            ctx.env.reset()
            state=np.asarray(ctx.env.get_sim_state(),dtype=np.float64).copy()
            assert state[1:].shape==refs[0].shape
            min_ref=float(np.linalg.norm(refs-state[1:],axis=1).min())
            if min_ref <= 1e-6:
                print("CANDIDATE_DUPLICATE",i,min_ref,flush=True)
                continue
            rec={"init":i+200,"state":state.tolist(),"reset_seed":seed,
                 "seed":NOISE_BASE+107*i,"min_reset_distance_to_known":min_ref}
            obs=reset(ctx,rec)
            geom=ctx.geo.build("scene2",ctx.env,np)
            ctx.geo.check_against(geom,ctx.geometry)
            assert not any(ctx.flags().values())
            rec.update(signature=ctx.signature(obs),z0=float(obs[ctx.obj][2]),
                       book_start=np.asarray(obs[ctx.obj],dtype=np.float64).tolist())
            assert rec["signature"]["sim"] not in {r["signature"]["sim"] for r in rows}
            rows.append(rec)
            print("CANDIDATE",rec["init"],rec["book_start"],min_ref,flush=True)
    finally:
        ctx.env.close()
    assert len(rows)>=NEEDED and len({tuple(r["book_start"]) for r in rows})==len(rows)
    write(RUN/"candidates.json",{"known_start_vectors":len(refs),"records":rows,"sampling":"fresh LIBERO task reset with fixed new seeds, then canonical-scene settle"})


def screen(lane,lanes):
    rows=read(RUN/"candidates.json")["records"]
    lab=Lab5()
    qualified=[]
    try:
        for rec in rows[lane::lanes]:
            obs=reset(lab.ctx,rec)
            assert lab.ctx.signature(obs)==rec["signature"]
            controls,trace=[],[]
            result={**rec,"boundary":None}
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                for r in range(15):
                    chunk=lab.native("F",INSTR["N"],obs,rec["seed"],r)
                    physical=lab.rt.physical(chunk)
                    controls.append(physical.tolist())
                    obs,part=lab.execute(obs,physical,r*10,rec["z0"])
                    trace.extend(part)
                    b=len(trace)
                    if b>=30 and lab.criterion(trace,rec["z0"],b)["ok"]:
                        result.update(boundary=b,controls=controls,handoff_signature=lab.ctx.signature(obs))
                        break
            qualified.append(result)
            print("SCREEN",rec["init"],result["boundary"],flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN/f"screen_{lane}.json",qualified)


def lock(lanes):
    rows=sorted([r for i in range(lanes) for r in read(RUN/f"screen_{i}.json")],key=lambda r:r["init"])
    all_candidates=read(RUN/"candidates.json")["records"]
    assert [r["init"] for r in rows]==[r["init"] for r in all_candidates]
    eligible=[r for r in rows if r["boundary"]]
    assert len(eligible)>=NEEDED,len(eligible)
    selected=eligible[:NEEDED]
    write(RUN/"holdout.json",{"selected":selected,"eligible":len(eligible),"candidates":len(rows),
          "rule":"first 20 in predetermined seed order qualifying on fixed Front12000 physical handoff",
          "candidate_sha256":hashlib.sha256((RUN/"candidates.json").read_bytes()).hexdigest()})
    print("HOLDOUT_LOCKED",len(eligible),[r["init"] for r in selected],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("generate","screen","lock"))
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=4)
    a=p.parse_args()
    if a.mode=="generate":
        generate()
    elif a.mode=="screen":
        screen(a.lane,a.lanes)
    else:
        lock(a.lanes)

