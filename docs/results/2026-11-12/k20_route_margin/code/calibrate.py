"""Calibrate a differentiable route score against paired native LIBERO rollouts."""
import argparse
import json
from pathlib import Path
import sys

from train_weak import RUN, read, write
import numpy as np
import torch

K19=RUN.parent/"k19_retention_20261111_01"
sys.path.insert(0,str(K19))
from eval_k19 import Lab5, weights, INSTR
from holdout import reset

TARGETS=("left","right","front","back")


def gather(lane,lanes):
    seen={r["init"] for r in read(K19/"holdout.json")["selected"]}
    pool=sorted([r for i in (0,1) for r in read(K19/f"screen_{i}.json")
                 if r["boundary"] and r["init"] not in seen],key=lambda r:r["init"])
    assert len(pool)==28
    lab=Lab5()
    rows,bank=[],[]
    try:
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            probe=torch.zeros(1,50,7,device="cuda")
            origin=lab.rt.post["F"](probe)[0,0].float()
            basis=lab.rt.post["F"](probe+1)[0,0].float()-origin
            write(RUN/f"action_affine_{lane}.json",{"offset":origin.tolist(),"scale":basis.tolist()})
        for rec in pool[lane::lanes]:
            for arm in ("parent","full"):
                weights(lab,20261021,arm,1000)
                for q,goal in (("A","left"),("B","right")):
                    obs=reset(lab.ctx,rec)
                    assert lab.ctx.signature(obs)==rec["signature"]
                    prefix=[]
                    for k,u in enumerate(rec["controls"]):
                        obs,part=lab.execute(obs,np.asarray(u,dtype=np.float32),10*k,rec["z0"])
                        prefix.extend(part)
                    assert lab.ctx.signature(obs)==rec["handoff_signature"]
                    assert not any(r["hits"] or any(r["pred"].values()) for r in prefix)
                    start=np.array(obs[lab.ctx.obj],dtype=np.float32)
                    centers=np.array([lab.ctx.geometry[k]["centre"][:2] for k in TARGETS],dtype=np.float32)
                    noise=lab.rt.noise(rec["seed"],len(prefix)//10)
                    allrows=[]
                    first=None
                    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                        for r in range(len(prefix)//10,40):
                            u=lab.native("F",INSTR[q],obs,rec["seed"],r)
                            phys=lab.rt.physical(u)
                            if not allrows:
                                first=np.clip(phys[:10,:2],-1,1).sum(0)
                                if arm=="parent":
                                    b=lab.rt.batch("F",obs,INSTR[q])
                                    bank.append({"init":rec["init"],"q":q,"pos":start[:2],"centers":centers,
                                      "batch":{k:v.detach().cpu() for k,v in b.items() if isinstance(v,torch.Tensor)},
                                      "noise":noise.cpu()})
                            obs,part=lab.execute(obs,phys,r*10,rec["z0"])
                            allrows.extend(part)
                    outcome=lab.ctx.geo.movement("scene2",lab.ctx.geometry,allrows,rec["z0"],None)
                    actual=np.asarray(allrows[9]["obj"])[:2]-start[:2]
                    rows.append({"init":rec["init"],"arm":arm,"q":q,"goal":goal,"pos":start[:2].tolist(),
                                 "centers":centers.tolist(),"commands":first.tolist(),"actual":actual.tolist(),
                                 "first":outcome["first_targets"]})
                    print("CALIB",rec["init"],arm,q,outcome["first_targets"],flush=True)
    finally:
        lab.ctx.env.close()
    # Use only plain tensors/lists in the bank so the loader can restrict deserialization.
    for b in bank:
        b["pos"]=b["pos"].tolist()
        b["centers"]=b["centers"].tolist()
    torch.save(bank,RUN/f"bank_{lane}.pt")
    write(RUN/f"calibration_{lane}.json",rows)


def scores(row,displacement):
    p=np.asarray(row["pos"])
    c=np.asarray(row["centers"])
    progress=np.linalg.norm(c-p,axis=-1)-np.linalg.norm(c-(p+displacement),axis=-1)
    j=TARGETS.index(row["goal"])
    return float(progress[j]),float(progress[j]-max(np.delete(progress,j)))


def auc(pos,neg):
    if not pos or not neg:
        return None
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))


def lock(lanes):
    rows=[r for i in range(lanes) for r in read(RUN/f"calibration_{i}.json")]
    assert len(rows)==112 and len({(r["init"],r["arm"],r["q"]) for r in rows})==112
    a=[r for r in rows if r["init"]<240]
    b=[r for r in rows if r["init"]>=240]
    x=np.asarray([r["commands"] for r in a])
    y=np.asarray([r["actual"] for r in a])
    coef=np.linalg.lstsq(x,y,rcond=None)[0]
    residual=[]
    for r in rows:
        d=np.asarray(r["commands"])@coef
        r["predicted"]=d.tolist()
        r["proxy_progress"],r["proxy_margin"]=scores(r,d)
        r["actual_progress"],r["actual_margin"]=scores(r,np.asarray(r["actual"]))
        if r["init"]>=240:
            residual.append(d-np.asarray(r["actual"]))
    report={"coef":coef.tolist(),"fit_rows":len(a),"heldout_rows":len(b),
            "heldout_xy_rmse_m":float(np.sqrt(np.square(residual).mean())),"scores":{}}
    for k in ("proxy_progress","proxy_margin","actual_progress","actual_margin"):
        yes=[r[k] for r in b if r["first"]==[r["goal"]]]
        front=[r[k] for r in b if r["first"]==["front"]]
        none=[r[k] for r in b if not r["first"]]
        report["scores"][k]={"correct":len(yes),"front":len(front),"none":len(none),
            "auc_correct_vs_front":auc(yes,front),"auc_correct_vs_none":auc(yes,none)}
    pairs=[r for i in range(lanes) for r in torch.load(RUN/f"bank_{i}.pt",weights_only=True)]
    ref={(r["init"],r["q"]):r for r in a if r["arm"]=="parent"}
    # Relative floors are set per valid reference query. Margin slack is measured in meters.
    kept=[]
    for p in pairs:
        r=ref.get((p["init"],p["q"]))
        if r is None or r["first"]!=[r["goal"]] or r["proxy_progress"]<=0 or r["proxy_margin"]<=0:
            continue
        p["floor"]=[.55*r["proxy_progress"],.55*r["proxy_margin"]]
        p["target"]=TARGETS.index(r["goal"])
        kept.append(p)
    assert {p["q"] for p in kept}=={"A","B"}
    report["eligible_by_prompt"]={q:sum(p["q"]==q for p in kept) for q in ("A","B")}
    torch.save(kept,RUN/"route_bank.pt")
    write(RUN/"calibration.json",{"report":report,"rows":rows})
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("gather","lock"))
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=4)
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.mode=="gather":
        gather(a.lane,a.lanes)
    else:
        lock(a.lanes)
