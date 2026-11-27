"""Verify paired K20 episodes, checkpoint budgets, and calibration evidence."""
import argparse
from collections import Counter,defaultdict
import json
from math import dist
from pathlib import Path
from statistics import median
import sys

RUN=Path(__file__).resolve().parent
D=RUN.parent/"k18_training_controls_20261111_01"
sys.path.insert(0,str(D))
try:
    from summarize import fixed_summary,demo_summary
except ImportError:
    from prior_summary import fixed_summary,demo_summary

GOALS={"A":"left","B":"right","N":"front"}
ARMS=("full","replay_full","weak","route")
SUITES={"dev":(20261021,range(20,40)),"confirm":(20261121,range(200,220))}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def label(row):
    return "/".join(row["first"]) or "none"


def calibration(path):
    raw=read(path/"calibration.json")
    report=raw["report"]
    pairs={(r["init"],r["q"],r["arm"]):r for r in raw["rows"]}
    report["floor_diagnostic"]={}
    for q,target in (("A","left"),("B","right")):
        states=[]
        for i in range(220,240):
            old=pairs[i,q,"parent"]
            new=pairs[i,q,"full"]
            if old["first"]!=[target] or min(old["proxy_progress"],old["proxy_margin"])<=0:
                continue
            fails=any(new[k]<.55*old[k] for k in ("proxy_progress","proxy_margin"))
            states.append((fails,new["first"]!=[target]))
        report["floor_diagnostic"][target]={"fit_pairs":len(states),
                                             "full_below_floor":sum(f for f,_ in states),
                                             "full_passes_floor_but_misses":sum(not f and m for f,m in states)}
    return report


def transport(path,suite,ids,endpoints,partial):
    files=sorted(path.glob(f"transport_{suite}_[0-1].jsonl"))
    if partial and not files:
        return {}
    rows=[r for f in files for r in lines(f)]
    lookup={(r["arm"],r["init"],r["q"]):r for r in rows}
    assert len(lookup)==len(rows)
    wanted={(a,i,q) for a in ARMS for i in ids for q in "AB"}
    assert set(lookup)<=wanted
    if not partial:
        assert set(lookup)==wanted
    result={}
    for arm in ARMS:
        if not all((arm,i,q) in lookup for i in ids for q in "AB"):
            continue
        by_goal={}
        for q,goal in (("A","left"),("B","right")):
            subset=[lookup[arm,i,q] for i in ids]
            by_goal[goal]={k:median(r[k] for r in subset)*1000
                           for k in ("proxy_progress","proxy_margin","actual_progress","actual_margin")}
            by_goal[goal]["actual_positive_margin"]=sum(r["actual_margin"]>0 for r in subset)
        separation,common=[],[]
        for i in ids:
            a,b=(lookup[arm,i,q] for q in "AB")
            left,right,front,_=a["centers"]
            width=dist(left,right)
            front_dist=dist(front,a["pos"])
            separation.append(sum((x-y)*(l-r)/width for x,y,l,r in zip(
                a["observed"],b["observed"],left,right,strict=True)))
            common.append(sum((x+y)*.5*(f-p)/front_dist for x,y,f,p in zip(
                a["observed"],b["observed"],front,a["pos"],strict=True)))
        timepoints=(10,20,30,40)
        course={}
        for q,goal in (("A","left"),("B","right")):
            by_end=defaultdict(list)
            for i in ids:
                if (arm,i,q) in endpoints:
                    by_end[label(endpoints[arm,i,q])].append(lookup[arm,i,q]["trajectory"])
            course[goal]={end:{"n":len(trajectories),"steps":timepoints,
                               "margin_mm":[1000*median(t[j]["actual_margin"] for t in trajectories)
                                            for j in range(len(timepoints))]}
                          for end,trajectories in by_end.items()}
        result[arm]={"per_goal":by_goal,"actual_left_right_separation_mm":1000*median(separation),
                     "actual_common_toward_front_mm":1000*median(common),"course_by_first":course}
    return result


def aggregate(path,partial=False):
    refs=lines(path/"baseline.jsonl")
    result={}
    for suite,(seed,ids) in SUITES.items():
        base={(r["arm"],r["init"],r["goal"]):r for r in refs if r["suite"]==suite}
        assert len(base)==120
        files=sorted(path.glob(f"{suite}_{seed}_w[0-3].jsonl"))
        rows=[r for f in files for r in lines(f)]
        keys=[(r["arm"],r["init"],r["goal"]) for r in rows]
        assert len(keys)==len(set(keys))
        expected={(a,i,g) for a in ARMS for i in ids for g in ("N" if a in ("full","replay_full") else "ABN")}
        assert set(keys)<=expected
        if not partial:
            assert set(keys)==expected and len(files)==4
        for r,k in zip(rows,keys,strict=True):
            if k in base:
                assert all(base[k][f]==r[f] for f in ("first","first_step","ever","terminal")),k
            base[k]=r
        out={}
        for arm in ARMS:
            if not all((arm,i,g) in base for i in ids for g in GOALS):
                continue
            armdata={}
            for g,t in GOALS.items():
                sub=[base[arm,i,g] for i in ids]
                armdata[t]={"first":dict(Counter(map(label,sub))),
                            "ever":sum(r["ever"][t] for r in sub),"terminal":sum(r["terminal"][t] for r in sub)}
                if g=="N" and all("reached" in r for r in sub):
                    armdata[t]["stable_grasp"]=sum(r["first_stable_grasp"] is not None for r in sub)
                    armdata[t]["ever_reached"]=sum(r["reached"][t] is not None for r in sub)
            armdata["both_first"]=sum(base[arm,i,"A"]["first"]==["left"] and
                                      base[arm,i,"B"]["first"]==["right"] for i in ids)
            out[arm]=armdata
        paired={}
        if "route" in out:
            for arm in (x for x in out if x!="route"):
                paired[arm]={t:{"first":dict(Counter(f'{label(base[arm,i,g])} -> {label(base["route",i,g])}' for i in ids)),
                               "terminal_gained":sum(base["route",i,g]["terminal"][t] and not base[arm,i,g]["terminal"][t] for i in ids),
                               "terminal_lost":sum(base[arm,i,g]["terminal"][t] and not base["route",i,g]["terminal"][t] for i in ids)}
                             for g,t in GOALS.items()}
        result[suite]={"seed":seed,"new_episodes":len(rows),"outcomes":out,"paired_route_vs":paired,
                       "first_block":transport(path,suite,ids,base,partial)}
    probe={}
    for file in sorted(path.glob("probe_*.json")):
        r=read(file)
        probe[file.stem.removeprefix("probe_")]={"fixed":fixed_summary(r["fixed"]),"front_demo":demo_summary(r["front_demo"])}
    if not partial:
        assert len(probe)==16
    train={}
    baseline={int(f.stem.split("_")[-1]):lines(f) for f in path.glob("train_ref_*.jsonl")}
    for seed,_ in SUITES.values():
        for a in ("both","route"):
            folder=path/"runs"/f"{seed}_{a}"
            if partial and not (folder/"COMPLETE.json").exists():
                continue
            done=read(folder/"COMPLETE.json")
            raw=lines(folder/"train_steps.jsonl")
            assert len(raw)==1000 and done["frozen_unchanged"]
            assert all((r["step"],r["indices"],r["lr"])==(b["step"],b["indices"],b["lr"])
                       for r,b in zip(raw,baseline[seed],strict=True))
            assert abs(raw[0]["front_loss"]-baseline[seed][0]["loss"])<1e-6
            meta={"arm":"weak" if a=="both" else a,"seed":seed,"steps":1000,
                  "all_front_batches_and_lr_match":True,"frozen_unchanged":True,"first_front_loss_matches":True}
            if a=="route":
                checks=[r for r in raw if r["route_loss"] is not None]
                meta.update(route_checks=len(checks),route_active_steps=sum(r["active_terms"]>0 for r in checks),
                            route_active_by_prompt={
                                g:sum(r["active_terms"]>0 for r in checks if (r["step"]-1)//3%2==j)
                                for j,g in enumerate(("left","right"))},
                            route_final_250_active=sum(r["active_terms"]>0 for r in checks if r["step"]>=751),
                            route_loss_quarter_medians=[median(r["route_loss"] for r in checks if k<=r["step"]<k+250)
                                                        for k in (1,251,501,751)])
            train[folder.name]=meta
    return {"calibration":calibration(path),"native":result,"probes":probe,"training":train}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("path",type=Path,nargs="?",default=RUN)
    p.add_argument("--partial",action="store_true")
    a=p.parse_args()
    print(json.dumps(aggregate(a.path,a.partial),indent=2,allow_nan=False))

