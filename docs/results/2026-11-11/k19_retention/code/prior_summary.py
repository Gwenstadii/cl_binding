"""Aggregate matched K18-D training, probes and locked-state outcomes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median

SEEDS=(20261021,20261121,20261122)
ARMS=("full","no_rot","random","norm")
STEPS=(30,100,300,1000)
GOALS={"A":"left","B":"right","N":"front"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def counts(rows):
    first=Counter("/".join(r["first"]) or "none" for r in rows)
    return {"n":len(rows),"first":dict(first),
            "ever":{g:sum(r["ever"][g] for r in rows) for g in GOALS.values()},
            "terminal":{g:sum(r["terminal"][g] for r in rows) for g in GOALS.values()}}


def native(root, selected, partial):
    files=sorted(root.glob("native_w[0-5].jsonl"))
    rows=[r for f in files for r in jsonl(f)]
    groups=defaultdict(dict)
    for r in rows:
        key=(r["seed"],r["arm"])
        identity=(r["init"],r["goal"])
        assert identity not in groups[key]
        assert r["init"] in selected and r["goal"] in GOALS
        if r["goal"] in "AB":
            assert r["first_step"] is None or r["first_step"]>r["boundary"]
        groups[key][identity]=r
    expected={(i,g) for i in selected for g in GOALS}
    planned={(0,"parent"),*((s,a) for s in SEEDS for a in ARMS)}
    if not partial:
        assert len(files)==6 and all((root/f"native_w{i}_COMPLETE.json").exists() for i in range(6))
        assert set(groups)==planned and all(set(x)==expected for x in groups.values())
        assert len(rows)==780
    results={}
    for (s,a), group in groups.items():
        if set(group)!=expected:
            continue
        stats={goal:counts([group[i,g] for i in selected]) for g,goal in GOALS.items()}
        stats["paired_old_first_correct"]=sum(group[i,"A"]["first"]==["left"] and group[i,"B"]["first"]==["right"]
                                             for i in selected)
        results[f"{s}_{a}"]=stats
    comparisons={}
    for s in SEEDS:
        for c in ("full","random","norm"):
            treatment,control=(s,"no_rot"),(s,c)
            if f"{s}_no_rot" not in results or f"{s}_{c}" not in results:
                continue
            pair={}
            for g,goal in GOALS.items():
                a,b=groups[treatment],groups[control]
                pair[goal]={"first":sum((a[i,g]["first"]==[goal])-(b[i,g]["first"]==[goal]) for i in selected),
                            "terminal":sum(a[i,g]["terminal"][goal]-b[i,g]["terminal"][goal] for i in selected),
                            "transitions":dict(Counter(
                                f'{"/".join(b[i,g]["first"]) or "none"} -> {"/".join(a[i,g]["first"]) or "none"}'
                                for i in selected))}
            comparisons[f"{s}_no_rot_vs_{c}"]=pair
    return {"n":len(rows),"outcomes":results,"paired":comparisons}


def fixed_summary(rows):
    assert len(rows)==120
    by_scene=defaultdict(list)
    for r in rows:
        by_scene[r["init"]].append(r)
    assert set(by_scene)==set(range(20,40)) and {len(v) for v in by_scene.values()}=={6}
    result={}
    for n in ("10","50"):
        values={"gain":lambda r:r[n]["gain"],"cosine":lambda r:r[n]["cosine"],
                "gain_H":lambda r:r[n]["contrast_by_channel"]["H"]["gain"],
                "norm_ratio":lambda r:r[n]["norm_ratio"]}
        entry={k:median(median(fn(r) for r in v) for v in by_scene.values()) for k,fn in values.items()}
        entry["mu_drift"]={k:median(median(r[n]["mu_drift_rms"][k] for r in v) for v in by_scene.values())
                           for k in "HZRGP"}
        result[n]=entry
    return result


def demo_summary(rows):
    assert len(rows)==90
    demos=defaultdict(list)
    for r in rows:
        demos[r["demo"]].append(r)
    assert len(demos)==10 and {len(v) for v in demos.values()}=={9}
    result={}
    phases=tuple(rows[0]["sq_error"])
    for name in (*phases,"all"):
        result[name]={}
        for channel in ("H","Z","R","G","all"):
            values=[]
            for rs in demos.values():
                pieces=[cell for r in rs for phase,cells in r["sq_error"].items()
                        if name=="all" or phase==name for k,cell in cells.items() if channel=="all" or channel==k]
                n=sum(x["count"] for x in pieces)
                if n:
                    values.append(sum(x["sum"] for x in pieces)/n)
            result[name][channel]={"median":median(values),"n_demos":len(values)} if values else None
    return result


def probes(root,partial):
    result={}
    for name in ("parent",*(f"{s}_{a}_{k:06d}" for s in SEEDS for a in ARMS for k in STEPS)):
        file=root/f"probe_{name}.json"
        if partial and not file.exists():
            continue
        r=read(file)
        result[name]={"fixed":fixed_summary(r["fixed"]),"front_demo":demo_summary(r["front_demo"]),
                      "kv":r.get("kv_update_from_30")}
    return result


def training(root,partial):
    result={}
    all_rows={}
    for seed in SEEDS:
        for arm in ARMS:
            directory=root/"runs"/f"{seed}_{arm}"
            if partial and not (directory/"COMPLETE.json").exists():
                continue
            complete=read(directory/"COMPLETE.json")
            rows=jsonl(directory/"train_steps.jsonl")
            assert complete["frozen_unchanged"] and complete["step"]==len(rows)==1000
            assert [r["step"] for r in rows]==list(range(1,1001))
            key=f"{seed}_{arm}"
            result[key]={"seconds":complete["seconds"],"checkpoints":{k:read(directory/f"{k:06d}"/"READY.json")
                          for k in STEPS},"update_l2_sum":sum(r["update_l2"] for r in rows)}
            all_rows[(seed,arm)]=rows
            if arm=="norm":
                err=[abs(r["update_l2"]/r["target_l2"]-1) for r in rows]
                result[key]["norm_gate"]={"max":max(err),"median":median(err),
                    "within_1_5_percent":sum(e<=.015 for e in err),"median_proposal_cosine":median(r["cosine"] for r in rows),
                    "min_proposal_cosine":min(r["cosine"] for r in rows)}
                if not partial:
                    assert sum(e<=.015 for e in err)>=990
    for seed in SEEDS:
        saved=[all_rows[(seed,a)] for a in ARMS if (seed,a) in all_rows]
        if saved:
            assert all(all(tuple(r[k] for k in ("indices","valid_positions","lr"))==
                           tuple(ref[k] for k in ("indices","valid_positions","lr")) for r,ref in zip(rows,saved[0]))
                       for rows in saved)
    reference=root/"historical_1000.jsonl"
    if reference.exists() and (SEEDS[0],"full") in all_rows:
        original=jsonl(reference)
        full=all_rows[SEEDS[0],"full"]
        assert len(original)==len(full)
        result["historical_mismatches"]={k:sum(a[k]!=b[j] for a,b in zip(full,original))
            for k,j in (("step","step"),("loss","loss"),("grad_norm","gradient_norm"),("indices","batch_indices"),("lr","lr"))}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("path",type=Path)
    p.add_argument("--partial",action="store_true")
    a=p.parse_args()
    lock=read(a.path/"holdout.json")
    selected=[r["init"] for r in lock["selected"]]
    assert len(set(selected))==20 and len({r["signature"]["sim"] for r in lock["selected"]})==20
    assert min(r["min_reset_distance_to_known"] for r in lock["selected"])>1e-6
    out={"native":native(a.path,selected,a.partial),"probes":probes(a.path,a.partial),
         "training":training(a.path,a.partial),
         "holdout":{"ids":selected,"candidates":lock["candidates"],"qualified":lock["eligible"]}}
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
