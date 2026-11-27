"""Audit K19 comparisons, rank the development screen, and summarize confirmation."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from statistics import mean,median
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"k18_training_controls_20261111_01"))
try:
    from summarize import fixed_summary,demo_summary
except ImportError:
    # Local archive stores the two verified K18-D reducers under this name.
    from prior_summary import fixed_summary,demo_summary

ARMS=("freeze_kv","mu","d","both","replay","both_replay","replay_full","replay_soft")
GOALS={"A":"left","B":"right","N":"front"}


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def native(rows,expected):
    group=defaultdict(dict)
    for r in rows:
        key=(r["seed"],r["arm"])
        ident=(r["init"],r["goal"])
        assert ident not in group[key]
        if r.get("boundary") is not None and r["goal"] in "AB":
            assert r["first_step"] is None or r["first_step"]>r["boundary"]
        group[key][ident]=r
    outcomes={}
    expected={(i,g) for i in expected for g in GOALS}
    for key,items in sorted(group.items()):
        if set(items)!=expected:
            continue
        result={}
        for goal,name in GOALS.items():
            rs=[r for r in items.values() if r["goal"]==goal]
            result[name]={"first":dict(Counter("/".join(r["first"]) or "none" for r in rs)),
                          "ever":{g:sum(r["ever"][g] for r in rs) for g in GOALS.values()},
                          "terminal":{g:sum(r["terminal"][g] for r in rs) for g in GOALS.values()}}
        result["both_old_first"]=sum(items[i,"A"]["first"]==["left"] and items[i,"B"]["first"]==["right"]
                                     for i,g in expected if g=="A")
        outcomes[f"{key[0]}_{key[1]}"]=result
    return outcomes,group


def score(o):
    return (o["left"]["first"].get("left",0),o["right"]["first"].get("right",0),o["front"]["terminal"]["front"])


def select(o,arms):
    parent,full=score(o["0_parent"]),score(o["20261021_full"])
    reference=(*parent[:2],full[2])
    points=[]
    for a in arms:
        row=o[f"20261021_{a}"]
        n=score(row)
        eligible=all(x>=y-z for x,y,z in zip(n,reference,(4,4,2),strict=True))
        points.append({"arm":a,"score":n,"eligible":eligible,
                       "min_reference_ratio":min(x/max(y,1) for x,y in zip(n,reference)),
                       "old_terminal":sum(row[g]["terminal"][g] for g in ("left","right"))})
    points.sort(key=lambda r:(r["eligible"],r["min_reference_ratio"],r["old_terminal"]),reverse=True)
    return {"rank":points,"candidate":points[0]["arm"],"passes_gate":points[0]["eligible"],"reference":reference}


def gradient(path,partial):
    result={}
    for s in (0,30,100,300):
        file=path/f"gradient_{s:06d}.jsonl"
        if partial and not file.exists():
            continue
        rows=jsonl(file)
        if not rows and partial:
            continue
        if not partial:
            assert len(rows)==8 and len({r["offset"] for r in rows})==8
        one={}
        for k in ("old_bc","mu","d"):
            one[k]={}
            for g in rows[0]["raw_front_alignment"][k]:
                vals=[r["raw_front_alignment"][k][g]["cosine"] for r in rows]
                vals=[v for v in vals if v is not None]
                one[k][g]={"median":median(vals),"negative":sum(v<0 for v in vals),"n":len(vals)} if vals else None
        updates={}
        for u in ("actual","zero","fresh"):
            vals=[r["updates"][u]["old_bc_linear"]["all"]["dot"] for r in rows]
            updates[u]={"dot":median(vals),"dot_positive":sum(v>0 for v in vals),"losses":{}}
            for k in ("old_bc","mu","d"):
                vals=[r["updates"][u]["actual_loss_delta"][k] for r in rows]
                updates[u]["losses"][k]={"median":median(vals),"positive":sum(v>0 for v in vals)}
        result[s]={"batches":len(rows),"raw":one,"updates":updates}
    return result


def probes(path):
    results={}
    for file in sorted(path.glob("probe_[0-9]*.json")):
        r=read(file)
        results[file.stem.removeprefix("probe_")]={"fixed":fixed_summary(r["fixed"]),
                                                 "front_demo":demo_summary(r["front_demo"])}
    return results


def transitions(groups,candidate):
    result={}
    for s in (20261121,20261122):
        if (s,candidate) not in groups or (s,"full") not in groups:
            continue
        a,b=groups[s,candidate],groups[s,"full"]
        if len(a)!=60 or len(b)!=60:
            continue
        by_goal={}
        for g,name in GOALS.items():
            cells=[(a[i,g],b[i,g]) for i,k in a if k==g]
            by_goal[name]={"first_correct_delta":sum((p["first"]==[name])-(q["first"]==[name]) for p,q in cells),
                           "terminal_delta":sum(p["terminal"][name]-q["terminal"][name] for p,q in cells),
                           "first_transitions":dict(Counter(
                               f'{"/".join(q["first"]) or "none"} -> {"/".join(p["first"]) or "none"}'
                               for p,q in cells))}
        result[s]=by_goal
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("path",type=Path,nargs="?",default=ROOT)
    p.add_argument("--mode",choices=("select","summary","partial"),default="summary")
    p.add_argument("--arms",nargs="+",default=ARMS)
    a=p.parse_args()
    path=a.path
    new=[r for f in path.glob("dev_w[0-3].jsonl") for r in jsonl(f)]
    ref=jsonl(path/"baseline_dev.jsonl")
    assert len(ref)==120
    dev,_=native([*new,*ref],range(20,40))
    screen=select(dev,a.arms) if all(f"20261021_{x}" in dev for x in a.arms) else None
    if a.mode=="select":
        assert screen and len(new)==len(a.arms)*60
        print(json.dumps(screen,indent=2,allow_nan=False))
        return
    rows=[r for f in path.glob("confirm_w[0-5].jsonl") for r in jsonl(f)]
    confirm,groups={},{}
    if rows:
        ids=[r["init"] for r in read(path/"holdout.json")["selected"]]
        confirm,groups=native(rows,ids)
    result={"dev_episodes":len(new),"dev":dev,"screen":screen,"confirm_episodes":len(rows),"confirm":confirm,
            "probe":probes(path),"gradient":gradient(path,a.mode=="partial"),
            "layer_effects":{f.stem:read(f)["effects"] for f in sorted(path.glob("mu_layers_*.json"))}}
    if screen:
        result["confirm_transitions"]=transitions(groups,screen["candidate"])
    if a.mode=="summary":
        assert screen and len(new)==len(a.arms)*60
        assert len(result["layer_effects"])==2
        assert len(rows)==300 and len(confirm)==5
        assert len(result["probe"])>=len(a.arms)*4+8
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
