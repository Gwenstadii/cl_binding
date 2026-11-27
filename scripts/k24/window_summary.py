"""Verify and summarize K24 continuous training, module swaps and response swaps."""
import argparse
from collections import Counter, defaultdict
import json
from math import dist
from pathlib import Path
from statistics import mean, median

TRAIN = ("full","protect","matched","freeze")
HYBRIDS = ("anchor","only_kv","only_cross","only_mlp","only_other","remove_kv",
           "remove_cross","remove_mlp","remove_other")
PATCH = ("repair_mu","repair_d","repair_mu_d","reverse_mu","reverse_d","repair_parallel","reverse_parallel")
GOALS = {"A":"left","B":"right","N":"front"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def label(row):
    return "/".join(row["first"]) or "none"


def margin(row):
    p = row["position_xy_m"]
    end = [a+b for a,b in zip(p,row["displacement_xy_m"],strict=True)]
    val = {k:dist(p,c)-dist(end,c) for k,c in row["centers_xy_m"].items()}
    expected=GOALS[row["goal"]]
    return 1000*(val[expected]-max(v for k,v in val.items() if k!=expected))


def outcomes(directory, tag, partial):
    rows={}
    layout={a:"B" for a in PATCH} if tag=="patch" else {
        **{a:"ABN" for a in (*TRAIN,"anchor")}, "remove_cross":"AB","remove_mlp":"AB"}
    if tag=="inputnative":
        layout={a:"B" for a in ("remove_time_out","repair_mu","reverse_mu")}
    if tag=="crossnative":
        layout={a:"B" for a in ("repair_d","reverse_d")}
    ids=range(200,220)
    for lane in range(4):
        path=directory/f"{tag}_{lane}.jsonl"
        if partial and not path.exists():
            continue
        if not partial:
            assert read(directory/f"{tag}_{lane}_COMPLETE.json")=={"lane":lane,"lanes":4}
        for r in lines(path):
            k=(r["arm"],r["init"],r["goal"])
            assert k not in rows and k[1] in ids and k[2] in layout[k[0]]
            rows[k]=r
    if not partial:
        assert rows.keys()=={(a,i,q) for a,qs in layout.items() for i in ids for q in qs}
    report={}
    for a,qs in layout.items():
        sections={}
        for q in qs:
            r=[rows[a,i,q] for i in ids if (a,i,q) in rows]
            if not r:
                continue
            stats={"n":len(r),"first":dict(Counter(map(label,r))),
                   "terminal":sum(x["terminal"][GOALS[q]] for x in r)}
            if q!="N":
                stats["median_first_block_margin_mm"]=median(map(margin,r))
            sections[q]=stats
        if "A" in qs:
            sections["both_old_first"]=sum(all((a,i,q) in rows and rows[a,i,q]["first"]==[GOALS[q]]
                                               for q in "AB") for i in ids)
        report[a]=sections
    return rows,report


def probes(directory, partial):
    result={}
    for a in (*TRAIN,*HYBRIDS):
        path=directory/f"probe_{a}.json"
        if partial and not path.exists():
            continue
        data=read(path)
        assert len(data)==120 and len({r["init"] for r in data})==20
        by={i:[r["10"] for r in data if r["init"]==i] for i in range(20,40)}
        assert all(len(x)==6 for x in by.values())
        metrics={"horizontal_gain":lambda r:r["contrast_by_channel"]["H"]["gain"],
                 "all_gain":lambda r:r["gain"],
                 **{f"mu_{c}":lambda r,c=c:r["mu_drift_rms"][c] for c in "HZRG"},
                 "off_axis":lambda r:r["noncollinear_ratio"]}
        result[a]={name:median(median(f(r) for r in lst) for lst in by.values()) for name,f in metrics.items()}
    return result


def training(directory):
    data={a:lines(directory/f"{a}.jsonl") for a in TRAIN}
    assert all(len(r)==100 for r in data.values())
    for a,r in data.items():
        assert read(directory/f"{a}_COMPLETE.json")["end"]==130
        assert all((x["step"],x["indices"],x["lr"])==(y["step"],y["indices"],y["lr"])
                   for x,y in zip(r,data["full"],strict=True))
    report={a:{"mean_last_20_front_loss":mean(x["front_loss"] for x in r[-20:]),
               "update_norm_median":median(x["actual_norm"] for x in r),
               "seconds":r[-1]["seconds"]} for a,r in data.items()}
    checks=[r for r in data["protect"] if r["controlled"] is not None]
    report["protect"].update(probes=len(checks),fractions=dict(Counter(r["fraction"] for r in checks)),
                             projection_active=sum(any(r["dual"]) for r in checks))
    ratios=[m["actual_norm"]/p["actual_norm"] for m,p in zip(data["matched"],data["protect"],strict=True)
            if p["actual_norm"]]
    report["matched"].update(norm_ratio_min=min(ratios),norm_ratio_median=median(ratios),
                             norm_ratio_max=max(ratios),within_five_percent=sum(abs(x-1)<=.05 for x in ratios),
                             compared_steps=len(ratios))
    return report


def demo_summary(directory,partial):
    output={}
    for arm in (*TRAIN,"anchor","remove_cross","remove_mlp"):
        path=directory/f"demo_{arm}.json"
        if partial and not path.exists():
            continue
        rows=read(path)
        assert len(rows)==90
        groups=defaultdict(list)
        for row in rows:
            groups[row["demo"]].append(row)
        assert len(groups)==10 and all(len(v)==9 for v in groups.values())
        result={}
        for channel in ("H","Z","R","G","all"):
            losses=[]
            for rr in groups.values():
                cells=[v for r in rr for phase in r["sq_error"].values() for k,v in phase.items()
                       if channel=="all" or channel==k]
                losses.append(sum(v["sum"] for v in cells)/sum(v["count"] for v in cells))
            result[channel]=median(losses)
        output[arm]=result
    return output


def run(directory,partial):
    native, cores=outcomes(directory,"native",partial)
    patch, changes=outcomes(directory,"patch",partial)
    inputs, selected=outcomes(directory,"inputnative",partial)
    cross, cross_changes=outcomes(directory,"crossnative",partial)
    for a in PATCH:
        ref="protect" if a.startswith("reverse") else "full"
        changes[a]["paired_right_vs_"+ref]=dict(Counter(
            label(native[ref,i,"B"])+" -> "+label(patch[a,i,"B"])
            for i in range(200,220) if (ref,i,"B") in native and (a,i,"B") in patch))
    for arm in (*TRAIN,"anchor","remove_cross","remove_mlp"):
        cores[arm]["paired_right_vs_full"]=dict(Counter(
            label(native["full",i,"B"])+" -> "+label(native[arm,i,"B"])
            for i in range(200,220) if ("full",i,"B") in native and (arm,i,"B") in native))
    for arm, base in (("repair_d","full"),("reverse_d","remove_cross")):
        cross_changes[arm]["paired_right_vs_"+base]=dict(Counter(
            label(native[base,i,"B"])+" -> "+label(cross[arm,i,"B"])
            for i in range(200,220) if (base,i,"B") in native and (arm,i,"B") in cross))
    if not partial:
        assert (directory/"layer_summary.json").is_file()
        assert set(read(directory/"probes_COMPLETE.json")["arms"])==set((*TRAIN,*HYBRIDS))
    out={"training":training(directory),"probes":probes(directory,partial),"native":cores,"patch":changes,
         "validation":demo_summary(directory,partial),
         "native_rows":len(native),"patch_rows":len(patch),"input_rows":len(inputs),"time_module":selected,
         "cross_rows":len(cross),"cross_module_d":cross_changes,
         "module_norms":read(directory/"blocks.json")}
    if (directory/"layer_summary.json").is_file():
        out["layer_probes"]=read(directory/"layer_summary.json")
    if (directory/"input_summary.json").is_file():
        out["input_probes"]=read(directory/"input_summary.json")
    (directory/("partial.json" if partial else "summary.json")).write_text(
        json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--partial",action="store_true")
    a=p.parse_args()
    run(a.directory,a.partial)
