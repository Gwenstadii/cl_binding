"""Localize shared-response drift with exact checkpoint parameter interventions."""
from pathlib import Path
import argparse
import re
from statistics import median
import sys

import torch

from train_k19 import RUN, cross_kv, write

C=RUN.parent/"k18_parameter_groups_20261111_01"
sys.path.insert(0,str(C))
from parameter_groups import Lab5, Window, INITS, make_bank, fields, summarize_fields, scenario_metrics


@torch.no_grad()
def apply(window,names,add):
    for n,p in window.params.items():
        p.copy_(window.b[n] if (n in names)==add else window.a[n])


def split(window):
    parts={f"KV_{i}":set() for i in range(1,16,2)}
    parts.update({f"MLP_norm_{i}_{i+3}":set() for i in range(0,16,4)})
    for n,g in window.groups.items():
        m=re.search(r"\.layers\.(\d+)\.",n)
        if cross_kv(n):
            parts[f"KV_{m[1]}"].add(n)
        elif g=="mlp_norm":
            i=(int(m[1])//4)*4 if m else 12
            parts[f"MLP_norm_{i}_{i+3}"].add(n)
    assert {len(parts[f"KV_{i}"]) for i in range(1,16,2)}=={2}
    assert sum(len(v) for v in parts.values())==97
    return parts


def effects(data,parts):
    result={}
    for part,names in parts.items():
        info={"tensor_names":sorted(names)}
        for metric in ("gain_H","gain",*(f"mu_{g}_projection" for g in "HZRG")):
            # Positive means weakening d or contributing to the actual shared drift.
            order=("early",f"add:{part}",f"undo:{part}","late") if metric.startswith("gain") else (
                f"add:{part}","early","late",f"undo:{part}")
            cells=[]
            for a,b in (order[:2],order[2:]):
                ds=[x-y for x,y in zip(scenario_metrics(data[a],metric),scenario_metrics(data[b],metric))]
                cells.append({"median":median(ds),"positive":sum(d>0 for d in ds)})
            info[metric]={"add":cells[0],"withdraw":cells[1]}
        result[part]=info
    return result


def run(lab,end):
    window=Window(lab,30,end)
    parts=split(window)
    bank=make_bank(lab,INITS)
    window.apply("early")
    early=fields(lab,bank)
    window.apply("late")
    late=fields(lab,bank)
    data={n:summarize_fields(bank,v,early,late) for n,v in (("early",early),("late",late))}
    for name,weights in parts.items():
        for add in (True,False):
            apply(window,weights,add)
            mode="add" if add else "undo"
            data[f"{mode}:{name}"]=summarize_fields(bank,fields(lab,bank),early,late)
            print("LOCALIZE",window.label,mode,name,flush=True)
    result=effects(data,parts)
    write(RUN/f"mu_layers_{window.label}.json",{"effects":result,"conditions":data})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ends",nargs="+",type=int,default=(100,1000))
    a=p.parse_args()
    torch.set_num_threads(2)
    lab=Lab5()
    try:
        for end in a.ends:
            run(lab,end)
    finally:
        lab.ctx.env.close()
