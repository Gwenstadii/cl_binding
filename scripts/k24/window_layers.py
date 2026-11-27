"""Localize K24 early-window drift below the whole-module level."""
from collections import defaultdict
import argparse
from statistics import median

import torch
from safetensors.torch import load_file

from window_eval import RUN, Lab5, assign, group, load_weights, make_bank, responses, write


def partition(name):
    k=group(name)
    if k in ("kv","cross_other","mlp","self_attention","expert_norm"):
        i=int(name.split(".layers.")[1].split(".")[0])
        return f"{k}_{4*(i//4):02d}-{4*(i//4)+3:02d}"
    return k


def reduce(rows,anchor,full):
    per_init=defaultdict(list)
    for r,a,f in zip(rows,anchor,full,strict=True):
        d,u,v=(x["10"] for x in (r,a,f))
        i=r["init"]
        assert i==a["init"]==f["init"]
        h=[x["contrast_by_channel"]["H"]["gain"] for x in (d,u,v)]
        # Report absolute changes relative to each background to avoid ratios near zero.
        per_init[i].append({"h_gain_delta_from_anchor":h[0]-h[1],
                            "h_gain_delta_from_full":h[0]-h[2],
                            **{f"mu_{c}_rms_delta_from_anchor":d["mu_drift_rms"][c]-u["mu_drift_rms"][c]
                               for c in "HZR"},
                            **{f"mu_{c}_rms_delta_from_full":d["mu_drift_rms"][c]-v["mu_drift_rms"][c]
                               for c in "HZR"}})
    return {k:median(median(r[k] for r in x) for x in per_init.values())
            for k in next(iter(per_init.values()))[0]}


def run(inputs):
    torch.set_num_threads(2)
    lab=Lab5()
    try:
        refs=make_bank(lab,(20,23,26,29,32,35,38,39))
        base=load_weights("anchor")
        end=load_weights("full")
        assign(lab,"F",base)
        anchor=responses(lab,refs,True)
        assign(lab,"F",end)
        full=responses(lab,refs,True)
        parts=defaultdict(list)
        for name in end:
            if inputs:
                if group(name)=="action_time":
                    parts[name.rsplit(".",1)[0]].append(name)
            else:
                parts[partition(name)].append(name)
        summary={}
        for label,names in sorted(parts.items()):
            keys=set(names)
            # Skip blocks with precisely zero realized change.
            if all(torch.equal(base[n],end[n]) for n in names):
                continue
            summary[label]={}
            for arm in ("add","remove"):
                assign(lab,"F",{n:end[n] if (n in keys)==(arm=="add") else base[n] for n in end})
                raw=responses(lab,refs,True)
                write(RUN/f"layer_{label}_{arm}.json",raw)
                summary[label][arm]=reduce(raw,anchor,full)
            print("LAYERS",label,flush=True)
        tag="input" if inputs else "layer"
        write(RUN/f"{tag}_ref.json",{"anchor":anchor,"full":full})
        write(RUN/f"{tag}_summary.json",summary)
    finally:
        lab.ctx.env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",action="store_true")
    run(p.parse_args().inputs)
