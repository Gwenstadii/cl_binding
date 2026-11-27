"""Exploratory split of the K18-C attention group by its computation at denoising."""

import argparse
import re
from statistics import median

import torch

from parameter_groups import (
    INITS, Lab5, RUN, Window, fields, make_bank, scenario_metrics, summarize_fields, write,
)


@torch.no_grad()
def splice(window, subset, add):
    for name, p in window.params.items():
        use_late = (name in subset) == add
        p.copy_(window.b[name] if use_late else window.a[name])


def run(lab, start, end):
    window = Window(lab,start,end)
    expert = window.model.model.vlm_with_expert
    assert expert.attention_mode == "cross_attn" and expert.self_attn_every_n_layers == 2
    parts = {key:set() for key in ("self", "cross", "self_kv", "self_qo", "cross_kv", "cross_qo")}
    all_attention = {n for n,k in window.groups.items() if k == "attention"}
    for name in all_attention:
        layer, proj = re.search(r"layers\.(\d+)\.self_attn\.([qkvo])_proj\.",name).groups()
        kind = "self" if int(layer)%2 == 0 else "cross"
        parts[kind].add(name)
        parts[f"{kind}_{'kv' if proj in 'kv' else 'qo'}"].add(name)
    assert len(all_attention) == 64
    assert parts["self"].isdisjoint(parts["cross"])
    assert parts["self"] | parts["cross"] == all_attention
    assert all(len(parts[k]) == 32 for k in ("self", "cross"))
    assert all(len(parts[k]) == 16 for k in ("self_kv", "self_qo", "cross_kv", "cross_qo"))
    bank = make_bank(lab,INITS)
    window.apply("early")
    early = fields(lab,bank)
    window.apply("late")
    late = fields(lab,bank)
    data = {k: summarize_fields(bank,v,early,late) for k,v in (("early",early),("late",late))}
    for k, names in parts.items():
        for add in (True,False):
            condition = f"{'add' if add else 'undo'}:{k}"
            splice(window,names,add)
            data[condition] = summarize_fields(bank,fields(lab,bank),early,late)
            print("ATTENTION_SPLIT",window.label,condition,flush=True)
    summaries = {}
    for k in parts:
        variants = (("drop", "early", f"add:{k}"), ("rescue", f"undo:{k}", "late"))
        result = {}
        for tag,c1,c2 in variants:
            for metric in ("gain_H","gain","mu_H_projection","mu_R_projection"):
                vals = [a-b for a,b in zip(scenario_metrics(data[c1],metric),scenario_metrics(data[c2],metric))]
                result[f"{tag}_{metric}"] = {"median":median(vals),"positive":sum(v>0 for v in vals)}
        result["weights"] = sorted(parts[k])
        summaries[k] = result
    write(RUN / f"attention_split_{window.label}.json", {"summary":summaries,"conditions":data})


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--start",type=int,default=30)
    p.add_argument("--ends",nargs="+",type=int,default=(100,1000))
    a=p.parse_args()
    torch.set_num_threads(2)
    lab=Lab5()
    try:
        for end in a.ends:
            run(lab,a.start,end)
    finally:
        lab.ctx.env.close()
