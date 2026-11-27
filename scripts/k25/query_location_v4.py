"""Resolve the depth of the MLP write-back and verify the two local entry points."""

from contextlib import contextmanager
from statistics import median

import torch
from torch.nn import functional as F

from query_upstream_v4 import (CROSS, DEV, INSTR, RUN, Fixed, Lab5, Queries, assign, checkpoint_files,
                               make_bank, read, rolled, rms, write)
from localize_query_v4 import layer_of


@contextmanager
def replace_embedding(model, source):
    original = model.embed_suffix

    def swap(x, t):
        _, pad, mask = original(x, t)
        value, donor_pad, donor_mask = source(x, t)
        assert torch.equal(pad, donor_pad) and torch.equal(mask, donor_mask)
        return value, pad, mask

    model.embed_suffix = swap
    try:
        yield
    finally:
        model.embed_suffix = original


@contextmanager
def replace_down(model, anchor, names):
    expert = model.vlm_with_expert
    handles = []
    for name in names:
        assert name.endswith(".down_proj.weight")
        module = expert.lm_expert.layers[layer_of(name)].mlp.down_proj
        weight = anchor[name].to(module.weight.device)
        assert module.bias is None

        def swap(_, args, _out, *, w=weight):
            return F.linear(args[0], w)

        handles.append(module.register_forward_hook(swap))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def inspect_entry(lab, bank, full, anchor, by_source):
    sites = Queries(lab)
    model = lab.rt.models["F"].model
    donor = lab.rt.models["R"].model
    result = {}
    for name, names in by_source.items():
        assign(lab, "F", full)
        assign(lab, "R", rolled(full, anchor, names))
        rows = []
        seen = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, route, token, obs, x, t, _, _ in bank:
                if seen != i:
                    f, r = ({q: lab.rt.prefix(role, obs, INSTR[q]) for q in "AB"} for role in "FR")
                    seen = i
                expected_embedding = rms(donor.embed_suffix(x, t)[0] - model.embed_suffix(x, t)[0])
                if name == "down_proj":
                    assert expected_embedding == 0
                moves = []
                for q in "AB":
                    _, late = sites.capture("F", f[q], x, t)
                    target, early = sites.capture("R", r[q], x, t)
                    swap = replace_embedding(model, donor.embed_suffix) if name == "time_out" else replace_down(
                        model, anchor, names)
                    with swap:
                        actual, now = sites.capture("F", f[q], x, t)
                    assert torch.equal(target, actual) and all(torch.equal(early[l], now[l]) for l in CROSS)
                    moves.append([rms(now[l] - late[l]) for l in CROSS])
                rows.append({"init": i, "route": route, "step": token, "embedding_rms_change": expected_embedding,
                             "h_rms_both_prompts": moves, "entry_equals_weight_swap_bitwise": True})
        result[name] = rows
    write(RUN / "local_entry.json", result)
    print("LOCAL_ENTRY_MATCH", list(result), flush=True)


def depth():
    selection = read(RUN / "selection.json")
    chosen = selection["selected_parts"]
    assert chosen == ["action_time__action_time_mlp_out", "mlp__down_proj"]
    groups = read(RUN / "inventory.json")
    time_out = [n for n in groups["action_time"] if ".action_time_mlp_out." in n]
    down = [n for n in groups["mlp"] if ".down_proj." in n]
    lab = Lab5()
    full, anchor = checkpoint_files()
    try:
        assert all(torch.equal(w, anchor[n]) for n, w in full.items() if n.endswith("lm_expert.norm.weight"))
        bank = make_bank(lab, DEV)
        fixed = Fixed(lab, bank, full, anchor)
        result = {}
        bands = {f"layers_{i:02}_{i+3:02}": [n for n in down if i <= layer_of(n) <= i+3]
                 for i in range(0, 16, 4)}
        bands["layer_15"] = [n for n in down if layer_of(n) == 15]
        for tag, names in bands.items():
            data = fixed.measure(rolled(full, anchor, names))
            if tag == "layer_15":
                assert all(r["q_gain_change"] == 0 and r["query_projection"] == 0 for r in data)
            result[tag] = {"names": names, "rows": data}
            print("DOWN_DEPTH", tag, median(r["q_gain_change"] for r in data), flush=True)
        write(RUN / "down_depth.json", result)
        inspect_entry(lab, bank, full, anchor, {"time_out": time_out, "down_proj": down})
    finally:
        lab.ctx.env.close()
    write(RUN / "location_COMPLETE.json", {"parts": chosen, "drafts": len(bank), "bands": len(bands)})


if __name__ == "__main__":
    torch.set_num_threads(2)
    depth()
