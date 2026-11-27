"""Test whether deep K/V and MLP updates affect target selection through d."""

import argparse
import copy
from hashlib import file_digest
import json
from pathlib import Path
import sys

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
K24 = RUN.parent / "k24_final_window_20261117_01"
sys.path.insert(0, str(K24))
from window_eval import INSTR, K19, Lab5, WINDOW, assign, group, load_weights, measure_episode, mix, read, reset, responses, write

BLOCKS = {"deep_kv": ({"kv"}, 4), "deep_mlp": ({"mlp"}, 12), "deep_joint": ({"kv", "mlp"}, 16)}
PATCHES = {
    "repair_d": ("F", "F", "R", "R"),
    "reverse_d": ("R", "R", "F", "F"),
    "repair_parallel": ("F", "F", "R", "F"),
    "reverse_parallel": ("R", "R", "F", "R"),
    "repair_mu": ("F", "R", "F", "F"),
}
ARMS = {"rollback": "AB", "add_only": "B", **dict.fromkeys(PATCHES, "B")}
JOINT_ARMS = {"rollback": "AB", "repair_d": "B", "reverse_d": "B"}


def weight_inputs(block):
    full = load_weights("full")
    anchor = load_weights("anchor")
    kinds, count = BLOCKS[block]
    chosen = sorted(n for n in full if group(n) in kinds and
                    int(n.split(".layers.")[1].split(".")[0]) in range(12, 16))
    assert len(chosen) == count and anchor.keys() == full.keys()
    assert all(not torch.equal(full[n], anchor[n]) for n in chosen)
    rollback = {n: anchor[n] if n in chosen else v for n, v in full.items()}
    added = {n: full[n] if n in chosen else v for n, v in anchor.items()}
    return full, anchor, rollback, added, chosen


def handoff(lab, rec):
    obs = reset(lab.ctx, rec)
    assert lab.ctx.signature(obs) == rec["signature"]
    for k, control in enumerate(rec["controls"]):
        obs, _ = lab.execute(obs, np.asarray(control, dtype=np.float32), 10 * k, rec["z0"])
    assert lab.ctx.signature(obs) == rec["handoff_signature"]
    return obs


def fixed_bank(lab, records):
    bank = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for rec in records:
            obs = copy.deepcopy(handoff(lab, rec))
            prefix = {q: lab.rt.prefix("R", obs, INSTR[q]) for q in "AB"}
            for q in "AB":
                drafts = []

                def velocity(x, t):
                    drafts.append((x.clone(), t.clone()))
                    return lab.rt.query("R", prefix[q], x, t)

                lab.rt.integrate(velocity, lab.rt.noise(rec["seed"], len(rec["controls"])), 10)
                assert len(drafts) == 10
                for i in (0, 4, 8):
                    x, t = drafts[i]
                    old_a, old_b = (lab.rt.query("R", prefix[g], x, t).float() for g in "AB")
                    bank.append((rec["init"], q, i, obs, x, t, old_a, old_b))
    return bank


def probes(records, joint=False):
    lab = Lab5()
    try:
        bank = fixed_bank(lab, records)
        full, anchor, _, _, _ = weight_inputs("deep_kv")
        weights = {} if joint else {"full": full, "anchor": anchor}
        module_inventory = {}
        for block in (("deep_joint",) if joint else ("deep_kv", "deep_mlp")):
            _, _, remove, add, names = weight_inputs(block)
            weights[block] = remove
            if not joint:
                weights[f"{block}_add"] = add
            module_inventory[block] = {
                "tensors": names,
                "realized_delta_norm": sum(float((full[n].float() - anchor[n].float()).square().sum())
                                           for n in names)**.5,
            }
        for checkpoint in ("full", "anchor"):
            with (K24 / f"w_{checkpoint}_130.safetensors").open("rb") as source:
                module_inventory[f"{checkpoint}_sha256"] = file_digest(source, "sha256").hexdigest()
        write(RUN / ("joint_inventory.json" if joint else "module_inventory.json"), module_inventory)
        for arm, state in weights.items():
            assign(lab, "F", state)
            write(RUN / f"probe_{arm}.json", responses(lab, bank, True))
            print("PROBE", arm, flush=True)
    finally:
        lab.ctx.env.close()
    write(RUN / ("joint_probe_COMPLETE.json" if joint else "probe_COMPLETE.json"), {"arms": list(weights)})


def gates(lab, rec):
    obs = handoff(lab, rec)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for role in "FR":
            for q in "AB":
                seed, replan = rec["seed"], len(rec["controls"])
                expected = lab.native(role, INSTR[q], obs, seed, replan)
                actual = mix(lab, obs, q, seed, replan, (role,) * 4)
                assert torch.equal(expected, actual), (role, q, float((expected - actual).abs().max()))


def check_baseline(lab, records, lane):
    chosen = records[0]
    row = measure_episode(lab, chosen, "B")
    raw = (K24 / f"native_{(chosen['init'] - 200) % 4}.jsonl").read_text(encoding="utf-8")
    refs = [r for r in map(json.loads, raw.splitlines())
            if r["arm"] == "full" and r["goal"] == "B" and r["init"] == chosen["init"]]
    assert len(refs) == 1 and row == {k: v for k, v in refs[0].items() if k != "arm"}
    print("BASELINE_AND_PATCH_GATES", chosen["init"], lane, flush=True)
    return chosen["init"]


def worker(block, lane, records):
    full, _, rollback, added, names = weight_inputs(block)
    lab = Lab5()
    raw_native = lab.native
    try:
        lab.rt.models["O"] = copy.deepcopy(lab.rt.models["R"])
        lab.rt.pre["O"] = lab.rt.pre["R"]
        assign(lab, "R", rollback)
        assign(lab, "F", full)
        gates(lab, records[0])
        verified = check_baseline(lab, records, lane)
        path = RUN / f"{block}_{lane}.jsonl"
        with path.open("x", encoding="utf-8") as out:
            for arm, qs in (JOINT_ARMS if block == "deep_joint" else ARMS).items():
                assign(lab, "R", added if arm == "add_only" else rollback)
                for rec in records:
                    start = len(rec["controls"])
                    for q in qs:
                        if arm in PATCHES:
                            spec = PATCHES[arm]

                            def patched(role, text, obs, seed, chunk):
                                if chunk < start + WINDOW:
                                    return mix(lab, obs, q, seed, chunk, spec)
                                return raw_native(spec[0], text, obs, seed, chunk)

                            lab.native = patched
                        else:
                            def selected(role, text, obs, seed, chunk):
                                return raw_native("R", text, obs, seed, chunk)

                            lab.native = selected
                        try:
                            row = {"block": block, "arm": arm, **measure_episode(lab, rec, q)}
                        finally:
                            lab.native = raw_native
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print(block, arm, rec["init"], q, row["first"], flush=True)
    finally:
        lab.ctx.env.close()
    write(path.with_suffix(".complete.json"), {"block": block, "lane": lane, "tensors": names,
                                               "case_ids": [r["init"] for r in records],
                                               "baseline_identity_init": verified,
                                               "same_source_chunk_identity": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("probe", "worker"))
    p.add_argument("--block", choices=BLOCKS)
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    records = read(K19 / "holdout.json")["selected"]
    assert [r["init"] for r in records] == list(range(200, 220))
    if a.mode == "probe":
        probes(records, a.block == "deep_joint")
    else:
        assert a.block is not None and 0 <= a.lane < a.lanes <= len(records)
        worker(a.block, a.lane, records[a.lane::a.lanes])
