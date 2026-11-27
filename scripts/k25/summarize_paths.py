"""Verify K25 path trials and compute matched endpoint and local-readout effects."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from summarize import IDS, METRICS, TARGET, endpoint, lines, margin, paired, paired_interval, probe, read

RELAY = {"repair_whole": "AB", "repair_mean": "AB", "repair_diff": "AB", "repair_flip": "B",
         "reverse_mean": "B", "reverse_diff": "B"}
COMBO = {"v_on_anchor": "AB", "v_mlp_on_anchor": "AB"}


def native_plan(chosen):
    components = ("gate", "up", "down")
    keep = "mlp_" + "_".join(x for x in components if x != chosen["mlp"])
    return {"cross_k": "AB", "cross_v": "AB", "cross_kv": "AB", "deep_k": "B", "deep_v": "B",
            **{f"mlp_{x}": "AB" if x == chosen["mlp"] else "B" for x in components}, keep: "AB"}


def load_trials(root, plan, prefix, lanes):
    result = {}
    for lane in range(lanes):
        marker = read(root / f"{prefix}_{lane}_COMPLETE.json")
        assert marker["lane"] == lane and marker["lanes"] == lanes
        if prefix == "native":
            assert marker["gate"] == IDS[lane] and marker["plan"] == plan
        if prefix == "relay_path":
            gate = marker["gate"]
            assert gate["init"] == IDS[lane] and gate["token_difference_indices"] == [9]
            assert all(gate[k] for k in ("reverse_True_whole_chunk_bitwise", "reverse_False_whole_chunk_bitwise",
                                         "value_decomposition_bitwise", "fp32_split_readout_verified"))
        actual = lines(root / f"{prefix}_{lane}.jsonl")
        expected = {(name, i, q) for name, qs in plan.items() for i in IDS[lane::lanes] for q in qs}
        assert len(actual) == len(expected)
        for row in actual:
            key = row["arm"], row["init"], row["goal"]
            assert key in expected and key not in result
            result[key] = row
    return result


def add_refs(rows, k24, previous, combined):
    for lane in range(4):
        for r in lines(k24 / f"native_{lane}.jsonl"):
            if r["arm"] in ("full", "anchor") and r["goal"] in "AB":
                rows[r["arm"], r["init"], r["goal"]] = r
    for lane in range(3):
        for r in lines(previous / f"deep_kv_{lane}.jsonl"):
            if r["arm"] == "rollback":
                rows["deep_kv", r["init"], r["goal"]] = r
    for lane in range(8):
        for r in lines(combined / f"eval_{lane}.jsonl"):
            if r["arm"] == "keep_mlp" and r["goal"] in "AB":
                rows["mlp_gate_up_down", r["init"], r["goal"]] = r


def scoreboard(rows, plans):
    result = {}
    for arm, goals in plans.items():
        result[arm] = {q: endpoint([rows[arm, i, q] for i in IDS], q) for q in goals}
        if goals == "AB":
            result[arm]["paired_correct"] = sum(
                all(rows[arm, i, q]["first"] == [TARGET[q]] for q in goals) for i in IDS)
    return result


def right_diff(rows, arm, base):
    return paired([rows[base, i, "B"] for i in IDS], [rows[arm, i, "B"] for i in IDS], "right")


def group_six(data, fn):
    by_case = defaultdict(list)
    for row in data:
        by_case[row["init"]].append(row)
    assert sorted(by_case) == IDS and all(len(group) == 6 for group in by_case.values())
    return {i: fn(group) for i, group in by_case.items()}


def relay_fixed(directory):
    markers = read(directory / "relay_path_probe_COMPLETE.json")
    data = {}
    for arm in markers["arms"]:
        group = group_six(read(directory / f"relay_path_probe_{arm}.json"),
                          lambda rs: {k: median(r[k] for r in rs) for k in ("h_gain", "mu_h", "mu_z", "mu_r")})
        data[arm] = {k: median(v[k] for v in group.values()) for k in next(iter(group.values()))}
    assert set(data) == {"base_full", "base_rolled", *RELAY}
    assert data["base_rolled"] == data["repair_whole"]
    return data


def up_fixed(directory):
    data = read(directory / "up_relay.json")
    assert data["whole_bitwise"]
    assert data["results"]["base_protected"] == data["results"]["repair_whole"]
    metrics = {a: median(group_six(rs, lambda g: median(r["h_gain"] for r in g)).values())
               for a, rs in data["results"].items()}
    checks = {a: {k: median(r[k] for r in rs) for k in ("rmse_first10", "h_gain_abs_difference")}
              for a, rs in data["batch_checks"].items()}
    return {"horizontal_gain": metrics, "paired_batch_vs_singleton_median_error": checks}


def tracer(directory, family):
    rows = read(directory / f"trace_{family}.json")
    layers = list(range(1, 16, 2)) if family == "v" else list(range(16))
    key = "local_v_correction_rms" if family == "v" else "local_up_correction_rms"
    assert len(rows) == len(IDS) * len(layers) * 6
    assert len({(r["init"], r["draft"], r["step"], r["layer"]) for r in rows}) == len(rows)
    per_layer = {}
    for i in layers:
        data = group_six([r for r in rows if r["layer"] == i], lambda v: v)
        keys = [key, "local_mean_rms", "local_diff_rms", "local_mean_signed_share", "local_diff_signed_share"]
        if family == "v":
            keys.append("k_changes_attention_total_variation")
        per_layer[i] = {k: median(median(r[k] for r in g) for g in data.values()) for k in keys}
    by_init = {i: [r for r in rows if r["init"] == i] for i in IDS}

    def weighted(data, lookup):
        energy = [r[key] ** 2 for r in data]
        return sum(e * lookup(r) for e, r in zip(energy, data, strict=True)) / sum(energy)

    share = {s: median(weighted(g, lambda r: r[f"local_{s}_signed_share"]) for g in by_init.values())
             for s in ("mean", "diff")}
    spans = {s: median(weighted(g, lambda r: r["spans"][s]["total_signed_share"]) for g in by_init.values())
             for s in ("image", "text", "state")} if family == "v" else {}
    return {"layers": per_layer, "correction_energy_weighted_signed_share_median": share,
            "weighted_modal_span_share_median": spans}


def summarize(args):
    d = args.directory
    choice = read(d / "selected.json")
    native = native_plan(choice)
    rows = {**load_trials(d, native, "native", args.native_lanes),
            **load_trials(d, RELAY, "relay_path", args.relay_lanes),
            **load_trials(d, COMBO, "combo", args.combo_lanes)}
    total = len(rows)
    add_refs(rows, args.k24, args.previous, args.combined)
    score = scoreboard(rows, {**dict.fromkeys(("full", "anchor", "deep_kv", "mlp_gate_up_down"), "AB"),
                              **native, **RELAY, **COMBO})
    comparisons = [(a, ("anchor" if a.startswith(("mlp_", "v_")) else "full")) for a in score if a != "full"]
    comparisons = [(a, "cross_v") if a.startswith("reverse_") else (a, b) for a, b in comparisons if a != b]
    comparisons.extend([
        ("cross_v", "cross_k"), ("cross_kv", "cross_v"),
        ("repair_diff", "repair_mean"), ("repair_flip", "repair_diff"),
        ("v_mlp_on_anchor", "mlp_gate_up_down"), ("v_mlp_on_anchor", "v_on_anchor"),
        ("mlp_" + "_".join(x for x in ("gate", "up", "down") if x != choice["mlp"]), "mlp_gate_up_down"),
    ])
    fixed = {}
    fixed_by_init = {}
    for file in sorted(d.glob("probe_held_*.json")):
        if file.stem.endswith("COMPLETE"):
            continue
        arm = file.stem.removeprefix("probe_held_")
        fixed_by_init[arm] = data = probe(file)
        fixed[arm] = {k: median(v[k] for v in data.values()) for k in METRICS}
    cases = [("v_mlp_on_anchor", 1), ("v_on_anchor", -1), ("mlp_gate_up_down", -1), ("anchor", 1)]
    probes = [sum(mult * fixed_by_init[a][i]["horizontal_gain"] for a, mult in cases) for i in IDS]
    endpoints = [sum(mult * (rows[a, i, "B"]["first"] == ["right"]) for a, mult in cases) for i in IDS]
    motion = [sum(mult * margin(rows[a, i, "B"]) for a, mult in cases) for i in IDS]
    results = {"new_episodes": total, "selection": choice, "fixed": fixed, "relay_fixed": relay_fixed(d),
               "up_fixed": up_fixed(d),
               "closed_loop": score,
               "right_paired": {f"{a}_vs_{b}": right_diff(rows, a, b) for a, b in comparisons},
               "v_trace": tracer(d, "v"), "up_trace": tracer(d, "mlp"),
               "joint_update_interaction": {
                   "h_gain_paired_median": median(probes), "h_gain_paired_mean_ci": paired_interval(probes),
                   "right_risk": mean(endpoints), "right_risk_ci": paired_interval(endpoints),
                   "right_margin_mm_mean": mean(motion), "right_margin_mm_ci": paired_interval(motion)}}
    (d / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("k24", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("combined", type=Path)
    p.add_argument("--native-lanes", type=int, default=6)
    p.add_argument("--relay-lanes", type=int, default=6)
    p.add_argument("--combo-lanes", type=int, default=4)
    summarize(p.parse_args())
