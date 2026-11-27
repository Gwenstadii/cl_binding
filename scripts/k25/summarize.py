"""Verify K25 coverage and compute per-initial-state counterfactual contrasts."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from math import comb, dist
from pathlib import Path
from random import Random
from statistics import median

BLOCKS = ("deep_kv", "deep_mlp")
CONDS = {"rollback": "AB", "add_only": "B", "repair_d": "B", "reverse_d": "B",
         "repair_parallel": "B", "reverse_parallel": "B", "repair_mu": "B"}
JOINT_CONDS = {k: CONDS[k] for k in ("rollback", "repair_d", "reverse_d")}
REFERENCES = {"rollback": "full", "add_only": "anchor", "repair_d": "full",
              "reverse_d": "rollback", "repair_parallel": "full", "reverse_parallel": "rollback",
              "repair_mu": "full"}
TARGET = {"A": "left", "B": "right"}
IDS = list(range(200, 220))
METRICS = ("gain", "horizontal_gain", "norm_ratio", "cosine", "noncollinear_ratio",
           "mu_h", "mu_z", "mu_r", "mu_g")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def label(row):
    return "/".join(row["first"]) or "none"


def margin(row):
    p = row["position_xy_m"]
    end = [a + b for a, b in zip(p, row["displacement_xy_m"], strict=True)]
    scores = {k: dist(p, c) - dist(end, c) for k, c in row["centers_xy_m"].items()}
    goal = TARGET[row["goal"]]
    return 1000 * (scores[goal] - max(v for k, v in scores.items() if k != goal))


def endpoint(rows, q):
    return {"n": len(rows), "first": dict(Counter(map(label, rows))),
            "terminal": sum(r["terminal"][TARGET[q]] for r in rows),
            "margin_mm_median": median(map(margin, rows))}


def paired_interval(values):
    rng = Random(20261118)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(10000))
    return [draws[250], draws[9749]]


def paired(base, changed, target):
    changes = [int(b["first"] == [target]) - int(a["first"] == [target])
               for a, b in zip(base, changed, strict=True)]
    gains, losses = changes.count(1), changes.count(-1)
    discordant = gains + losses
    arrivals = [margin(b) - margin(a) for a, b in zip(base, changed, strict=True)]
    return {"n": len(changes), "correct_gains": gains, "correct_losses": losses,
            "first_transitions": dict(Counter(f"{label(a)} -> {label(b)}"
                                             for a, b in zip(base, changed, strict=True))),
            "right_margin_change_median_mm": median(arrivals),
            "risk_difference": (gains - losses) / len(changes),
            "paired_bootstrap_95": paired_interval(changes),
            "mcnemar_exact_p_descriptive": min(1., 2 * sum(comb(discordant, k) for k in range(min(gains, losses) + 1))
                                              / 2**discordant)}


def probe(path):
    data = defaultdict(list)
    rows = lines(path) if path.suffix == ".jsonl" else read(path)
    for row in rows:
        r = row["10"]
        data[row["init"]].append({"gain": r["gain"], "horizontal_gain": r["contrast_by_channel"]["H"]["gain"],
                                  "norm_ratio": r["norm_ratio"], "cosine": r["cosine"],
                                  "noncollinear_ratio": r["noncollinear_ratio"],
                                  **{f"mu_{k.lower()}": r["mu_drift_rms"][k] for k in "HZRG"}})
    assert sorted(data) == IDS and all(len(r) == 6 for r in data.values())
    return {i: {k: median(r[k] for r in data[i]) for k in METRICS} for i in IDS}


def summarize(directory, k24, partial):
    reference = {}
    for lane in range(4):
        for row in lines(k24 / f"native_{lane}.jsonl"):
            if row["arm"] in ("full", "anchor", "remove_cross", "remove_mlp") and row["goal"] in "AB":
                reference[row["arm"], row["init"], row["goal"]] = row
    assert len(reference) == 160
    results = {"reference": {
        arm: {q: endpoint([reference[arm, i, q] for i in IDS], q) for q in "AB"}
        for arm in ("full", "anchor", "remove_cross", "remove_mlp")}, "blocks": {}}
    inventory = read(directory / "module_inventory.json")
    if (directory / "joint_inventory.json").exists() or not partial:
        joint = read(directory / "joint_inventory.json")
        for checkpoint in ("full", "anchor"):
            assert inventory[f"{checkpoint}_sha256"] == joint[f"{checkpoint}_sha256"]
        inventory["deep_joint"] = joint["deep_joint"]
    for parent in ("full", "anchor"):
        expected = next(s.split()[0] for s in (k24 / "weight_sha256.txt").read_text().splitlines()
                        if s.split()[1] == f"w_{parent}_130.safetensors")
        assert inventory[f"{parent}_sha256"] == expected
    probe_data = {}
    for a in ("full", "anchor", *(x for b in BLOCKS for x in (b, f"{b}_add")), "deep_joint"):
        path = directory / f"probe_{a}.json"
        if path.exists() or not partial:
            probe_data[a] = probe(path)
    results["fixed_probes"] = {a: {k: median(v[k] for v in rows.values()) for k in METRICS}
                               for a, rows in probe_data.items()}
    rollbacks = {}
    for block in (*BLOCKS, "deep_joint"):
        if block not in inventory:
            continue
        lanes, conditions = (2, JOINT_CONDS) if block == "deep_joint" else (3, CONDS)
        rows = {}
        for lane in range(lanes):
            path = directory / f"{block}_{lane}.jsonl"
            if partial and not path.exists():
                continue
            if not partial:
                marker = read(path.with_suffix(".complete.json"))
                assert marker["case_ids"] == IDS[lane::lanes] and marker["tensors"] == inventory[block]["tensors"]
                assert marker["baseline_identity_init"] == IDS[lane] and marker["same_source_chunk_identity"]
            for row in lines(path):
                key = row["arm"], row["init"], row["goal"]
                assert row["block"] == block and key not in rows and key[1] in IDS[lane::lanes]
                rows[key] = row
        if not partial:
            assert rows.keys() == {(a, i, q) for a, qs in conditions.items() for i in IDS for q in qs}
        data = {"episodes": len(rows), "outcomes": {}, "paired": {}, "weights": inventory[block]}
        for a, qs in conditions.items():
            data["outcomes"][a] = {}
            for q in qs:
                ids = [i for i in IDS if (a, i, q) in rows]
                if not ids:
                    continue
                r = [rows[a, i, q] for i in ids]
                data["outcomes"][a][q] = endpoint(r, q)
                ref = REFERENCES[a]
                source = rows if ref == "rollback" else reference
                if all((ref, i, q) in source for i in ids):
                    data["paired"][f"{a}_{q}_vs_{ref}"] = paired([source[ref, i, q] for i in ids], r, TARGET[q])
        if all(("rollback", i, q) in rows for i in IDS for q in "AB"):
            data["both_old_first_rollback"] = sum(
                all(rows["rollback", i, q]["first"] == [TARGET[q]] for q in "AB") for i in IDS)
            rollbacks[block] = [rows["rollback", i, "B"] for i in IDS]
        for arm, ref in ((block, "full"), (f"{block}_add", "anchor")):
            if arm in probe_data and ref in probe_data:
                data[f"{arm}_minus_{ref}_probe_paired_median"] = {
                    k: median(probe_data[arm][i][k] - probe_data[ref][i][k] for i in IDS) for k in METRICS}
        results["blocks"][block] = data
    if set(rollbacks) == {*BLOCKS, "deep_joint"}:
        baseline = [reference["full", i, "B"] for i in IDS]
        interaction = [(j["first"] == ["right"]) - (k["first"] == ["right"])
                       - (m["first"] == ["right"]) + (b["first"] == ["right"])
                       for j, k, m, b in zip(rollbacks["deep_joint"], rollbacks["deep_kv"],
                                             rollbacks["deep_mlp"], baseline, strict=True)]
        results["exploratory_joint_interaction"] = {
            "mean_risk_interaction": sum(interaction) / len(interaction),
            "paired_bootstrap_95": paired_interval(interaction)}
    path = directory / ("partial.json" if partial else "summary.json")
    if not partial:
        assert read(directory / "probe_COMPLETE.json")["arms"] == [
            "full", "anchor", "deep_kv", "deep_kv_add", "deep_mlp", "deep_mlp_add"]
        assert read(directory / "joint_probe_COMPLETE.json")["arms"] == ["deep_joint"]
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("k24", type=Path)
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    summarize(a.directory, a.k24, a.partial)
