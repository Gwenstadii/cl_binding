"""Validate the K20 bridge and summarize both direct interventions."""
import argparse
from collections import Counter
import json
from math import dist
from pathlib import Path
from statistics import median

SUITES = {"dev": (20261021, range(20, 40)), "confirm": (20261121, range(200, 220))}
TARGETS = {"A": "left", "B": "right", "N": "front"}
PATCHES = {"swap_d": "AB", "swap_mu": "AB", "swap_both": "B", "remove_d": "B"}
CENTERS = ("left", "right", "front", "back")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()]


def label(row):
    return "/".join(row["first"]) or "none"


def margin(row, y):
    p = row["pos"]
    end = [x + dx for x, dx in zip(p, y, strict=True)]
    scores = [dist(p, c) - dist(end, c) for c in row["centers"]]
    target = CENTERS.index(row["goal"])
    return scores[target] - max(s for j, s in enumerate(scores) if j != target)


def describe_margin(values):
    return {"median_mm": 1000 * median(values), "positive": sum(x > 0 for x in values)}


def axis_components(a, b, ya, yb):
    left, right = a["centers"][:2]
    assert a["pos"] == b["pos"] and a["centers"] == b["centers"]
    u = [(r - l) / dist(left, right) for l, r in zip(left, right, strict=True)]
    la, rb = (sum(x * v for x, v in zip(y, u, strict=True)) for y in (ya, yb))
    return [(la + rb) * .5, (rb - la) * .5]


def axis_summary(values):
    mu, d = zip(*values, strict=True)
    return {"mu_toward_right_mm": median(mu) * 1000, "d_strength_mm": median(d) * 1000}


def describe(rows, q):
    target = TARGETS[q]
    result = {"first": dict(Counter(map(label, rows))), "terminal": sum(r["terminal"][target] for r in rows)}
    if q == "N" and all("first_stable_grasp" in r for r in rows):
        result.update(grasp=sum(r["first_stable_grasp"] is not None for r in rows),
                      reached=sum(r["reached"][target] is not None for r in rows))
    return result


def index(rows):
    by_key = {(r["arm"], r["init"], r["goal"]): r for r in rows}
    assert len(by_key) == len(rows)
    return by_key


def motion(root, prior, suite, ids, has_probe):
    original = [r for lane in range(2) for r in lines(prior / f"transport_{suite}_{lane}.jsonl")]
    old = {(r["arm"], r["init"], r["q"]): r for r in original}
    assert len(old) == len(original) == 160
    recomposed = {}
    for q, s in (("A", 1), ("B", -1)):
        for mu in ("full", "route"):
            for d in ("full", "route"):
                values = []
                for i in ids:
                    m1, m2 = (old[mu, i, text]["observed"] for text in "AB")
                    d1, d2 = (old[d, i, text]["observed"] for text in "AB")
                    y = [(a + b + s * (c - e)) * .5 for a, b, c, e in zip(m1, m2, d1, d2, strict=True)]
                    values.append(margin(old[mu, i, q], y))
                recomposed[f"{q}:{mu}:{d}"] = describe_margin(values)
    vectors = {}
    for arm in ("full", "route", "weak", "replay_full"):
        vec = []
        for i in ids:
            a, b = (old[arm, i, q] for q in "AB")
            vec.append(axis_components(a, b, a["observed"], b["observed"]))
        vectors[arm] = vec
    axis = {arm: axis_summary(v) for arm, v in vectors.items()}
    axis["route_minus_full_paired"] = axis_summary([
        [v - w for v, w in zip(a, b, strict=True)] for a, b in zip(vectors["route"], vectors["full"], strict=True)])
    new = {}
    if has_probe:
        probe = read(root / f"probe_{suite}.json")
        assert len(probe["fixed"]) == 120 and len(probe["motion"]) == 40
        for q in "AB":
            records = [r["trajectory"] for r in probe["motion"] if r["q"] == q]
            assert len(records) == len(ids)
            new[q] = {j: describe_margin([r[j // 10 - 1]["margin_m"] for r in records])
                      for j in (10, 20, 30, 40)}
        trained = {(r["init"], r["q"]):r for r in probe["motion"]}
        values = []
        for i in ids:
            a, b = (trained[i, q] for q in "AB")
            values.append(axis_components(a, b, a["trajectory"][0]["observed"],
                                          b["trajectory"][0]["observed"]))
        axis["d_train"] = axis_summary(values)
    return old, recomposed, axis, new


def summary(root, prior, partial):
    references = lines(prior / "baseline.jsonl")
    result = {}
    for suite, (seed, ids) in SUITES.items():
        base = index([r for r in references if r["suite"] == suite])
        for lane in range(4):
            for row in lines(prior / f"{suite}_{seed}_w{lane}.jsonl"):
                base[row["arm"], row["init"], row["goal"]] = row
        new = {}
        for mode, arms in (("patch", PATCHES), ("native", {"d_train": "ABN"})):
            files = sorted(root.glob(f"{mode}_{suite}_[01].jsonl"))
            rows = [r for path in files for r in lines(path)]
            lookup = index(rows)
            wanted = {(a, i, q) for a, qs in arms.items() for i in ids for q in qs}
            assert set(lookup) <= wanted and all(r["seed"] == seed for r in rows)
            if not partial:
                assert set(lookup) == wanted
                assert all((root / f"{mode}_{suite}_{j}_COMPLETE.json").exists() for j in range(2))
            new.update(lookup)
        all_rows = base | new
        outcomes, transitions = {}, {}
        for arm, qs in {"full": "ABN", "route": "ABN", "weak": "ABN", "replay_full": "ABN",
                        "d_train": "ABN", **PATCHES}.items():
            outcomes[arm] = {}
            for q in qs:
                if not all((arm, i, q) in all_rows for i in ids):
                    continue
                outcomes[arm][q] = describe([all_rows[arm, i, q] for i in ids], q)
                if arm in PATCHES or arm == "d_train":
                    ref = "route" if arm == "remove_d" else "full"
                    transitions[f"{arm}:{q}"] = dict(Counter(
                        f"{label(base[ref,i,q])} -> {label(new[arm,i,q])}" for i in ids))
            if {"A", "B"} <= outcomes[arm].keys():
                outcomes[arm]["both_old_first"] = sum(
                    all_rows[arm,i,"A"]["first"] == ["left"] and all_rows[arm,i,"B"]["first"] == ["right"]
                    for i in ids)
        folder = root / "runs" / f"{seed}_d"
        train = None
        if (folder / "COMPLETE.json").exists():
            assert read(folder / "COMPLETE.json")["frozen_unchanged"]
            train_rows = lines(folder / "train_steps.jsonl")
            reference = lines(prior / f"train_ref_{seed}.jsonl")
            assert len(train_rows) == len(reference) == 1000
            assert all((r["step"], r["indices"], r["lr"]) == (b["step"], b["indices"], b["lr"])
                       for r, b in zip(train_rows, reference, strict=True))
            assert abs(train_rows[0]["front_loss"] - reference[0]["loss"]) < 1e-6
            checks = [r["old"] for r in train_rows if r["old"] is not None]
            assert len(checks) == 167
            train = {"checks": len(checks), "old_queries": 2 * len(checks),
                     "active": sum(r["loss"] > 0 for r in checks),
                     "first_front_loss_matches": True, "all_front_batches_and_lr_match": True,
                     "frozen_unchanged": True, "seconds": train_rows[-1]["seconds"]}
        elif not partial:
            raise FileNotFoundError(folder / "COMPLETE.json")
        probe_path = root / f"probe_{suite}.json"
        if not partial:
            assert probe_path.exists()
        old_motion, recomposed, axis, trained_motion = motion(root, prior, suite, ids, probe_path.exists())
        patched_motion = {}
        for arm, qs in PATCHES.items():
            for q in qs:
                if q not in outcomes[arm]:
                    continue
                patched_motion[f"{arm}:{q}"] = {j: describe_margin([
                    margin(old_motion["full", i, q], new[arm, i, q]["early"][j // 10 - 1]["displacement"][:2])
                    for i in ids]) for j in (10, 20, 30, 40)}
        result[suite] = {"seed": seed, "new_episodes": len(new), "outcomes": outcomes, "paired": transitions,
                         "algebra_first_block": recomposed, "axis_first_block": axis, "patch_motion": patched_motion,
                         "trained_motion": trained_motion, "training": train,
                         "fixed_d_train": read(probe_path)["fixed_summary"] if probe_path.exists() else None}
    return result


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, nargs="?", default=here)
    p.add_argument("prior", type=Path, nargs="?", default=here.parent / "k20_route_margin_20261112_01")
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    print(json.dumps(summary(a.root, a.prior, a.partial), indent=2, allow_nan=False))

