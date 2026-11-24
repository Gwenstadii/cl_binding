"""Paired Scene2 rollouts for reference-action routing after Back continuation."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

from train_back import REFERENCE, RUN, read, read_weights, write

sys.path[:0] = [str(RUN.parent / "k25_deep_d_20261118_01")]
from deep_selection import INSTR, Lab5, assign, reset


OLD = ("left", "front", "right")
ALL = (*OLD, "back")
TEXT = dict(zip(ALL, (INSTR["A"], INSTR["N"], INSTR["B"],
                     "pick up the book and place it in the back compartment of the caddy"), strict=True))
ALT = "Please set the book inside the {goal} compartment of the caddy."
ALT2 = "Move the book to the caddy compartment on the {goal}."
CASES = RUN.parent / "k25v3_v_20261120_01" / "holdout.json"
MAIN_STEPS = 3000
DEV_IDS = tuple(range(630, 640))
WINDOW = {"reset": 150, "grip": 60}
PAIRS = tuple(combinations(OLD, 2))
HORIZON = 400


def main_plans(phase):
    for mode in ("parent", "native"):
        yield mode, ALL, ALL
    for mode in ("readout", "oracle"):
        yield mode, OLD, OLD
    if phase == "grip":
        for mode in ("readout_step", "rotated", "readout_alt"):
            yield mode, OLD, OLD
        for pair in PAIRS:
            yield "pair", pair, pair
        for mode in ("native_alt", "mixed_alt"):
            yield mode, ALL, ALL


def plans(stage, phase):
    if stage == "back_dev" or (stage == "long" and phase == "reset"):
        yield "native", ("back",), ("back",)
    elif stage == "alt2" and phase == "grip":
        yield "native_alt2", ALL, ALL
        yield "readout_alt2", OLD, OLD
        yield "mixed_alt2", ALL, ALL
    elif stage == "long" and phase == "grip":
        yield "native", ALL, ALL
        yield "readout", OLD, OLD
        yield "oracle", OLD, OLD
        yield "native_alt2", ALL, ALL
        yield "mixed_alt2", ALL, ALL
    elif stage in ("parent", "main"):
        for plan in main_plans(phase):
            if (plan[0] == "parent") == (stage == "parent"):
                yield plan


def prefix_set(rt, obs, q, candidates, reference):
    late = [rt.prefix("F", obs, TEXT[g]) for g in candidates]
    old = [rt.prefix("R", obs, TEXT[g]) for g in candidates] if reference else []
    return late, old, rt.prefix("F", obs, q)


def similarities(value, late, ref):
    residual = (value.float() - late.float().mean(0))[..., :7].flatten()
    basis = ref.float() - ref.float().mean(0)
    vectors = basis[..., :7].flatten(1)
    scores = (vectors * residual).sum(1) / (vectors.norm(dim=1) * residual.norm()).clamp_min(1e-20)
    return tuple(scores.tolist())


def score(rt, prefixes, x, t):
    late, old, actual = prefixes
    now = torch.stack([rt.query("F", p, x, t) for p in late])
    value = rt.query("F", actual, x, t)
    ref = torch.stack([rt.query("R", p, x, t) for p in old]) if old else now
    return similarities(value, now, ref), ref


def choice(scores):
    ranking = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    return ranking[0], scores[ranking[0]] - scores[ranking[1]]


def routed(lab, obs, q, seed, block, candidates, mode, diagnostics):
    rt = lab.rt
    reference = not mode.startswith("mixed")
    prefixes = prefix_set(rt, obs, q, candidates, reference)
    x0 = rt.noise(seed, block)

    def pick(scores):
        idx, margin = choice(scores)
        selected = (idx + 1) % len(candidates) if mode == "rotated" else idx
        diagnostics.append({"block": block, "scores": scores, "decoded": candidates[idx],
                            "selected": candidates[selected], "margin": margin})
        return selected

    if mode == "readout_step":
        # Once the old branch is selected, mu_parent + d_parent[j] equals this exact old velocity.
        def field(x, t):
            scores, ref = score(rt, prefixes, x, t)
            return ref[pick(scores)]
    else:
        t0 = torch.ones(x0.shape[0], device=x0.device, dtype=torch.float32)
        scores, _ = score(rt, prefixes, x0, t0)
        idx = pick(scores)
        if candidates[idx] == "back":
            role, selected_prefix = "F", prefixes[2]
        else:
            role = "R"
            selected_prefix = prefixes[1][idx] if reference else rt.prefix("R", obs, TEXT[candidates[idx]])

        def field(x, t):
            return rt.query(role, selected_prefix, x, t)

    return rt.integrate(field, x0, 10)[..., :7]


def start_state(lab, rec, phase):
    obs = reset(lab.ctx, rec)
    assert lab.ctx.signature(obs) == rec["signature"]
    rows = []
    if phase == "grip":
        for k, controls in enumerate(rec["controls"]):
            obs, events = lab.execute(obs, np.asarray(controls, dtype=np.float32), k*10, rec["z0"])
            rows.extend(events)
        assert lab.ctx.signature(obs) == rec["handoff_signature"]
        assert len(rows) == rec["boundary"]
        assert not any(r["hits"] or any(r["pred"].values()) for r in rows)
    return obs, rows


def episode(lab, rec, goal, phase, mode, candidates):
    started = monotonic()
    obs, rows = start_state(lab, rec, phase)
    z0, decisions = rec["z0"], []
    boundary = len(rows)
    stop = boundary + WINDOW[phase]
    template = ALT2 if mode.endswith("_alt2") else ALT if mode.endswith("_alt") else None
    text = template.format(goal=goal) if template else TEXT[goal]
    for k in range(boundary // 10, HORIZON // 10):
        step = k*10
        if mode == "parent" or (mode == "oracle" and step < stop):
            action = lab.native("R", TEXT[goal], obs, rec["seed"], k)
        elif mode.startswith("native") or step >= stop:
            action = lab.native("F", text, obs, rec["seed"], k)
        else:
            action = routed(lab, obs, text, rec["seed"], k, candidates, mode, decisions)
        obs, events = lab.execute(obs, lab.rt.physical(action), step, z0)
        rows.extend(events)
    assert len(rows) == HORIZON
    movement = lab.ctx.geo.movement("scene2", lab.ctx.geometry, rows, z0, None)
    before = lab.ctx.geo.movement("scene2", lab.ctx.geometry, rows[:stop], z0, None)
    final = lab.ctx.geo.first_run([r["step"] for r in rows if r["obj"][2]-z0 > .03 and r["c1"] and r["c2"]])
    return {"phase": phase, "mode": mode, "goal": goal, "text": text, "pool": candidates,
            "init": rec["init"], "boundary": boundary, "window_end": stop,
            "first": movement["first_targets"], "first_step": movement["first_step"], "times": movement["times"],
            "first_in_window": before["first_targets"], "first_grasp": final,
            "ever": {g: any(r["pred"][g] for r in rows) for g in ALL}, "terminal": rows[-1]["pred"],
            "decoded_counts": dict(Counter(d["decoded"] for d in decisions)),
            "decisions": decisions, "seconds": monotonic()-started}


def log_row(out, lane, row):
    out.write(json.dumps(row, allow_nan=False)+"\n")
    out.flush()
    print("K26_EVAL", lane, row["phase"], row["mode"], row["pool"], row["init"], row["goal"],
          row["first"], row["terminal"][row["goal"]], f'{row["seconds"]:.1f}s', flush=True)


def fixed_probe(lab, rec, phase):
    """Inspect all text variants on the same obs, query states, and noise."""
    obs, events = start_state(lab, rec, phase)
    rt = lab.rt
    pref_old = [rt.prefix("R", obs, TEXT[g]) for g in OLD]
    prompt_sets = (tuple(TEXT.values()), *(tuple(t.format(goal=g) for g in ALL) for t in (ALT, ALT2)))
    pref_new = [[rt.prefix("F", obs, t) for t in texts] for texts in prompt_sets]
    samples = []

    def parent_field(x, t):
        samples.append((x.clone(), t.clone()))
        return rt.query("R", pref_old[0], x, t)

    rt.integrate(parent_field, rt.noise(rec["seed"], len(events)//10), 10)
    assert len(samples) == 10 and samples[0][1].eq(1.).all()
    rows = []
    for at in (0, 4, 8):
        x, t = samples[at]
        ref = torch.stack([rt.query("R", p, x, t) for p in pref_old])
        now, *alternate = (torch.stack([rt.query("F", p, x, t) for p in pp]) for pp in pref_new)
        for i, values in enumerate((now, *alternate)):
            scores = {}
            for method, pool, proto in (("parent3", OLD, ref), ("late3", OLD, now[:3]), ("late4", ALL, now)):
                scores[method] = {g: similarities(val, now[:len(pool)], proto)
                                  for g, val in zip(pool, values[:len(pool)], strict=True)}
            for pair in PAIRS:
                idx = [OLD.index(g) for g in pair]
                old_pair, late_pair = ref[idx], now[idx]
                key = "_".join(pair)
                scores[key] = {g: similarities(values[j], late_pair, old_pair)
                               for g, j in zip(pair, idx, strict=True)}
                axis = (old_pair[0].float() - old_pair[1].float())[..., :7]
                for g, j in zip(pair, idx, strict=True):
                    direction = float(((values[j].float() - late_pair.float().mean(0))[..., :7] * axis).sum())
                    if abs(direction) > 1e-8:
                        assert choice(scores[key][g])[0] == (0 if direction > 0 else 1)
            rows.append({"init": rec["init"], "phase": phase, "denoise_index": at, "template": i,
                         "scores": scores})
    return rows


def state_set(development):
    if development:
        files = sorted(CASES.parent.glob("screen_*.json"))
        recs = sorted((r for f in files for r in read(f) if r["init"] in DEV_IDS), key=lambda r: r["init"])
        assert tuple(r["init"] for r in recs) == DEV_IDS
    else:
        files = [CASES]
        recs = read(CASES)["selected"]
        assert [r["init"] for r in recs] == list(range(600,620))
    assert len({r["signature"]["sim"] for r in recs}) == len(recs)
    digest = sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    return recs, digest


def run(stage, step, lane, lanes):
    recs, digest = state_set(stage == "back_dev")
    recs = recs[lane::lanes]
    group = f"{stage}_{step}" if stage in ("back_dev", "long") else stage
    key = f"{group}_{lane}"
    fpath = RUN / f"{key}.jsonl"
    seen = {(r["phase"], r["mode"], r["init"], r["goal"], tuple(r["pool"])) for r in
            (json.loads(s) for s in fpath.read_text().splitlines())} if fpath.exists() else set()
    parent, mp = read_weights(REFERENCE)
    sha = {"R": mp["sha256"]}
    lab = Lab5()
    try:
        assign(lab, "R", parent)
        del parent
        if stage != "parent":
            folder = "back_long" if step > MAIN_STEPS else "back"
            late, mf = read_weights(RUN / folder / f"{step:06d}")
            assert mf["step"] == step and mf["parent_sha256"] == mp["sha256"] and mf["frozen_unchanged"]
            assign(lab, "F", late)
            del late
            sha["F"] = mf["sha256"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if stage == "probe":
                results = [row for phase in ("reset", "grip") for rec in recs for row in fixed_probe(lab, rec, phase)]
                write(RUN / f"{key}.json", results)
                print("K26_PROBE", lane, len(results), flush=True)
            else:
                rollouts(lab, fpath, stage, lane, recs, seen)
    finally:
        lab.ctx.env.close()
    write(RUN / f"{key}_COMPLETE.json", {"stage": group, "lane": lane, "lanes": lanes,
                                          "ids": [r["init"] for r in recs], "sha256": sha,
                                          "cases_sha256": digest,
                                          "window": WINDOW, "text": TEXT, "alternative": ALT,
                                          "alternative2": ALT2,
                                          "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()})


def rollouts(lab, fpath, stage, lane, recs, seen):
    with fpath.open("a") as out:
        for phase in ("reset", "grip"):
            for mode, candidates, goals in plans(stage, phase):
                for rec in recs:
                    for goal in goals:
                        identity = phase, mode, rec["init"], goal, candidates
                        if identity not in seen:
                            log_row(out, lane, episode(lab, rec, goal, phase, mode, candidates))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("parent", "main", "probe", "alt2", "back_dev", "long"), required=True)
    p.add_argument("--step", type=int, choices=(3000, 6000, 9000), default=MAIN_STEPS)
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--lanes", type=int, required=True)
    args = p.parse_args()
    assert 0 <= args.lane < args.lanes <= 20
    assert args.stage in ("back_dev", "long") or args.step == MAIN_STEPS
    assert args.stage != "long" or args.step > MAIN_STEPS
    torch.set_num_threads(2)
    run(args.stage, args.step, args.lane, args.lanes)
