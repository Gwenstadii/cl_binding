"""K23 paired physical rollouts using the existing project simulator and sampler."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

FIELDS = {
    "new_parallel": ("F", False),
    "new_full": ("F", True),
    "old_parallel": ("R", False),
    "old_full": ("R", True),
}
NATIVES = {"new_native": "F", "old_window": "R"}
CONDITIONS = {**FIELDS, **NATIVES}
EFFECTIVE = 7
CONTROL_STEPS = 10
WINDOW = 150
HORIZON = 400
LINES = {
    "r1_front": ("scene2", {"A": "left", "B": "right", "N": "front"}),
    "c_s2a": ("scene2", {"A": "left", "B": "front", "N": "right"}),
    "c_goal": ("goal", {"A": "left", "B": "front", "N": "right"}),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def field_parts(old_a, old_b, new_a, new_b, new_q, gain):
    """All responses use the same observation, draft and time."""
    eff = [np.asarray(v[..., :EFFECTIVE], dtype=np.float64)
           for v in (old_a, old_b, new_a, new_b, new_q)]
    oa, ob, na, nb, nq = eff
    means = {"R": (oa + ob) * .5, "F": (na + nb) * .5}
    axis = (oa - ob) * .5
    full = gain * (nq - means["F"])
    energy = np.vdot(axis, axis).real
    c = float(np.vdot(full, axis).real / energy) if energy > 1e-20 else 0.
    parallel = c * axis
    return means, parallel, full, c, float(np.linalg.norm(full - parallel)), bool(energy <= 1e-20)


def native(rt, role, text, obs, noise):
    prefix = rt.prefix(role, obs, text)
    return rt.integrate(lambda x, t: rt.query(role, prefix, x, t), noise, 10)[..., :EFFECTIVE]


def intervene(rt, obs, texts, q, noise, condition, gain):
    mu_role, use_full = FIELDS[condition]
    keys = [("R", "A"), ("R", "B"), ("F", "A"), ("F", "B")]
    if q == "N":
        keys.append(("F", q))
    prefixes = {key: rt.prefix(key[0], obs, texts[key[1]]) for key in keys}
    diagnostics = []

    def velocity(x, t):
        values = {key: rt.query(key[0], prefix, x, t) for key, prefix in prefixes.items()}
        baseline = values["F", q]
        args = [values[key].float().cpu().numpy() for key in keys[:4]]
        means, parallel, full, c, off, degenerate = field_parts(
            *args, baseline.float().cpu().numpy(), gain)
        result = baseline.clone()
        effective = means[mu_role] + (full if use_full else parallel)
        result[..., :EFFECTIVE] = rt.torch.as_tensor(effective, device=x.device, dtype=result.dtype)
        diagnostics.append((c, off, degenerate))
        return result

    actions = rt.integrate(velocity, noise, 10)[..., :EFFECTIVE]
    assert len(diagnostics) == 10
    return actions, diagnostics


def reset(ctx, record):
    ctx.evaluation.reset_common(ctx.env, np.asarray(record["state"], dtype=np.float64),
                                record["reset_seed"], ctx.root["evaluation"]["settle_steps"],
                                ctx.xml)
    obs = ctx.fresh(ctx.env)
    assert ctx.signature(obs) == record["signature"], record["init"]
    return obs


def episode(ctx, rt, record, texts, q, condition, gain):
    obs = reset(ctx, record)
    z0 = float(obs[ctx.obj][2])
    plate = ctx.plate_xy(obs)
    rows, diag = [], []
    for step in range(0, HORIZON, CONTROL_STEPS):
        noise = rt.noise(record["noise_seed"], step // CONTROL_STEPS)
        if step >= WINDOW or condition in NATIVES:
            role = NATIVES.get(condition, "F") if step < WINDOW else "F"
            chunk = native(rt, role, texts[q], obs, noise)
        else:
            chunk, block_diag = intervene(rt, obs, texts, q, noise, condition, gain)
            diag.extend(block_diag)
        physical = rt.physical(chunk)
        for pos in range(CONTROL_STEPS):
            obs, executed = ctx.step(physical[pos])
            if pos + 1 == CONTROL_STEPS:
                obs = ctx.fresh(ctx.env)
            rows.append(ctx.row(step + pos + 1, obs, physical[pos], executed))
    assert len(rows) == HORIZON
    move = ctx.geo.movement(ctx.family, ctx.geometry, rows, z0, plate)
    return {"init": record["init"], "goal": q, "condition": condition, "first": move["first_targets"],
            "first_step": move["first_step"], "reached": move["times"], "terminal": rows[-1]["pred"],
            "ever": {goal: any(r["pred"][goal] for r in rows) for goal in rows[-1]["pred"]},
            "position_every_block": [r["obj"] for r in rows[CONTROL_STEPS - 1::CONTROL_STEPS]],
            "query_diagnostics": diag}


def run(a):
    sys.path.insert(0, str(a.runtime_code))
    from k1_main import Ctx, make_runtime

    family, goals = LINES[a.line]
    spec = read(a.manifest)[a.line]
    assert 1.0 < spec["gain"] < 20.0 and spec["gain_source"]
    assert {"R", "F"} == spec["sha256"].keys()
    state_path = a.states / f"{family}.json"
    state_sha = file_hash(state_path)
    manifest_sha = file_hash(a.manifest)
    suite = read(state_path)
    records = suite["records"]
    assert len(records) == 20 and len({r["signature"]["sim"] for r in records}) == 20
    assert len({r["init"] for r in records}) == len({r["noise_seed"] for r in records}) == 20
    ctx = Ctx(family)
    try:
        assert ctx.xml_sha == suite["xml_sha256"]
        ctx.torch.set_num_threads(2)
        rt = make_runtime(ctx, spec["pair"], {"R", "F"})
        assert spec["sha256"] == rt.sha, (spec["sha256"], rt.sha)
        texts = {q: ctx.all_texts[g] for q, g in goals.items()}
        a.output.mkdir(parents=True, exist_ok=True)
        path = a.output / f"{a.line}_{a.lane}.jsonl"
        with ctx.torch.inference_mode(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16), path.open("x", encoding="utf-8") as f:
            # Same-model reconstruction must reproduce each old canonical action exactly.
            rec = records[a.lane]
            obs = reset(ctx, rec)
            for q in "AB":
                noise = rt.noise(rec["noise_seed"], 0)
                direct = native(rt, "F", texts[q], obs, noise)
                rebuilt, _ = intervene(rt, obs, texts, q, noise, "new_full", 1.)
                assert ctx.torch.equal(direct, rebuilt), q
            for rec in records[a.lane::a.lanes]:
                for condition in CONDITIONS:
                    for q in "ABN":
                        row = episode(ctx, rt, rec, texts, q, condition, spec["gain"])
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        f.flush()
                        print(a.line, rec["init"], condition, q, row["first"], flush=True)
        write(path.with_suffix(".complete.json"), {"line": a.line, "lane": a.lane, "lanes": a.lanes,
                                                   "gain": spec["gain"], "sha256": rt.sha,
                                                   "state_sha256": state_sha, "manifest_sha256": manifest_sha,
                                                   "init_ids": [r["init"] for r in records[a.lane::a.lanes]]})
    finally:
        ctx.env.close()


def aggregate(output, states, lanes):
    report = {}
    manifest_hashes = set()
    for line, (family, goals) in LINES.items():
        state_path = states / f"{family}.json"
        state_sha = file_hash(state_path)
        records = read(state_path)["records"]
        ids = [r["init"] for r in records]
        episodes = {}
        checkpoints = set()
        for lane in range(lanes):
            stem = output / f"{line}_{lane}"
            marker = read(stem.with_suffix(".complete.json"))
            assert marker["init_ids"] == ids[lane::lanes] and marker["lanes"] == lanes
            assert marker["line"] == line and marker["lane"] == lane and marker["state_sha256"] == state_sha
            manifest_hashes.add(marker["manifest_sha256"])
            checkpoints.add((marker["sha256"]["R"], marker["sha256"]["F"], marker["gain"]))
            for s in stem.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(s)
                key = (row["init"], row["condition"], row["goal"])
                assert key not in episodes and row["init"] in marker["init_ids"]
                episodes[key] = row
        assert episodes.keys() == {(i, c, q) for i in ids for c in CONDITIONS for q in goals}
        assert len(checkpoints) == 1
        cells, paired = {}, {}
        for condition in CONDITIONS:
            cells[condition] = {}
            paired[condition] = []
            for q, goal in goals.items():
                rows = [episodes[i, condition, q] for i in ids]
                cells[condition][q] = {"first": dict(Counter("/".join(r["first"]) or "none" for r in rows)),
                                       "terminal": sum(r["terminal"][goal] for r in rows)}
            paired[condition] = [
                int(all(episodes[i, condition, q]["first"] == [goals[q]] for q in "AB")) for i in ids]
            cells[condition]["paired_old_first"] = sum(paired[condition])
        comparisons = {}
        for ref, test in (("new_full", "new_parallel"), ("old_full", "old_parallel"),
                          ("new_parallel", "old_parallel"), ("new_full", "old_full")):
            transitions = Counter(f"{x}->{y}" for x, y in zip(paired[ref], paired[test], strict=True))
            comparisons[f"{ref} vs {test}"] = dict(transitions)
        report[line] = {"episodes": len(episodes), "outcomes": cells, "paired_transitions": comparisons,
                        "paired_difference_of_differences": sum(
                            old_p - old_f - new_p + new_f for old_p, old_f, new_p, new_f in zip(
                                paired["old_parallel"], paired["old_full"], paired["new_parallel"],
                                paired["new_full"], strict=True)) / len(ids)}
    assert len(manifest_hashes) == 1
    write(output / "summary.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("line", choices=LINES)
    worker.add_argument("lane", type=int)
    worker.add_argument("--runtime-code", type=Path, required=True)
    worker.add_argument("--manifest", type=Path, required=True)
    summarize = commands.add_parser("summarize")
    for command in (worker, summarize):
        command.add_argument("--states", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--lanes", type=int, default=4)
    a = p.parse_args()
    if a.command == "worker":
        assert 0 <= a.lane < a.lanes <= 20
        run(a)
    else:
        aggregate(a.output, a.states, a.lanes)
