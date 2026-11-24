"""Only intervention timing changes; checkpoints, V deltas, gain and starts are fixed."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch

import v_timing_protocol as spec

RUN = Path(__file__).resolve().parent
PREVIOUS = RUN.parent / "k25_order_restore_20261122_01"
sys.path.insert(0, str(PREVIOUS))
import order_restore as base


def digest_bytes(data):
    return sha256(data).hexdigest()


def signature(lab, obs):
    return digest_bytes(json.dumps(lab.ctx.signature(obs), sort_keys=True, separators=(",", ":")).encode())


def prior_records(line):
    paths = sorted(PREVIOUS.glob(f"main_{line}_*.jsonl"))
    if line == "lf_right_1000":
        paths += sorted(PREVIOUS.glob("standard_*.jsonl"))
    result = {}
    for path in paths:
        for text in path.read_text().splitlines():
            r = json.loads(text)
            if r["variant"] != "canon" or r["goal"] not in "AB":
                continue
            if r["arm"] == "native":
                result["native", "native", r["goal"], r["init"]] = r
            elif r["gain"] == spec.GAIN:
                result["delayed_100", r["arm"], r["goal"], r["init"]] = r
    return result


def prepare():
    assert not (RUN / "protocol.json").exists()
    original = base.read(PREVIOUS / "protocol.json")
    assert base.sha(Path(base.__file__)) == original["source_sha256"]
    cases = base.read(PREVIOUS / "cases.json")["test"]
    assert [r["init"] for r in cases] == list(spec.IDS)
    ctx = base.sr.km.Ctx("scene2")
    try:
        geometry = ctx.geometry
        for case in cases:
            obs = base.sr.reset(ctx, case)
            assert ctx.signature(obs) == case["signature"]
    finally:
        ctx.env.close()
    base.write(RUN / "cases.json", cases)
    checked = {}
    for line in spec.LINES:
        for key in ("parent", "late", "anchor"):
            path = Path(original["checkpoints"][line][key])
            path = path / "model.safetensors" if key == "parent" else path
            if str(path) not in checked:
                checked[str(path)] = base.sha(path)
            assert checked[str(path)] == original["checkpoints"][line][key+"_sha256"]
    versions = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "lerobot", "mujoco", "numpy", "safetensors")}
    base.write(RUN / "protocol.json", dict(lines=spec.LINES, goals=spec.GOALS, ids=spec.IDS,
        gain=spec.GAIN, windows=spec.WINDOWS, fixed_length=spec.FIXED_LENGTH, tails=spec.TAILS,
        arms=spec.ARMS, jobs=spec.jobs(), horizon=spec.HORIZON, variant="canon", checkpoints={line: original["checkpoints"][line] for line in spec.LINES},
        source_sha256=base.sha(Path(__file__)), spec_sha256=base.sha(Path(spec.__file__)),
        base_source_sha256=base.sha(Path(base.__file__)), baseline_source_sha256=base.sha(Path(base.sr.__file__)),
        cases_sha256=base.sha(RUN / "cases.json"), previous_protocol_sha256=base.sha(PREVIOUS / "protocol.json"),
        geometry=geometry, versions=versions, expected_episodes=1200,
        rules="fixed checkpoints, eight V differences, alpha, true prompts, noise, horizon and scoring; vary temporal mask only",
        timing_selection="seven fixed schedules; all outcomes reported; no timing chosen using these outcomes"))
    print("TIMING_PREPARED", len(cases), len(spec.jobs()), flush=True)


def episode(lab, model, case, text, window, arm):
    started = monotonic()
    obs = base.sr.reset(lab.ctx, case)
    assert lab.ctx.signature(obs) == case["signature"]
    trace = dict(start=dict(object=obs[lab.ctx.obj].tolist(), eef=obs["robot0_eef_pos"].tolist()),
                 object=[], eef=[], contacts=[], success=[], regions=[], actions=[])
    signatures = [signature(lab, obs)]
    action_hashes, decisions, events = [], [], []
    active = set(spec.active_blocks(window))
    for block in range(40):
        if block in active:
            controls, decision = model.patch(text, obs, case["seed"], block,
                                             "v_forward" if arm == "forward" else "v_reverse")
            decisions.append(decision)
        else:
            controls = lab.native("F", text, obs, case["seed"], block)
        physical = lab.rt.physical(controls)
        actual = np.clip(physical[:10], -1, 1).astype(np.float32)
        action_hashes.append(digest_bytes(actual.tobytes()))
        obs, rows = lab.execute(obs, physical, block*10, case["z0"])
        events.extend(rows)
        for row, action in zip(rows, actual, strict=True):
            trace["object"].append(row["obj"])
            trace["eef"].append(row["eef"])
            trace["contacts"].append(int(row["c1"])+2*int(row["c2"]))
            trace["success"].append(sum(1 << j for j, g in enumerate(spec.SITES) if row["pred"][g]))
            trace["regions"].append(sum(1 << j for j, g in enumerate(spec.SITES) if g in row["hits"]))
            trace["actions"].append(action.tolist())
        signatures.append(signature(lab, obs))
    assert len(events) == 400
    result = spec.score(trace, lab.ctx.geometry, case["z0"])
    expected = lab.ctx.geo.movement("scene2", lab.ctx.geometry, events, case["z0"], None)
    assert result["first"] == expected["first_targets"] and result["times"] == expected["times"]
    low, high = spec.WINDOWS[window] if window != "native" else (0, 400)
    interval = spec.interval_metrics(trace, lab.ctx.geometry, case["z0"], low, high)
    return dict(init=case["init"], text=text, window=window, offsets=[low, high], arm=arm,
        gain=0. if arm == "native" else spec.GAIN, **result, **interval,
        decisions=decisions, routed=dict(Counter(d["decoded"] for d in decisions)),
        signatures=signatures, action_hashes=action_hashes, trace=trace, seconds=monotonic()-started)


def worker(line, lane, lanes):
    protocol = base.read(RUN / "protocol.json")
    assert protocol["source_sha256"] == base.sha(Path(__file__))
    assert protocol["spec_sha256"] == base.sha(Path(spec.__file__))
    assert protocol["base_source_sha256"] == base.sha(Path(base.__file__))
    assert protocol["cases_sha256"] == base.sha(RUN / "cases.json")
    base.configure(line)
    parent, full, deltas = base.weights(line)
    # Hash the eight tensors that actually participate, not a label for the pathway.
    delta_hash = digest_bytes(b"".join(n.encode()+v.contiguous().numpy().tobytes() for n, v in sorted(deltas.items())))
    cases = base.read(RUN / "cases.json")[lane::lanes]
    reference = prior_records(line)
    lab = base.Lab(parent)
    path = RUN / f"{line}_{lane}.jsonl.gz"
    assert not path.exists()
    replay = []
    try:
        base.sr.assign(lab, "F", full)
        model = base.sr.Restore(lab, full, deltas, spec.GAIN, 0.)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            gate = base.gate(lab, model, full, deltas, cases[0])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            with gzip.open(path, "xt", encoding="utf-8", compresslevel=5) as out:
                for window, arm in spec.jobs():
                    for case in cases:
                        for goal in "AB":
                            row = episode(lab, model, case, base.sr.INSTR[goal], window, arm)
                            row.update(line=line, goal=goal, target=base.sr.LABEL[goal], seed=case["seed"], delta_sha256=delta_hash)
                            key = window, arm, goal, case["init"]
                            if key in reference:
                                old = reference[key]
                                assert all(row[k] == old[k] for k in ("text", "first", "times", "first_step", "terminal", "first_grasp")), (line, key, "prior endpoint mismatch")
                                old_sig = digest_bytes(json.dumps(old["boundary_signature"], sort_keys=True, separators=(",", ":")).encode())
                                assert row["signatures"][8] == old_sig, (line, key, "prior step-80 mismatch")
                                replay.append(list(key))
                            out.write(json.dumps(row, allow_nan=False, separators=(",", ":"))+"\n")
                            out.flush()
                            print("TIMING", line, lane, window, arm, case["init"], goal, row["first"],
                                  row["terminal"][row["target"]], round(row["seconds"], 2), flush=True)
    finally:
        lab.ctx.env.close()
    base.write(RUN / f"{line}_{lane}_COMPLETE.json", dict(line=line, lane=lane, lanes=lanes,
        ids=[c["init"] for c in cases], gates=gate, protocol_sha256=base.sha(RUN / "protocol.json"),
        source_sha256=base.sha(Path(__file__)), spec_sha256=base.sha(Path(spec.__file__)),
        delta_sha256=delta_hash, rows_sha256=base.sha(path), previous_replays=replay))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "worker"))
    parser.add_argument("--line", choices=spec.LINES)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        prepare()
    else:
        assert args.line and 0 <= args.lane < args.lanes
        worker(args.line, args.lane, args.lanes)
