"""Compare pre-continuation and step-30 anchors without changing the late executor."""

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from safetensors import safe_open
import torch

RUN = Path(__file__).resolve().parent
TIMING = RUN.parent / "k25_v_timing_20261122_01"
sys.path.insert(0, str(TIMING))
import v_timing as timing
import anchor_protocol as spec

base = timing.base


def digest(path):
    return base.sha(path)


def tensors_hash(values):
    return sha256(b"".join(n.encode()+v.contiguous().numpy().tobytes() for n, v in sorted(values.items()))).hexdigest()


def checkpoint_paths(line, previous):
    p = previous["checkpoints"][line]
    return {0: Path(p["parent"])/"model.safetensors", 30: Path(p["anchor"]), 1000: Path(p["late"])}


def prepare():
    assert not (RUN / "protocol.json").exists()
    assert shutil.disk_usage(RUN).free > 256*1024**2
    prior = base.read(TIMING / "protocol.json")
    assert digest(Path(timing.__file__)) == prior["source_sha256"]
    assert digest(Path(spec.temporal.__file__)) == prior["spec_sha256"]
    assert digest(Path(base.__file__)) == prior["base_source_sha256"]
    assert digest(Path(base.sr.__file__)) == prior["baseline_source_sha256"]
    cases = base.read(TIMING / "cases.json")
    ctx = base.sr.km.Ctx("scene2")
    try:
        assert ctx.geometry == prior["geometry"]
        for case in cases:
            assert ctx.signature(base.sr.reset(ctx, case)) == case["signature"]
    finally:
        ctx.env.close()
    checked, frozen = {}, {}
    names = set(base.names())
    for line in spec.LINES:
        paths = checkpoint_paths(line, prior)
        for step, path in paths.items():
            actual = digest(path)
            expected = prior["checkpoints"][line][{0: "parent", 30: "anchor", 1000: "late"}[step]+"_sha256"]
            assert actual == expected, (line, step)
            checked[f"{line}:{step}"] = dict(path=str(path), sha256=actual)
        with safe_open(paths[0], framework="pt") as parent, safe_open(paths[30], framework="pt") as early:
            keys = set(early.keys())
            assert names <= keys
            if keys == names:
                training = base.read(base.RUN / "lf_right_train/COMPLETE.json")
                assert training["frozen_unchanged"]
                frozen[line] = dict(partial_checkpoint=True, trainable_tensors=len(names),
                    frozen_remainder="inherits parent; training completion verifies frozen_unchanged")
            else:
                assert set(parent.keys()) == keys
                others = sorted(keys-names)
                assert all(torch.equal(parent.get_tensor(n), early.get_tensor(n)) for n in others)
                frozen[line] = dict(partial_checkpoint=False, frozen_tensors_equal=len(others), trainable_tensors=len(names))
    base.write(RUN / "cases.json", cases)
    old_runtime = base.read(TIMING / "runtime.json")
    versions = {p: importlib.metadata.version(p) for p in old_runtime["packages"]}
    assert versions == old_runtime["packages"]
    # All actual imported project modules keep the previous byte-identical versions.
    for relative, expected in old_runtime["validated_dependency_sha256"].items():
        assert digest(Path("/home/lzx")/relative) == expected, relative
    base.write(RUN / "protocol.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        lines=spec.LINES, ids=spec.IDS, goals=spec.GOALS, anchors=spec.ANCHORS, gains=spec.GAINS,
        windows={w: spec.temporal.WINDOWS[w] for w in spec.WINDOWS}, arms=spec.ARMS,
        jobs=spec.jobs(), all_jobs=spec.jobs(True), expected_new=spec.EXPECTED_NEW, expected_total=spec.EXPECTED_TOTAL,
        imported_timing_episodes=320, baseline_late_episodes=120, horizon=400, checkpoints=checked,
        frozen_reconstruction=frozen, geometry=prior["geometry"], versions=versions,
        source_sha256=digest(Path(__file__)), spec_sha256=digest(Path(spec.__file__)),
        timing_source_sha256=digest(Path(timing.__file__)), timing_spec_sha256=digest(Path(spec.temporal.__file__)),
        timing_protocol_sha256=digest(TIMING/"protocol.json"), cases_sha256=digest(RUN/"cases.json"),
        base_source_sha256=digest(Path(base.__file__)), baseline_source_sha256=digest(Path(base.sr.__file__)),
        validated_dependency_sha256=old_runtime["validated_dependency_sha256"],
        rule="same late step1000 executor, text, reset, noise, normalization and endpoints; factorial anchor0/30 x gain1/5.294 x delayed/full x sign",
        anchor0="qualified two-old-task parent immediately before new-task continuation; not the unfine-tuned SmolVLA base",
        eligibility="full donor policy competence is measured separately from V-only compatibility and mu-preserving delta repair",
        data_note="20 reused physical starts; anchor audit, not independent holdout or independent training seeds"))
    print("ANCHOR_PREPARED", spec.EXPECTED_NEW, "new;", spec.EXPECTED_TOTAL, "total grid", frozen, flush=True)


def load_previous(line, lane):
    path = TIMING / f"{line}_{lane}.jsonl.gz"
    marker = base.read(TIMING / f"{line}_{lane}_COMPLETE.json")
    assert digest(path) == marker["rows_sha256"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {spec.temporal.condition_key(r): r for r in map(json.loads, stream)}, marker


def verify_first_blocks(lab, model, case, previous):
    """Reuse qualifications only after exact same-input action and readout probes."""
    obs = base.sr.reset(lab.ctx, case)
    checked = 0
    for goal in "AB":
        native = lab.native("F", base.sr.INSTR[goal], obs, case["seed"], 0)
        actual = np.clip(lab.rt.physical(native)[:10], -1, 1).astype(np.float32)
        assert timing.digest_bytes(actual.tobytes()) == previous["native", "native", goal, case["init"]]["action_hashes"][0]
        for arm in spec.ARMS:
            controls, decision = model.patch(base.sr.INSTR[goal], obs, case["seed"], 0,
                                            "v_forward" if arm == "forward" else "v_reverse")
            old = previous["full_400", arm, goal, case["init"]]
            actual = np.clip(lab.rt.physical(controls)[:10], -1, 1).astype(np.float32)
            assert timing.digest_bytes(actual.tobytes()) == old["action_hashes"][0]
            assert decision == old["decisions"][0]
            checked += 1
    return checked


def worker(line, lane, lanes):
    p = base.read(RUN / "protocol.json")
    assert p["source_sha256"] == digest(Path(__file__))
    assert p["spec_sha256"] == digest(Path(spec.__file__))
    assert p["timing_source_sha256"] == digest(Path(timing.__file__))
    assert p["cases_sha256"] == digest(RUN/"cases.json")
    base.configure(line)
    paths = {step: Path(p["checkpoints"][f"{line}:{step}"]["path"]) for step in (0,30,1000)}
    weights = {step: base.selected_weights(path) for step, path in paths.items()}
    _, selected = base.sr.parameters_for("cross_v", weights[1000])
    assert len(selected) == 8
    deltas = {a: {n: weights[a][n].double()-weights[1000][n].double() for n in sorted(selected)} for a in spec.ANCHORS}
    for a in spec.ANCHORS:
        assert all(torch.equal((weights[1000][n].double()+v).to(weights[1000][n].dtype), weights[a][n]) for n,v in deltas[a].items())
    dh = {a: tensors_hash(d) for a,d in deltas.items()}
    previous, old_marker = load_previous(line, lane)
    assert dh[30] == old_marker["delta_sha256"]
    cases = base.read(RUN/"cases.json")[lane::lanes]
    assert [c["init"] for c in cases] == old_marker["ids"]
    parent_reference = {}
    if line == "lf_right_1000":
        for text in (base.RUN/f"parent_{lane}.jsonl").read_text().splitlines():
            r = json.loads(text)
            parent_reference[r["init"], r["goal"]] = r
    parent = paths[0].parent
    lab = base.Lab(parent)
    assert lab.rt.sha == {"F": p["checkpoints"][f"{line}:0"]["sha256"]}
    dest = RUN/f"{line}_{lane}.jsonl.gz"
    assert not dest.exists()
    gates, probes, counts, parent_replays = [], 0, 0, 0
    try:
        base.sr.assign(lab, "F", weights[1000])
        models = {(a,g): base.sr.Restore(lab, weights[1000], deltas[a], g, 0.) for a in spec.ANCHORS for g in spec.GAINS}
        for (a,g), model in models.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                gates.append(dict(anchor=a, gain=g, **base.gate(lab, model, weights[1000], deltas[a], cases[0])))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for case in cases:
                probes += verify_first_blocks(lab, models[30, spec.temporal.GAIN], case, previous)
        with gzip.open(dest, "xt", encoding="utf-8", compresslevel=5) as out:
            for job in spec.jobs():
                a, gain, family = job["anchor"], job["gain"], job["family"]
                # Assignment is outside a fresh autocast context to avoid stale cast-weight caches.
                base.sr.assign(lab, "F", weights[a] if family == "policy" else weights[1000])
                if family == "v_weights":
                    params = dict(lab.rt.models["F"].named_parameters())
                    with torch.no_grad():
                        for n in selected:
                            params[n].copy_(weights[a][n])
                actual_weights = dict(lab.rt.models["F"].named_parameters())
                for n in weights[1000]:
                    reference = weights[a][n] if family == "policy" or (family == "v_weights" and n in selected) else weights[1000][n]
                    assert torch.equal(actual_weights[n].detach().cpu(), reference), (family,a,n)
                model = models[a,gain] if family == "delta" else None
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for case in cases:
                        for goal in job["goals"]:
                            row = timing.episode(lab, model, case, base.sr.INSTR[goal],
                                job["window"] if family == "delta" else "native", job["arm"] if family == "delta" else "native")
                            row.update(**{k:v for k,v in job.items() if k != "goals"}, offsets=list(spec.offsets(job)),
                                line=line, goal=goal, target=base.sr.LABEL[goal], seed=case["seed"],
                                delta_sha256=dh[a] if family != "policy" else None,
                                origin="new_episode")
                            if family == "policy" and a == 0 and (case["init"],goal) in parent_reference:
                                old = parent_reference[case["init"],goal]
                                assert all(row[k] == old[k] for k in ("text","first","first_step","times","terminal","first_grasp"))
                                assert row["signatures"][8] == timing.digest_bytes(json.dumps(old["boundary_signature"],sort_keys=True,separators=(",",":")).encode())
                                parent_replays += 1
                            out.write(json.dumps(row, allow_nan=False, separators=(",",":"))+"\n")
                            out.flush()
                            counts += 1
                            print("ANCHOR", line, lane, family, a, gain, job["window"], job["arm"], case["init"], goal,
                                  row["first"], row["terminal"][row["target"]], round(row["seconds"],2), flush=True)
    finally:
        lab.ctx.env.close()
    assert counts == len(spec.expected_keys([c["init"] for c in cases]))
    base.write(RUN/f"{line}_{lane}_COMPLETE.json", dict(line=line,lane=lane,lanes=lanes,
        ids=[c["init"] for c in cases], episodes=counts, rows_sha256=digest(dest),
        source_sha256=p["source_sha256"], spec_sha256=p["spec_sha256"], protocol_sha256=digest(RUN/"protocol.json"),
        weights_fingerprints={str(a):tensors_hash(v) for a,v in weights.items()},
        delta_sha256={str(a):v for a,v in dh.items()}, gates=gates, imported_condition_first_block_probes=probes,
        previous_parent_replays=parent_replays))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare","worker"))
    parser.add_argument("--line", choices=spec.LINES)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        prepare()
    else:
        assert args.line and 0 <= args.lane < args.lanes == 8
        worker(args.line,args.lane,args.lanes)
