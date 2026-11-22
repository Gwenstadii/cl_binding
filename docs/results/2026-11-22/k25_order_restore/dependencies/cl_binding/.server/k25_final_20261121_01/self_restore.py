"""Restore the old action contrast using one continued policy and eight V deltas."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from safetensors.torch import load_file, save_file
import torch

RUN = Path(__file__).resolve().parent
V3 = RUN.parent / "k25v3_20261120_01"
V_DIRECTION = RUN.parent / "k25v3_v_20261120_01"
K24 = RUN.parent / "k24_final_window_20261117_01"
LATE = K24 / "w_full_130.safetensors"
ANCHOR = K24 / "w_anchor_130.safetensors"
HOLDOUT = V_DIRECTION / "holdout.json"
DELTA = RUN / "value_delta.safetensors"
MANIFEST = RUN / "method.json"
sys.path.insert(0, str(V3))
from v_attribution_v3 import INSTR, Lab5, Relay, assign, parameters_for, read, write
from window_eval import reset
from k14 import EXPECTED, HORIZON, PAIR, km

LABEL = {"A": "left", "B": "right", "N": "front"}
SIGN = {"A": 1.0, "B": -1.0, "N": 0.0}
OLD = "AB"
GOALS = "ABN"
VARIANTS = {"move": "Move the book to the caddy compartment on the {goal}.",
            "please": "Please set the book inside the {goal} compartment of the caddy."}
WINDOWS = {"reset": (0, 150), "grip": (0, 60), "late": (80, 180)}
CORE = ("native", "v_forward", "v_reverse")
SETUP = {"reset": {"canon": {**dict.fromkeys(CORE, GOALS), "own_d": OLD}},
         "grip": {"canon": {**dict.fromkeys(CORE, GOALS), **dict.fromkeys(("v_natural", "own_d", "extrap_d"), OLD)},
                  "move": dict.fromkeys(CORE, GOALS)},
         "late": {"canon": dict.fromkeys(("v_forward", "v_reverse"), OLD)}}


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def prepare():
    full, anchor = load_file(str(LATE)), load_file(str(ANCHOR))
    late_sha = read(K24 / "full_COMPLETE.json")["sha256"]
    assert digest(LATE) == late_sha and full.keys() == anchor.keys() and len(full) == 155
    family, names = parameters_for("cross_v", full)
    assert family == "cross" and len(names) == 8
    deltas = {n: anchor[n].double() - full[n].double() for n in sorted(names)}
    assert all(full[n].dtype == anchor[n].dtype and torch.count_nonzero(d) for n, d in deltas.items())
    assert all(torch.equal((full[n].double() + d.double()).to(anchor[n].dtype), anchor[n]) for n, d in deltas.items())
    assert not MANIFEST.exists() and not DELTA.exists()
    save_file(deltas, str(DELTA))
    gain = read(V3 / "v_calibration.json")["scale"]
    residual = read(V_DIRECTION / "plan.json")["residual_scale"]
    assert gain == read(V_DIRECTION / "plan.json")["scale"]
    write(MANIFEST, {"late_sha256": late_sha, "anchor_sha256": digest(ANCHOR), "delta_sha256": digest(DELTA),
                     "delta_bytes": DELTA.stat().st_size, "tensors": {n: list(d.shape) for n,d in deltas.items()},
                     "role": "F", "gain": gain, "own_d_gain": residual, "plan": SETUP,
                     "control_offsets": WINDOWS, "candidate_prompts": INSTR, "variants": VARIANTS,
                     "readout": "nearest centered late F response, 50 action slots by 7 executed dimensions",
                     "case_sha256": digest(HOLDOUT), "source_sha256": digest(Path(__file__))})
    print("K25_FINAL_PREPARED", len(deltas), DELTA.stat().st_size, flush=True)


class SingleLab(Lab5):
    def __init__(self):
        self.ctx = km.Ctx("scene2")
        self.torch, self.np = self.ctx.torch, self.ctx.np
        self.rt = km.make_runtime(self.ctx, PAIR, ["F"])
        assert self.rt.sha == {"F": EXPECTED["F"]}


def prompt(variant, goal):
    return INSTR[goal] if variant == "canon" else VARIANTS[variant].format(goal=LABEL[goal])


def contrast(a, b):
    return (a.float() - b.float()) * .5


def rank(actual, left, right, front):
    mean = (left.float() + right.float()) * .5
    actual_residual = actual.float() - mean
    late_axis = contrast(left, right)
    vectors = [late_axis, -late_axis, front.float() - mean]
    dists = torch.stack([(actual_residual[..., :7] - v[..., :7]).square().mean() for v in vectors]).tolist()
    i, j, *_ = sorted(range(3), key=dists.__getitem__)
    axis = late_axis[..., :7]
    projection = (actual_residual[..., :7] * axis).sum() / axis.square().sum().clamp_min(1e-20)
    return {"decoded": GOALS[i], "distance": dict(zip(GOALS, dists, strict=True)),
            "margin": dists[j]-dists[i], "old_axis_score": float(projection)}


class Restore:
    def __init__(self, lab, full, deltas, gain, residual):
        self.rt = lab.rt
        self.full, self.deltas, self.gain, self.residual = full, deltas, gain, residual
        self.relays = {s: Relay(lab, full, self.overlay(s)) for s in (1., gain)}

    def overlay(self, scale):
        return {n: (self.full[n].double()+scale*d.double()).to(self.full[n].dtype) for n,d in self.deltas.items()}

    def prefixes(self, obs, texts):
        # Deduplicate identical queries without deriving any route from the text string.
        return {q: self.rt.prefix("F", obs, q) for q in dict.fromkeys(texts)}

    def value(self, p, x, t):
        return self.rt.query("F", p, x, t)

    def local_contrast(self, relay, projected, pair, x, t):
        values = []
        for g in OLD:
            with relay.whole(projected, g, False):
                values.append(self.value(pair[g], x, t))
        return contrast(*values)

    def patch(self, text, obs, seed, block, mode):
        # This API has the actual text and current observation. It has no goal label.
        prefixes = self.prefixes(obs, (text, *(INSTR[g] for g in GOALS)))
        pair = {g: prefixes[INSTR[g]] for g in OLD}
        x0 = self.rt.noise(seed, block)
        t0 = torch.ones(x0.shape[0], device=x0.device)
        initial = {q: self.value(p, x0, t0) for q, p in prefixes.items()}
        route = rank(initial[text], *(initial[INSTR[g]] for g in GOALS))
        amplitude = {"v_forward": self.gain, "v_reverse": -self.gain, "v_natural": 1.,
                     "extrap_d": 1., "own_d": self.residual, "zero": 0.}[mode]
        coefficient = amplitude * SIGN[route["decoded"]]
        relay = self.relays[self.gain if mode == "extrap_d" else 1.]
        projected = relay.cache(pair) if coefficient and mode != "own_d" else None
        steps = 0
        delta_norms = []

        def velocity(x, t):
            nonlocal steps
            if steps == 0:
                evaluate = initial.__getitem__
            else:
                cache = {}

                def evaluate(q):
                    if q not in cache:
                        cache[q] = self.value(prefixes[q], x, t)
                    return cache[q]
            steps += 1
            base = evaluate(text)
            if not coefficient:
                return base
            d_f = contrast(*(evaluate(INSTR[g]) for g in OLD))
            delta = d_f if mode == "own_d" else self.local_contrast(relay, projected, pair, x, t) - d_f
            delta_norms.append(float(delta[..., :7].square().mean().sqrt()))
            # This is mu_F + ((v_F(actual) - mu_F) + coefficient * delta_d).
            return (base.float() + coefficient * delta).to(base.dtype)

        controls = self.rt.integrate(velocity, x0, 10)[..., :7]
        assert steps == 10 and (not coefficient or len(delta_norms) == 10)
        route.update(block=block, coefficient=coefficient,
                     contrast_rms=sum(delta_norms)/len(delta_norms) if delta_norms else 0.)
        return controls, route


def start_state(lab, rec, phase):
    obs = reset(lab.ctx, rec)
    assert lab.ctx.signature(obs) == rec["signature"]
    rows = []
    if phase == "grip":
        for k, controls in enumerate(rec["controls"]):
            obs, events = lab.execute(obs, np.asarray(controls, dtype=np.float32), 10*k, rec["z0"])
            rows.extend(events)
        assert lab.ctx.signature(obs) == rec["handoff_signature"] and len(rows) == rec["boundary"]
        assert not any(r["hits"] or any(r["pred"].values()) for r in rows)
    return obs, rows


def rms(x):
    return float(x.float().square().mean().sqrt())


def check_gate(lab, model, rec):
    obs, rows = start_state(lab, rec, "grip")
    rt, seed, block = lab.rt, rec["seed"], len(rows)//10
    x, t = rt.noise(seed, block), torch.ones(1, device="cuda")
    prefixes = model.prefixes(obs, (INSTR[g] for g in GOALS))
    pair = {g: prefixes[INSTR[g]] for g in OLD}
    base = {g: model.value(prefixes[INSTR[g]], x, t) for g in GOALS}
    d_f = contrast(base["A"], base["B"])
    model_policy = rt.models["F"]
    v_params = {n: p for n,p in model_policy.named_parameters() if n in model.deltas}
    assert v_params.keys() == model.deltas.keys() and tuple(rt.models) == ("F",)
    max_leak, equal_probe = 0., []
    for strength, relay in model.relays.items():
        projected = relay.cache(pair)
        corrected = {}
        for g in OLD:
            with relay.whole(projected, g, False, verify=True):
                corrected[g] = model.value(pair[g], x, t)
        alt_weights = model.overlay(strength)
        try:
            for name,p in v_params.items():
                p.copy_(alt_weights[name])
            expected = {g: model.value(pair[g], x, t) for g in OLD}
            assert all(torch.equal(expected[g], corrected[g]) for g in OLD)
        finally:
            for name,p in v_params.items():
                p.copy_(model.full[name])
        assert all(torch.equal(base[g], model.value(pair[g], x, t)) for g in OLD)
        equal_probe.append(strength)
        delta = contrast(corrected["A"], corrected["B"]) - d_f
        scale = model.gain if strength == 1. else 1.
        for polarity in (1., -1.):
            pair_corrections = [polarity*SIGN[g]*scale*delta.double() for g in OLD]
            exec_fields = [base[g].double()+c for g,c in zip(OLD, pair_corrections, strict=True)]
            torch.testing.assert_close(exec_fields[0]+exec_fields[1], base["A"].double()+base["B"].double(),
                                       atol=1e-12, rtol=1e-12)
            mean = (base["A"].float()+base["B"].float())*.5
            cast = sum(v.to(base["A"].dtype).float() for v in exec_fields)*.5
            max_leak = max(max_leak, rms(cast-mean))
        assert max_leak < .01
    for g in GOALS:
        reference = lab.native("F", INSTR[g], obs, seed, block)
        tested, decision = model.patch(INSTR[g], obs, seed, block, "zero")
        assert decision["decoded"] == g and torch.equal(reference, tested)
        if g == "N":
            for mode in ("v_forward", "v_reverse"):
                tested, decision = model.patch(INSTR[g], obs, seed, block, mode)
                assert decision["coefficient"] == 0. and torch.equal(reference, tested)
    return {"init": rec["init"], "single_model_roles": list(rt.models), "overlay_equals_v_only_weight_change": equal_probe,
            "canonical_native_bitwise_if_zero": True, "canonical_front_bypass_bitwise": True,
            "canonical_pair_mean_fp64_preserved_before_cast": True, "pair_mean_cast_max_rms": max_leak}


def episode(lab, model, rec, text, mode, phase):
    started = monotonic()
    obs, rows = start_state(lab, rec, phase)
    boundary = len(rows)
    low, high = (boundary+dt for dt in WINDOWS[phase])
    decode = []
    for k in range(boundary//10, HORIZON//10):
        if mode == "native" or k*10 < low or k*10 >= high:
            action = lab.native("F", text, obs, rec["seed"], k)
        else:
            action, decision = model.patch(text, obs, rec["seed"], k, mode)
            decode.append(decision)
        obs, events = lab.execute(obs, lab.rt.physical(action), k*10, rec["z0"])
        rows.extend(events)
    assert len(rows) == HORIZON
    c = lab.ctx
    end = c.geo.movement("scene2", c.geometry, rows, rec["z0"], None)
    z0 = rec["z0"]
    grasp = c.geo.first_run([r["step"] for r in rows if r["obj"][2]-z0>.03 and r["c1"] and r["c2"]])
    end_window = c.geo.movement("scene2", c.geometry, rows[:high], z0, None)
    book_first = np.array(rows[low-1]["obj"] if low else rec["book_start"])[:2]
    book_last = np.array(rows[low+9]["obj"])[:2]
    first_mm = {}
    for g in LABEL.values():
        center = np.array(c.geometry[g]["centre"])[:2]
        first_mm[g] = 1000*float(np.linalg.norm(book_first-center)-np.linalg.norm(book_last-center))
    return {"init": rec["init"], "text": text, "phase": phase, "arm": mode,
            "boundary": boundary, "window": [low,high], "first": end["first_targets"], "times": end["times"],
            "first_step": end["first_step"], "first_in_window": end_window["first_targets"], "first_grasp": grasp,
            "terminal": rows[-1]["pred"], "ever": {g: any(r["pred"][g] for r in rows) for g in c.geo.GOALS},
            "first_10_approach_mm": first_mm, "decisions": decode, "routed": dict(Counter(d["decoded"] for d in decode)),
            "seconds": monotonic()-started}


def fixed_probe(lab, model, rec, phase):
    obs, events = start_state(lab, rec, phase)
    inputs = {v: {g: prompt(v,g) for g in GOALS} for v in ("canon", *VARIANTS)}
    prefixes = model.prefixes(obs, (q for texts in inputs.values() for q in texts.values()))
    out = []
    for offset in (0, 1, 2):
        x = lab.rt.noise(rec["seed"], len(events)//10+offset)
        t = torch.ones(1, device=x.device)
        changed = {q: model.value(p, x, t) for q,p in prefixes.items()}
        native = [changed[INSTR[g]] for g in GOALS]
        out.append({"phase": phase, "init": rec["init"], "noise_offset": offset,
                    "scores": {v: {g: rank(changed[q], *native) for g,q in texts.items()} for v,texts in inputs.items()}})
    return out


def run(suite, lane, lanes):
    manifest = read(MANIFEST)
    assert digest(LATE) == manifest["late_sha256"] and digest(DELTA) == manifest["delta_sha256"]
    assert digest(HOLDOUT) == manifest["case_sha256"] and manifest["plan"] == SETUP
    full, deltas = load_file(str(LATE)), load_file(str(DELTA))
    assert {n: list(d.shape) for n,d in deltas.items()} == manifest["tensors"]
    recs = read(HOLDOUT)["selected"]
    assert [r["init"] for r in recs] == list(range(600, 620))
    assert len({r["signature"]["sim"] for r in recs}) == 20
    recs = recs[lane::lanes]
    lab = SingleLab()
    try:
        assign(lab, "F", full)
        model = Restore(lab, full, deltas, manifest["gain"], manifest["own_d_gain"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            gate = check_gate(lab, model, recs[0])
        print("K25_FINAL_GATE", lane, gate, flush=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if suite == "probe":
                out = [r for p in ("reset", "grip") for rec in recs for r in fixed_probe(lab, model, rec, p)]
                write(RUN / f"probe_{lane}.json", out)
            else:
                compute_episodes(lab, model, recs, suite, lane)
    finally:
        lab.ctx.env.close()
    write(RUN / f"{suite}_{lane}_COMPLETE.json", {"suite": suite, "lane": lane, "lanes": lanes,
          "ids": [r["init"] for r in recs], "method_sha256": digest(MANIFEST), "gates": gate,
          "source_sha256": digest(Path(__file__))})


def compute_episodes(lab, model, recs, suite, lane):
    path = RUN / f"{suite}_{lane}.jsonl"
    seen = {(r["phase"],r["variant"],r["arm"],r["init"],r["goal"]) for r in
            map(json.loads,path.read_text().splitlines())} if path.exists() else set()
    phases = ("reset", "grip") if suite == "main" else ("late",)
    with path.open("a", encoding="utf-8") as out:
        for phase in phases:
            for variant, arms in SETUP[phase].items():
                for arm, goals in arms.items():
                    for rec in recs:
                        for g in goals:
                            if (phase,variant,arm,rec["init"],g) in seen:
                                continue
                            row = episode(lab, model, rec, prompt(variant,g), arm, phase)
                            row.update(goal=g, variant=variant)
                            out.write(json.dumps(row,allow_nan=False)+"\n")
                            out.flush()
                            print("K25_FINAL_EVAL", lane, phase, variant, arm, rec["init"], g, row["first"],
                                  row["terminal"][LABEL[g]], f'{row["seconds"]:.1f}s', flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=("prepare", "probe", "main", "late"))
    p.add_argument("--lane", type=int, default=0)
    p.add_argument("--lanes", type=int, default=1)
    args = p.parse_args()
    assert 0 <= args.lane < args.lanes <= 20
    torch.set_num_threads(2)
    if args.suite == "prepare":
        prepare()
    else:
        run(args.suite,args.lane,args.lanes)
