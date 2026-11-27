"""Check whether canonical action axes read held-out instruction phrasings."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic

import torch

RUN = Path(__file__).resolve().parent
K21 = RUN.parent / "k21_transfer_20261113_01"
K22 = RUN.parent / "k22_deepen_20261113_01"
K1 = RUN.parent / "k1_screen_20261101_01"
sys.path[:0] = [str(K22 / "code"), str(K21 / "code"), str(K1 / "code")]
from k22 import Lab2, P1_WINDOW
from k21 import HORIZON

LINES = ("r0_s2", "r1_front")
ARMS = ("native", "readout", "reverse", "oracle")
TEMPLATES = (
    "Please put the book in the caddy's {target} compartment.",
    "Pick up the book and set it down in the {target} section of the caddy.",
    "Move the book into the {target} compartment of the caddy.",
)


def canonical_prefixes(lab, obs):
    return {(role, q): lab.rt.prefix(role, obs, lab.text[q]) for role in "RF" for q in "AB"}


def canonical_fields(rt, prefixes, x, t):
    (a, b), (p, q) = ([rt.query(role, prefixes[role, g], x, t).float() for g in "AB"] for role in "RF")
    return (a+b)*.5, (a-b)*.5, (p+q)*.5, (p,q)


def signed_axis_score(value, mean, axis):
    ref = axis[..., :7]
    energy = ref.square().sum()
    return ((value[..., :7].float()-mean[..., :7])*ref).sum()/energy.clamp_min(1e-20)


def paraphrase(lab, q, i):
    return TEMPLATES[i].format(target=lab.goal[q])


def check_cases(lab, cases):
    rt = lab.rt
    rows = []
    for rec in cases:
        obs = lab.reset_rec(rec)
        assert lab.ctx.signature(obs) == rec["signature"]
        canon = canonical_prefixes(lab, obs)
        unseen = {(q, i): rt.prefix("F", obs, paraphrase(lab, q, i)) for q in "AB" for i in range(len(TEMPLATES))}
        for route in "AB":
            drafts = []

            def velocity(x, t):
                drafts.append((x.clone(), t.clone()))
                return rt.query("R", canon["R", route], x, t)

            rt.integrate(velocity, rt.noise(rec["seed"], 0), 10)
            assert len(drafts) == 10
            for step in (0,4,8):
                x, t = drafts[step]
                _, axis, mean, reference = canonical_fields(rt, canon, x, t)
                scores = [{q: float(signed_axis_score(reference[j], mean, axis)) for j,q in enumerate("AB")}]
                scores.extend({q: float(signed_axis_score(rt.query("F", unseen[q,i], x, t), mean, axis))
                               for q in "AB"} for i in range(len(TEMPLATES)))
                rows.append({"init": rec["init"], "route": route, "draft_step": step, "scores": scores})
    return rows


def execute(lab, rec, q, arm):
    started = monotonic()
    rt = lab.rt
    obs = lab.reset_rec(rec)
    assert lab.ctx.signature(obs) == rec["signature"]
    prompt = paraphrase(lab, q, 0)
    rows, coefficient = [], []
    for block in range(HORIZON//10):
        seed = rec["seed"]
        if arm == "native" or block*10 >= P1_WINDOW:
            action = lab.native("F", prompt, obs, seed, block)
        elif arm == "oracle":
            action = lab.native("R", lab.text[q], obs, seed, block)
        else:
            canon = canonical_prefixes(lab, obs)
            prefix = rt.prefix("F", obs, prompt)

            def velocity(x, t):
                center, axis, late_mean, _ = canonical_fields(rt, canon, x, t)
                score = signed_axis_score(rt.query("F", prefix, x, t), late_mean, axis)
                coefficient.append(float(score))
                sign = score.sign() * (-1 if arm == "reverse" else 1)
                return (center + sign*axis).to(rt.vdtype)

            action = rt.integrate(velocity, rt.noise(seed, block), 10)[..., :7]
        obs, part = lab.execute(obs, rt.physical(action), block*10, rec["z0"], rec["plate"])
        rows.extend(part)
    assert len(rows) == HORIZON and len(coefficient) == (P1_WINDOW if arm in ("readout", "reverse") else 0)
    return lab.finish(rows, rec, q, rec["z0"], rec["plate"], 0, {}, started,
                      {"arm": arm, "decoded_scores": coefficient})


def run(line, lane, lanes):
    file = K21 / "states" / "scene2.json"
    candidates = json.loads(file.read_text(encoding="utf-8"))
    cases = [r for r in candidates if r["geometry_ok"] and not any(r["pred_at_reset"].values())][:20]
    assert len(cases) == 20 and len({c["signature"]["sim"] for c in cases}) == 20
    cases = cases[lane::lanes]
    lab = Lab2(line)
    try:
        assert all(paraphrase(lab, q, i).casefold() != lab.text[q].casefold() for q in "AB" for i in range(len(TEMPLATES)))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            scores = check_cases(lab, cases)
            (RUN / f"phrasing_probe_{line}_{lane}.json").write_text(json.dumps(scores, allow_nan=False)+"\n", encoding="utf-8")
            with (RUN / f"phrasing_eval_{line}_{lane}.jsonl").open("x", encoding="utf-8") as out:
                for arm in ARMS:
                    for row in cases:
                        for q in "AB":
                            result = execute(lab, row, q, arm)
                            out.write(json.dumps(result, allow_nan=False)+"\n")
                            out.flush()
                            print("PHRASING", line, lane, arm, q, row["init"], result["first"], flush=True)
    finally:
        lab.ctx.env.close()
    marker = {"line": line, "lane": lane, "lanes": lanes, "ids": [r["init"] for r in cases],
              "sha256": lab.rt.sha, "state_sha256": sha256(file.read_bytes()).hexdigest(),
              "canonical": {q: lab.text[q] for q in "AB"}, "templates": TEMPLATES, "window": P1_WINDOW}
    (RUN / f"phrasing_{line}_{lane}_COMPLETE.json").write_text(json.dumps(marker)+"\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--line", choices=LINES, required=True)
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--lanes", type=int, default=4)
    args = p.parse_args()
    run(args.line, args.lane, args.lanes)
