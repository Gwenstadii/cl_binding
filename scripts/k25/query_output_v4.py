"""Separate the action mean from the condition difference induced by restored query inputs."""

import argparse
import json

import torch

from query_upstream_v4 import (INSTR, RUN, WINDOW, Lab5, Queries, assign, checkpoint_files, handoff,
                               measure_episode, read, records, rolled, same_role_gate, write)

ARMS = ("query_mu", "query_d")


def prefixes(lab, obs):
    return {(role, q): lab.rt.prefix(role, obs, INSTR[q]) for role in "FR" for q in "AB"}


def responses(sites, prompts, x, t):
    native, patched = [], []
    for q in "AB":
        _, h = sites.capture("R", prompts["R", q], x, t)
        native.append(sites.rt.query("F", prompts["F", q], x, t))
        patched.append(sites.replace(prompts["F", q], x, t, h))
    return native, patched


def chunk(lab, sites, obs, seed, chunk, arm):
    prompts = prefixes(lab, obs)
    rt = lab.rt

    def velocity(x, t):
        ordinary, patched = responses(sites, prompts, x, t)
        l, r = (b.float() - a.float() for a, b in zip(ordinary, patched, strict=True))
        desired = (l + r) * .5 if arm == "query_mu" else (r - l) * .5
        return (ordinary[1].float() + desired).to(ordinary[1].dtype)

    return rt.integrate(velocity, rt.noise(seed, chunk), 10)[..., :7]


def run(lane, lanes):
    cases = records()[20:][lane::lanes]
    full, anchor = checkpoint_files()
    lab = Lab5()
    sites = Queries(lab)
    native = lab.native
    try:
        verified = same_role_gate(lab, sites, cases[0], full, anchor, read(RUN / "inventory.json"))
        assign(lab, "R", rolled(full, anchor, read(RUN / "selection.json")["joint"]))
        obs = handoff(lab, cases[0])
        x = lab.rt.noise(cases[0]["seed"], len(cases[0]["controls"]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ordinary, patched = responses(sites, prefixes(lab, obs), x, torch.ones(1, device="cuda"))
            a, b = (t.double() - s.double() for s, t in zip(ordinary, patched, strict=True))
            mu, d = (a + b)*.5, (a - b)*.5
            assert torch.equal((ordinary[0].double()+mu+d).to(patched[0].dtype), patched[0])
            assert torch.equal((ordinary[1].double()+mu-d).to(patched[1].dtype), patched[1])
        verified["two_output_parts_reconstruct_query_transfer"] = True
        with (RUN / f"output_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm in ARMS:
                for rec in cases:
                    start = len(rec["controls"])

                    def sample(role, text, observation, seed, step):
                        if start <= step < start+WINDOW:
                            return chunk(lab, sites, observation, seed, step, arm)
                        return native(role, text, observation, seed, step)

                    lab.native = sample
                    row = {"arm": arm, **measure_episode(lab, rec, "B")}
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                    print("QUERY_OUTPUT", lane, arm, rec["init"], row["first"], flush=True)
    finally:
        lab.native = native
        lab.ctx.env.close()
    write(RUN / f"output_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes, "gate": verified})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lane", type=int, required=True)
    p.add_argument("--lanes", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(2)
    run(a.lane, a.lanes)
