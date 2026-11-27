"""Select partial retention of Front updates without tuning on K25 holdout."""

import argparse
import json
import sys

import torch

from combine_rollback import (DEV, K19, Lab5, RUN, assign, fields, load_weights,
                              make_bank, measure_episode, measures, read, weights, write)


def retain(mask, scale, full, anchor):
    kept = weights(mask, full, anchor)
    return {n: (anchor[n].float() + scale * (v.float() - anchor[n].float())).to(v.dtype)
            if not torch.equal(v, anchor[n]) else v for n, v in kept.items()}


def screen():
    selection = read(RUN / "selection.json")
    full, anchor = load_weights("full"), load_weights("anchor")
    errors = ("d_h_to_anchor_rms", "mu_h_to_anchor_rms", "mu_zr_to_anchor_rms")
    energy = sum(read(RUN / "inventory.json")["delta_squared_norm"].values())
    candidates = []
    lab = Lab5()
    try:
        bank = make_bank(lab, DEV)
        assign(lab, "F", anchor)
        ref = fields(lab, bank)
        for mask in (55, 57, 47, 0):
            for scale in (.25, .5, .75):
                tentative = retain(mask, scale, full, anchor)
                fraction = sum(float((v.float() - anchor[n].float()).square().sum())
                               for n, v in tentative.items()) / energy
                assign(lab, "F", tentative)
                row = {"mask": mask, "scale": scale, "retained_delta_squared_norm_fraction": fraction,
                       **measures(fields(lab, bank), ref)}
                row["max_normalized_error"] = max(row[k] / selection["baseline"][k] for k in errors)
                candidates.append(row)
                print("GRADED", row, flush=True)
    finally:
        lab.ctx.env.close()
    # Prefer retaining the most update under each response-restoration budget.
    selected = {}
    for arm, max_err in (("graded_tight", .20), ("graded_moderate", .30)):
        eligible = [r for r in candidates if r["max_normalized_error"] <= max_err
                    and r["d_gain_h"] >= .85 and r not in selected.values()]
        if eligible:
            selected[arm] = max(eligible, key=lambda r: r["retained_delta_squared_norm_fraction"])
        else:
            selected[arm] = min((r for r in candidates if r not in selected.values()),
                                key=lambda r: (r["max_normalized_error"],
                                               -r["retained_delta_squared_norm_fraction"]))
    write(RUN / "graded_selection.json", {"candidates": selected, "all_rows": candidates,
                                          "dev": list(DEV), "error_budgets": [.20, .30],
                                          "gain_floor": .85})


def evaluate(lane, lanes):
    reference_dir = RUN.parent / "k25_deep_d_20261118_01"
    sys.path.insert(0, str(reference_dir))
    from deep_selection import fixed_bank, responses
    from eval_k20 import validation_bank, validation_fields

    choices = read(RUN / "graded_selection.json")["candidates"]
    full, anchor = load_weights("full"), load_weights("anchor")
    cases = read(K19 / "holdout.json")["selected"]
    assert [r["init"] for r in cases] == list(range(200, 220))
    lab = Lab5()
    try:
        if lane == 0:
            demo, phases = validation_bank(lab)
            fixed = fixed_bank(lab, cases)
        with (RUN / f"graded_eval_{lane}.jsonl").open("x", encoding="utf-8") as out:
            for arm, pick in choices.items():
                assign(lab, "F", retain(pick["mask"], pick["scale"], full, anchor))
                for rec in cases[lane::lanes]:
                    for q in "ABN":
                        row = {"arm": arm, **measure_episode(lab, rec, q)}
                        out.write(json.dumps(row, allow_nan=False) + "\n")
                        out.flush()
                        print("EVAL", lane, arm, rec["init"], q, row["first"], flush=True)
                if lane == 0:
                    write(RUN / f"demo_{arm}.json", validation_fields(lab, demo, phases))
                    write(RUN / f"held_probe_{arm}.json", responses(lab, fixed, True))
    finally:
        lab.ctx.env.close()
    write(RUN / f"graded_eval_{lane}_COMPLETE.json", {"lane": lane, "lanes": lanes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("screen", "eval"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "screen":
        screen()
    else:
        evaluate(args.lane, args.lanes)
