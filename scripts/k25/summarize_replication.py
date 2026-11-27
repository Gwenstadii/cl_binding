"""Audit the extra training-stream and held-phrasing experiments in the synthesis."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from summarize import endpoint, label, lines, read
from summarize_v3 import aggregate, compare, rf_margin

SEEDS = (20261121, 20261122)
IDS = list(range(600, 620))
RIGHT = ("late", "anchor", "v_natural", "v_amplified", "v_reverse", "mlp_window", "mlp_late_query", "mlp_query_only")
LINES = {"r0_s2": ("left", "front", "right"), "r1_front": ("left", "right", "front")}
PHRASING = ("native", "readout", "reverse", "oracle")


def fixed_metrics(rows):
    ids = {(r["init"], r["route"], r["step"]) for r in rows}
    assert len(ids) == len(rows) == 120
    statistic = aggregate(rows, IDS)
    return {k: statistic(lambda r: r[k]) for k in rows[0] if k not in ("init", "route", "step", "by_layer")}


def training(directory):
    streams = {}
    for seed in SEEDS:
        data = read(directory / f"probe_{seed}.json")
        assert read(directory / f"probe_{seed}_COMPLETE.json") == {"seed":seed, "states":20, "drafts":120}
        assert data["seed"] == seed and sorted(data["sha256"]) == ["100", "30"]
        vals = fixed_metrics(data["value"])
        vals["v_fraction_of_anchor_correction"] = vals["v_gain"]/vals["anchor_correction"]
        rows = {}
        for lane in range(4):
            marker = read(directory / f"eval_{seed}_{lane}_COMPLETE.json")
            assert (marker["seed"], marker["lane"], marker["lanes"]) == (seed,lane,4)
            assert data["sha256"] == marker["sha256"] and marker["gate"]["init"] == IDS[lane]
            assert all(marker["gate"][k] for k in ("same_query_is_native_bitwise",
                                                     "action_output_cannot_change_fixed_query",
                                                     "zero_v_correction_is_native_bitwise"))
            expected = {(a,q,i) for i in IDS[lane::4] for a in RIGHT for q in ("ABN" if a in ("late","anchor") else "B")}
            found = lines(directory / f"eval_{seed}_{lane}.jsonl")
            assert len(found) == len(expected)
            for row in found:
                key = row["arm"],row["goal"],row["init"]
                assert row["seed"] == seed and key in expected and key not in rows
                expected.remove(key)
                rows[key] = row
        right_rows = {(a,i): r for (a,q,i),r in rows.items() if q=="B"}
        outcomes = {}
        for arm in RIGHT:
            outcomes[arm] = {}
            for q in ("ABN" if arm in ("late","anchor") else "B"):
                trials = [rows[arm,q,i] for i in IDS]
                outcome = (endpoint(trials,q) if q != "N" else
                           {"n":len(trials), "first":dict(Counter(map(label,trials))),
                            "terminal":sum(r["terminal"]["front"] for r in trials)})
                if q == "B":
                    outcome["initial_rf_median_mm"] = median(rf_margin(r) for r in trials)
                outcomes[arm][q] = outcome
        contrasts = [(a,"late") for a in RIGHT if a != "late"]
        contrasts.extend((a,b) for a,b in (("v_amplified","v_reverse"),("mlp_late_query","mlp_window"),
                                           ("mlp_query_only","mlp_window")))
        paired = {f"{a}_vs_{b}": compare(right_rows, IDS,b,a) for a,b in contrasts}
        reaches = {a: {i for i in IDS if right_rows["late",i]["first"]==["front"] and right_rows[a,i]["first"]==["right"]}
                   for a in ("mlp_window", "mlp_late_query")}
        streams[str(seed)] = {"value":vals,"query":{k:fixed_metrics(v) for k,v in data["query"].items()},
                              "sha256":data["sha256"], "outcomes":outcomes, "matched":paired,
                              "retained_mlp_front_recoveries_after_late_query":len(set.intersection(*reaches.values())),
                              "n":len(rows)}
    assert all(len({s["sha256"][step] for s in streams.values()})==2 for step in ("30","100"))
    return streams


def scores_summary(values):
    return {q: {"fraction_correct": sum(v*(1 if q=="A" else -1)>0 for v in series)/len(series),
                "median_signed_score": median(series), "queries":len(series)} for q,series in values.items()}


def held_phrasing(directory):
    source = directory / "scene2.json"
    ids = [r["init"] for r in read(source) if r["geometry_ok"] and not any(r["pred_at_reset"].values())][:20]
    assert len(ids) == len(set(ids)) == 20
    sha = sha256(source.read_bytes()).hexdigest()
    result = {}
    for line, (a,b,n) in LINES.items():
        rows, probes, checkpoint_hashes = {}, [], set()
        for lane in range(4):
            marker = read(directory / f"phrasing_{line}_{lane}_COMPLETE.json")
            assert (marker["line"], marker["lane"], marker["lanes"])==(line,lane,4)
            assert marker["ids"]==ids[lane::4] and marker["state_sha256"]==sha and marker["window"]==150
            checkpoint_hashes.add(tuple(sorted(marker["sha256"].items())))
            found = read(directory / f"phrasing_probe_{line}_{lane}.json")
            assert {(r["init"],r["route"],r["draft_step"]) for r in found}=={(i,q,x) for i in ids[lane::4] for q in "AB" for x in (0,4,8)}
            probes.extend(found)
            expected = {(r,q,i) for i in ids[lane::4] for r in PHRASING for q in "AB"}
            for row in lines(directory / f"phrasing_eval_{line}_{lane}.jsonl"):
                key = row["arm"],row["key"],row["init"]
                assert row["line"]==line and key in expected and key not in rows
                assert row["requested"]==dict(zip("AB",(a,b),strict=True))[row["key"]]
                assert len(row["decoded_scores"])==(marker["window"] if row["arm"] in ("readout","reverse") else 0)
                expected.remove(key)
                rows[key] = row
            assert not expected
        assert len(checkpoint_hashes)==1 and len(probes)==120
        fixed = []
        for j in range(1+len(marker["templates"])):
            values = {q:[r["scores"][j][q] for r in probes] for q in "AB"}
            pairs = list(zip(values["A"], values["B"], strict=True))
            fixed.append({**scores_summary(values),
                          "pair_order_fraction": sum(a_score>b_score for a_score,b_score in pairs)/len(pairs),
                          "median_pair_center_shift": median((a_score+b_score)*.5 for a_score,b_score in pairs),
                          "median_pair_half_difference": median((a_score-b_score)*.5 for a_score,b_score in pairs)})
        outcomes = {}
        for arm in PHRASING:
            o = {}
            for q,target in zip("AB",(a,b),strict=True):
                cases = [rows[arm,q,i] for i in ids]
                o[q] = {"first":dict(Counter(map(label,cases))), "terminal":sum(r["terminal"][target] for r in cases)}
            o["paired_first_correct"] = sum(rows[arm,"A",i]["first"]==[a] and rows[arm,"B",i]["first"]==[b] for i in ids)
            o["first_new_target"] = sum(rows[arm,q,i]["first"]==[n] for q in "AB" for i in ids)
            if arm in ("readout","reverse"):
                o["actual_phrase_signs"] = scores_summary({q:[v for i in ids for v in rows[arm,q,i]["decoded_scores"]] for q in "AB"})
                o["actual_sign_by_50_step_window"] = [
                    scores_summary({q: [v for i in ids for v in rows[arm,q,i]["decoded_scores"][start:start+50]]
                                    for q in "AB"}) for start in (0,50,100)]
            outcomes[arm] = o
        same = sum(all(rows["readout",q,i][k]==rows["oracle",q,i][k] for k in ("first","first_step","terminal"))
                   for q in "AB" for i in ids)
        result[line] = {"n":len(rows), "sha256":dict(next(iter(checkpoint_hashes))), "templates":marker["templates"],
                        "canonical":marker["canonical"], "fixed":fixed,"outcomes":outcomes,
                        "same_first_step_and_terminal_as_oracle":same}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("part",choices=("all","training","phrasing"),default="all",nargs="?")
    args = parser.parse_args()
    tasks = {"training":training, "phrasing":held_phrasing}
    for name in tasks if args.part=="all" else (args.part,):
        result = tasks[name](args.directory)
        (args.directory / f"{name}_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
