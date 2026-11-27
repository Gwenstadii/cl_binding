"""Audit coverage and summarize paired K24 update forks."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
from math import dist

ROLES = {"A": "left", "B": "right", "N": "front"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fails(measurement, floor):
    return any(b + d < f - .0001 for b, d, f in zip(
        measurement["before"][-4:], measurement["delta"][-4:], floor, strict=True))


def stats(rows, choices):
    held = [(r, a["measurements"][str(i)], str(i)) for r,a in zip(rows, choices, strict=True)
            for i in r["held"]]
    controlled = [(r,a["measurements"][str(r["controlled"])],str(r["controlled"]))
                  for r,a in zip(rows, choices, strict=True)]
    metrics = rows[0]["metrics"]
    predicted = held[0][1]["prediction"]
    mae = {key: mean(abs(m["prediction"][key] - m["delta"][metrics.index(key)]) for _,m,_ in held)
           for key in predicted}
    sign = {}
    for key in predicted:
        samples = [(m["prediction"][key], m["delta"][metrics.index(key)]) for _,m,_ in held]
        eligible = [(a,b) for a,b in samples if abs(b) > 1e-5]
        sign[key] = [sum(a*b>0 for a,b in eligible), len(eligible)]
    return {"n": len(choices), "norm_median": median(a["norm"] for a in choices),
            "front_delta_median": median(a["front_delta"] for a in choices),
            "controlled_breach": sum(fails(m,r["thresholds"][i]) for r,m,i in controlled),
            "held_breach": sum(fails(m,r["thresholds"][i]) for r,m,i in held),
            "held_median_delta": dict(zip(metrics, [median(m["delta"][j] for _, m,_ in held)
                                                   for j in range(len(metrics))], strict=True)),
            "held_prediction_mae": mae, "held_sign_agreement": sign}


def rejected(row):
    measurements = {k: {**v, "delta": [0.] * len(v["delta"]),
                        "prediction": dict.fromkeys(v["prediction"], 0.)}
                    for k,v in row["arms"]["full"]["measurements"].items()}
    return {"norm": 0., "front_delta": 0., "measurements": measurements}


def closed_loop(directory):
    rows = []
    for lane in range(4):
        assert read(directory/f"native_{lane}_COMPLETE.json") == {"lane":lane, "lanes":4}
        rows.extend(json.loads(s) for s in (directory/f"native_{lane}.jsonl").read_text(encoding="utf-8").splitlines())
    table = {(r["arm"],r["init"],r["goal"]):r for r in rows}
    arms = ("anchor","full","scale","project","focused")
    ids = sorted({r["init"] for r in rows})
    assert len(ids) == 20 and len(rows) == len(table) == 300
    assert table.keys() == {(a,i,q) for a in arms for i in ids for q in ROLES}
    result = {}
    for a in arms:
        out = {}
        for q,target in ROLES.items():
            curr = [table[a,i,q] for i in ids]
            full = [table["full",i,q] for i in ids]
            moves = [dist(x["displacement_xy_m"],y["displacement_xy_m"])*1000 for x,y in zip(curr,full,strict=True)]
            out[q] = {"first": dict(Counter("/".join(r["first"]) or "none" for r in curr)),
                      "terminal":sum(r["terminal"][target] for r in curr),
                      "paired_first_changed_vs_full":sum(x["first"] != y["first"] for x,y in zip(curr,full,strict=True)),
                      "first_block_displacement_difference_vs_full_median_mm":median(moves)}
        out["both_old_first"] = sum(all(table[a,i,q]["first"] == [ROLES[q]] for q in "AB") for i in ids)
        result[a] = out
    for a in arms:
        result[a]["right_transitions_vs_full"] = dict(Counter(
            ("/".join(table["full",i,"B"]["first"]) or "none") + " -> " +
            ("/".join(table[a,i,"B"]["first"]) or "none") for i in ids))
    return result


def summarize(directory):
    result = {}
    for anchor in (30,100):
        rows = [json.loads(s) for s in (directory/f"attribution_{anchor}.jsonl").read_text(encoding="utf-8").splitlines()]
        done = read(directory/f"attribution_{anchor}_COMPLETE.json")
        assert [r["offset"] for r in rows] == done["offsets"] and len(rows) == 8
        assert all(r["anchor"] == anchor and len(r["held"]) == 2 and r["controlled"] not in r["held"] for r in rows)
        assert all(r["metrics"] == rows[0]["metrics"] and r["arms"].keys() == rows[0]["arms"].keys() for r in rows)
        arms = {k:stats(rows,[r["arms"][k] for r in rows]) for k in rows[0]["arms"]}
        for name in ("project","focused","scale"):
            keys = [r["picks"][name] for r in rows]
            choices = [r["arms"][k] if k is not None else rejected(r) for r,k in zip(rows,keys,strict=True)]
            arms[name + "_verified"] = {"picks":dict(Counter(k or "rejected" for k in keys)), **stats(rows,choices)}
            if name != "scale":
                same = [r["arms"]["scale_" + k.rsplit("_",1)[1]] if k else rejected(r)
                        for r,k in zip(rows,keys,strict=True)]
                ratios = [a["norm"]/b["norm"] for a,b in zip(choices,same,strict=True) if b["norm"]]
                arms["scale_at_" + name + "_fractions"] = {
                    **stats(rows,same), "projected_to_scaled_realized_norm_ratio_median":median(ratios)}
        def interaction(index):
            values = []
            for r in rows:
                for i in r["held"]:
                    a,b,c,d = (r["arms"][k]["measurements"][str(i)]["delta"][index]
                               for k in ("full","without_kv","without_mlp47","without_both"))
                    values.append(a-b-c+d)
            return median(values)
        result[str(anchor)] = {"arms":arms, "module_tensor_counts":{k:len(v) for k,v in done["modules"].items()},
                               "kv_mlp47_interaction": {"gain_h":interaction(0), "right_margin_m":interaction(14)}}
    if (directory/"export_100.jsonl").exists():
        export = [json.loads(s) for s in (directory/"export_100.jsonl").read_text(encoding="utf-8").splitlines()]
        first = json.loads((directory/"attribution_100.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert len(export) == 1 and export[0]["offset"] == first["offset"] == 0
        assert export[0]["front_ids"] == first["front_ids"] and export[0]["picks"] == first["picks"]
        for arm in ("full", *first["picks"].values()):
            if arm is not None:
                assert export[0]["arms"][arm] == first["arms"][arm], arm
        result["export_100"] = first["picks"]
    if all((directory/f"native_{lane}_COMPLETE.json").is_file() for lane in range(4)):
        result["closed_loop"] = closed_loop(directory)
    (directory/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    summarize(p.parse_args().directory)
