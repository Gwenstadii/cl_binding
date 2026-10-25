import json, sys
import numpy as np
r = json.load(open(sys.argv[1]))
print("centres", {g: np.round(v, 3).tolist() for g, v in r["compartment_centres"].items()}, r["centre_support"])
print("model      | L: ok/near-correct/toward-RIGHT/other | F: same | right ever | back ever")
for m, v in r["models"].items():
    if m.endswith("_p1"):
        continue
    cells = []
    for g in ("left", "front"):
        ok = miss = right = other = 0
        for e in v["episodes"].get(g, []):
            if e["correct_ever"]:
                ok += 1; continue
            if e["first_goal"] == "right":
                right += 1; continue
            if e["first_goal"] is not None or not e["lifted"]:
                other += 1; continue
            cl = e["closest_xy_m"]; k = min(cl, key=cl.get)
            if k == g: miss += 1
            elif k == "right": right += 1
            else: other += 1
        cells.append(f"{ok:2d}/{miss:2d}/{right:2d}/{other:2d}")
    src = r["models"][m.replace("_p0", "_p1")] if m.endswith("_p0") else v
    rs, bs = src["summary"].get("right"), src["summary"].get("back")
    print(f"{m:10s} | {cells[0]} | {cells[1]} | {rs['episodes'] - rs['never_requested_success']} | {bs['episodes'] - bs['never_requested_success']}")
