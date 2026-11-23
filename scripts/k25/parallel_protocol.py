"""Only the K22 old-mean + projected-residual method; no new ablations."""

import v_timing_protocol as scoring

IDS=tuple(range(800,820))
PROFILES=("lr_front_1000","lf_right_1000","lr_front_12000","lf_right_12000")
GOALS={p:dict(zip("ABN",("left","right","front") if p.startswith("lr_") else ("left","front","right"))) for p in PROFILES}
WINDOW=(0,150)
HORIZON=400
EXPECTED=160


def ids(lane):
    return list(IDS[lane::8])


def expected(profile,lane):
    return {(profile,g,i) for i in ids(lane) for g in "AB"}


def upper_median(values):
    values=sorted(values)
    return values[len(values)//2]
