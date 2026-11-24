"""Locked anchor audit: policy competence, eight-V replacement, and delta-only repair."""

import v_timing_protocol as temporal

LINES = temporal.LINES
IDS = temporal.IDS
GOALS = temporal.GOALS
ANCHORS = (0, 30)
GAINS = (1., temporal.GAIN)
WINDOWS = ("delayed_100", "full_400")
ARMS = ("forward", "reverse")
HORIZON = 400


def jobs(include_reused=False):
    result = []
    for anchor in ANCHORS:
        result.append(dict(family="policy", anchor=anchor, gain=0., window="native", arm="native", goals="ABN"))
    for anchor in ANCHORS:
        result.append(dict(family="v_weights", anchor=anchor, gain=1., window="full_400", arm="native", goals="AB"))
    for anchor in ANCHORS:
        for gain in GAINS:
            for window in WINDOWS:
                for arm in ARMS:
                    if include_reused or not is_reused(anchor, gain):
                        result.append(dict(family="delta", anchor=anchor, gain=gain, window=window, arm=arm, goals="AB"))
    return result


def is_reused(anchor, gain):
    return anchor == 30 and gain == temporal.GAIN


def key(row):
    return tuple(row[k] for k in ("family", "anchor", "gain", "window", "arm", "goal", "init"))


def job_key(job):
    return tuple(job[k] for k in ("family", "anchor", "gain", "window", "arm"))


def expected_keys(ids, include_reused=False):
    return {(*job_key(j), g, i) for j in jobs(include_reused) for g in j["goals"] for i in ids}


def offsets(job):
    return (0, HORIZON) if job["window"] == "native" else temporal.WINDOWS[job["window"]]


EXPECTED_NEW = len(LINES)*len(expected_keys(IDS))
EXPECTED_TOTAL = len(LINES)*len(expected_keys(IDS, True))
assert EXPECTED_NEW == 1360 and EXPECTED_TOTAL == 1680
