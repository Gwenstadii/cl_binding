"""Pure timing/endpoint definitions shared by simulation, audit and tests."""

from collections import Counter
import math

SITES = ("left", "front", "right", "back")
LINES = ("lr_front_1000", "lf_right_1000")
IDS = tuple(range(800, 820))
GAIN = 5.294038631688923
HORIZON = 400
LIFT = .03
RUN = 5
WINDOWS = {
    "early_100": (0, 100),
    "mid_100": (40, 140),
    "delayed_100": (80, 180),
    "late_100": (160, 260),
    "full_400": (0, 400),
    "delayed_tail": (80, 400),
    "late_tail": (160, 400),
}
FIXED_LENGTH = ("early_100", "mid_100", "delayed_100", "late_100")
TAILS = ("full_400", "delayed_tail", "late_tail")
ARMS = ("forward", "reverse")
GOALS = {"lr_front_1000": dict(A="left", B="right", N="front"),
         "lf_right_1000": dict(A="left", B="front", N="right")}


def jobs():
    return [("native", "native")]+[(window, arm) for window in WINDOWS for arm in ARMS]


def active_blocks(window):
    if window == "native":
        return []
    low, high = WINDOWS[window]
    return list(range(low//10, high//10))


def first_run(steps):
    previous, count = None, 0
    for step in steps:
        count = count+1 if previous is not None and step == previous+1 else 1
        previous = step
        if count == RUN:
            return step-RUN+1
    return None


def region_mask(point, z0, geometry):
    if point[2]-z0 <= LIFT:
        return 0
    mask = 0
    for j, goal in enumerate(SITES):
        site = geometry[goal]
        ext = [sum(abs(site["rotation"][i][k])*site["half_size"][k] for k in range(3)) for i in (0, 1)]
        if all(abs(point[i]-site["centre"][i]) <= ext[i] for i in (0, 1)):
            mask |= 1 << j
    return mask


def movement(masks, low=0, high=None):
    """low/high are control offsets; all five confirming steps must be inside."""
    high = len(masks) if high is None else high
    times = {g: first_run(i+1 for i in range(low, high) if masks[i] & (1 << j)) for j, g in enumerate(SITES)}
    earliest = min((t for t in times.values() if t is not None), default=None)
    return dict(first=[g for g, t in times.items() if t is not None and t == earliest],
                first_step=earliest, times=times)


def score(trace, geometry, z0):
    count = len(trace["object"])
    assert count == HORIZON
    assert all(len(trace[k]) == count for k in ("eef", "contacts", "success", "actions", "regions"))
    assert all(len(p) == 3 and all(math.isfinite(x) for x in p) for key in ("object", "eef") for p in trace[key])
    assert all(len(a) == 7 and all(math.isfinite(x) and -1 <= x <= 1 for x in a) for a in trace["actions"])
    assert all(type(m) is int and 0 <= m < 16 for key in ("success", "regions") for m in trace[key])
    assert all(type(m) is int and 0 <= m < 4 for m in trace["contacts"])
    masks = [region_mask(p, z0, geometry) for p in trace["object"]]
    assert masks == trace["regions"]
    grasp = first_run(i+1 for i in range(count) if trace["contacts"][i] == 3 and trace["object"][i][2]-z0 > LIFT)
    return dict(**movement(masks), first_grasp=grasp,
                grasp_confirmed_step=grasp+RUN-1 if grasp is not None else None,
                terminal={g: bool(trace["success"][-1] & (1 << j)) for j, g in enumerate(SITES)},
                ever={g: any(m & (1 << j) for m in trace["success"]) for j, g in enumerate(SITES)})


def interval_metrics(trace, geometry, z0, low, high):
    masks = trace["regions"]
    grasp = first_run(i+1 for i in range(low) if trace["contacts"][i] == 3 and trace["object"][i][2]-z0 > LIFT)
    first = movement(masks, 0, low)
    before_point = trace["start"]["object"] if low == 0 else trace["object"][low-1]
    after_point = trace["object"][min(low+10, HORIZON)-1]
    approaches = {}
    for goal in SITES:
        center = geometry[goal]["centre"][:2]
        approaches[goal] = 1000*(math.dist(before_point[:2], center)-math.dist(after_point[:2], center))
    return dict(pre=first, pre_grasp=grasp,
                pre_grasp_confirmed_step=grasp+RUN-1 if grasp is not None else None,
                post=movement(masks, low), by_window_end=movement(masks, 0, high),
                first_10_approach_mm=approaches)


def condition_key(row):
    return row["window"], row["arm"], row["goal"], row["init"]


def primary(row):
    return "+".join(row["first"]) or "none"


def required_common_blocks(first, second):
    """Number of identical control blocks required before schedule first differs."""
    a, b = set(active_blocks(first)), set(active_blocks(second))
    return next((k for k in range(40) if (k in a) != (k in b)), 40)
