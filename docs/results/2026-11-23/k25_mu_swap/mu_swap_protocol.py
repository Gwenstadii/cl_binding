"""LF -> Right: change the common response while keeping each contrast fixed."""

import v_timing_protocol as temporal

LINE = "lf_right_1000"
IDS = temporal.IDS
GOALS = temporal.GOALS[LINE]
GAIN = temporal.GAIN
BETA = 2.959
WINDOWS = ("delayed_100", "full_400")
HORIZON = 400


def jobs():
    result = [dict(family="native",background="late",beta=1.,gain=0.,window="native",arm="native")]
    for window in WINDOWS:
        result.append(dict(family="mu_only",background="parent",beta=1.,gain=0.,window=window,arm="forward"))
        for arm in ("forward","reverse"):
            for background in ("late","parent"):
                result.append(dict(family="v",background=background,beta=1.,gain=GAIN,window=window,arm=arm))
        for background in ("late","parent"):
            result.append(dict(family="own",background=background,beta=BETA,gain=0.,window=window,arm="forward"))
    return result


FIELDS = ("family","background","beta","gain","window","arm","goal","init")


def key(row):
    return tuple(row[k] for k in FIELDS)


def job_key(job):
    return tuple(job[k] for k in FIELDS[:6])


def expected_keys(ids=IDS,goals="AB"):
    return {(*job_key(job),g,i) for job in jobs() for g in goals for i in ids}


def reference_job(job):
    return job["family"] == "native" or (job["family"] == "v" and job["background"] == "late")


def unit(lane):
    return IDS[lane//2],"AB"[lane%2]


EXPECTED = len(expected_keys())
assert EXPECTED == 600

