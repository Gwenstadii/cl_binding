"""Fixed target geometry and first-goal endpoints for the K1 screen.

scene2 / scene1b (book -> caddy): the collaborator's rule, unchanged: the book centre is lifted
> 3 cm above its step-1 height and its horizontal projection lies inside the compartment's
contain-region box, for 5 consecutive steps (stage2r_handoff_protocol.movement).
goal suite (bowl): same lift and 5-step rule; regions = stove cook region (xy), cabinet top side
(xy and bowl centre at or above the top surface - 5 mm), plate (horizontal distance to the plate's
step-1 position < 7 cm, the plate's outer radius), top drawer (3-D box of the drawer region at reset).
"""
import math

GOALS = ("left", "front", "right", "back")
LABEL = {"left": "A", "front": "B", "right": "new", "back": "unseen"}
OBJECT = {"scene2": "black_book_1_pos", "scene1b": "black_book_1_pos", "goal": "akita_black_bowl_1_pos"}
PLATE_RADIUS = 0.07
LIFT = 0.03
RUN = 5
SITES = {"scene2": {g: f"desk_caddy_1_{g}_contain_region" for g in GOALS},
         "scene1b": {g: f"desk_caddy_1_{g}_contain_region" for g in GOALS},
         "goal": {"left": "flat_stove_1_cook_region", "front": "wooden_cabinet_1_top_side",
                  "back": "wooden_cabinet_1_top_region"}}


def first_run(steps, minimum=RUN):
    count, previous = 0, None
    for step in steps:
        count = count + 1 if previous is not None and step == previous + 1 else 1
        previous = step
        if count >= minimum:
            return step - minimum + 1
    return None


def extents(site, axes):
    rot, half = site["rotation"], site["half_size"]
    for row in rot:
        if sum(abs(x) > 1e-8 for x in row) != 1:
            raise ValueError("Tilted site requires another projection rule")
    return [sum(abs(rot[i][j]) * half[j] for j in range(3)) for i in axes]


def inside(p, site, axes=(0, 1)):
    e = extents(site, axes)
    return all(abs(p[i] - site["centre"][i]) <= e[k] for k, i in enumerate(axes))


def build(family, env, np):
    """Region boxes read from the reset scene (sites are static bodies)."""
    m, d = env.sim.model, env.sim.data
    out = {}
    for g, name in SITES[family].items():
        i = m.site_name2id(name)
        out[g] = {"site": name, "centre": [float(x) for x in d.site_xpos[i]],
                  "rotation": np.asarray(d.site_xmat[i]).reshape(3, 3).tolist(),
                  "half_size": [float(x) for x in m.site_size[i]]}
    return out


def check_against(geometry, reference, tol=1e-5):
    for g in GOALS:
        a, b = geometry[g], reference[g]
        err = max(max(abs(x - y) for x, y in zip(a["centre"], b["centre"])),
                  max(abs(x - y) for x, y in zip(a["half_size"], b["half_size"])),
                  max(abs(x - y) for ra, rb in zip(a["rotation"], b["rotation"]) for x, y in zip(ra, rb)))
        if err > tol:
            raise AssertionError(f"Geometry differs from the collaborator's fixed geometry: {g} {err}")
    return True


def hits(family, geometry, p, z0, plate_xy):
    """Goals whose region contains object position p (only when lifted)."""
    if p[2] - z0 <= LIFT:
        return []
    out = []
    if family in ("scene2", "scene1b"):
        for g in GOALS:
            if inside(p, geometry[g]):
                out.append(g)
        return out
    if inside(p, geometry["left"]):
        out.append("left")
    top = geometry["front"]
    if inside(p, top) and p[2] >= top["centre"][2] - 0.005:
        out.append("front")
    if plate_xy is not None and math.dist(p[:2], plate_xy) < PLATE_RADIUS:
        out.append("right")
    if inside(p, geometry["back"], axes=(0, 1, 2)):
        out.append("back")
    return out


def movement(family, geometry, rows, z0, plate_xy, start=1):
    """First 5-step region entry per goal over rows with step >= start."""
    eligible = {g: [] for g in GOALS}
    for row in rows:
        if row["step"] < start:
            continue
        for g in hits(family, geometry, row["obj"], z0, plate_xy):
            eligible[g].append(row["step"])
    times = {g: first_run(v) for g, v in eligible.items()}
    earliest = min((t for t in times.values() if t is not None), default=None)
    first = [g for g, t in times.items() if earliest is not None and t == earliest]
    return {"times": times, "first_targets": first, "first_step": earliest}


def lifted_run(rows, z0, start):
    steps = [r["step"] for r in rows if r["step"] >= start and r["obj"][2] - z0 > LIFT]
    return first_run(steps) is not None
