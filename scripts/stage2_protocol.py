"""Pure, testable rules for stage 2A: matched FT4, ordinary ER and the full-replay ceiling.

Every random choice has its own named stream, so drawing old samples can never shift the new-sample
schedule, noise or flow time that FT4, ER and full replay must share.
"""

import hashlib
import random

OLD_GOALS = ("left", "front")
BRANCHES = {
    # new right samples per step, old samples per old goal per step, where old samples come from
    "ft4": {"new": 4, "old_per_goal": 0, "old_source": None},
    "er": {"new": 4, "old_per_goal": 1, "old_source": "memory"},
    "full_replay": {"new": 4, "old_per_goal": 1, "old_source": "full_old_train_split"},
    # Counterfactual instruction distillation: the old slots hold parent actions for NEW-task frames under the old
    # instructions (labelled offline once); no old data is stored or read.
    "cid": {"new": 4, "old_per_goal": 1, "old_source": "pseudo"},
    # Each old slot is a parent pseudo-label or a memory record with probability 1/2 (own stream).
    "cid_er": {"new": 4, "old_per_goal": 1, "old_source": "pseudo_or_memory"},
}
LOSS_DENOMINATOR = 6
# SmolVLA flow-time recipe: Beta(1.5, 1.0) * 0.999 + 0.001. Beta(a, 1) has CDF x**a, so u**(1/a) is exact.
TIME_ALPHA, TIME_SCALE, TIME_OFFSET = 1.5, 0.999, 0.001


def stream_rng(stream, seed, step=None):
    """Independent deterministic stream; string seeding hashes the whole key (Python random version 2)."""
    key = f"stage2a:{stream}:{seed}" if step is None else f"stage2a:{stream}:{seed}:{step}"
    return random.Random(key)


def stream_seed(stream, seed, step):
    """64-bit integer seed for torch generators derived from the same stream key."""
    digest = hashlib.sha256(f"stage2a:{stream}:{seed}:{step}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def select_memory(groups, per_goal, seed):
    """Uniform sampling without replacement inside each old goal's training frames, nothing else."""
    if set(groups) != set(OLD_GOALS) or per_goal < 1:
        raise ValueError("Memory covers exactly the left/front old goals with a positive budget")
    selected = {}
    for goal in OLD_GOALS:
        frames = sorted(groups[goal])
        if len(set(frames)) != len(frames) or len(frames) < per_goal:
            raise ValueError(f"Goal {goal} has too few unique training frames")
        selected[goal] = sorted(stream_rng(f"memory-{goal}", seed).sample(frames, per_goal))
    if set(selected["left"]) & set(selected["front"]):
        raise ValueError("A frame cannot belong to two goals")
    return selected


def nested_subset(positions_by_goal, per_goal, seed):
    """Prefix of one fixed per-goal permutation, so smaller budgets are subsets of larger ones (4 in 16 in 64 ...)."""
    if set(positions_by_goal) != set(OLD_GOALS) or per_goal < 1:
        raise ValueError("Subset covers exactly the left/front old goals with a positive budget")
    chosen = {}
    for goal in OLD_GOALS:
        positions = sorted(positions_by_goal[goal])
        if len(positions) < per_goal:
            raise ValueError(f"Parent memory has too few {goal} records")
        order = stream_rng(f"subset-{goal}", seed).sample(positions, len(positions))
        chosen[goal] = sorted(order[:per_goal])
    return chosen


def step_indices(population, count, seed, step, stream):
    """Uniform draws with replacement for one update; identical for every branch that uses this stream."""
    population = list(population)
    if not population or count < 0:
        raise ValueError("Nonempty population and nonnegative count required")
    rng = stream_rng(stream, seed, step)
    return [rng.choice(population) for _ in range(count)]


def step_plan(branch, step, seeds, new_frames, old_pools):
    """Indices consumed by one update. New indices depend only on the new-sample seed and step."""
    spec = BRANCHES[branch]
    plan = {"new": step_indices(new_frames, spec["new"], seeds["new_samples"], step, "new")}
    for goal in OLD_GOALS:
        plan[goal] = (step_indices(old_pools[goal], spec["old_per_goal"], seeds["old_samples"], step, f"old-{goal}")
                      if spec["old_per_goal"] else [])
    return plan


def loss_weights(branch):
    """Per-record weights: every record is divided by the same fixed denominator in all branches."""
    spec = BRANCHES[branch]
    count = spec["new"] + len(OLD_GOALS) * spec["old_per_goal"]
    if count > LOSS_DENOMINATOR:
        raise ValueError("Batch exceeds the fixed loss denominator")
    return [1 / LOSS_DENOMINATOR] * count


def flow_time_from_uniform(u):
    return u ** (1 / TIME_ALPHA) * TIME_SCALE + TIME_OFFSET


def exposures(branch, steps):
    spec = BRANCHES[branch]
    return {"new": spec["new"] * steps, **{goal: spec["old_per_goal"] * steps for goal in OLD_GOALS}}


def memory_record_bytes(image_shape, cameras, chunk, action_dim, state_dim):
    """Raw payload of one record before container overhead (uint8 images, float32 actions/state, bool mask)."""
    pixels = 1
    for dimension in image_shape:
        pixels *= dimension
    return cameras * pixels + chunk * action_dim * 4 + chunk + state_dim * 4


def check_budget(file_sizes, limit):
    total = sum(file_sizes.values())
    return {"files": dict(file_sizes), "total_bytes": total, "limit_bytes": limit, "within_budget": total <= limit}
