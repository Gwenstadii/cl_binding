"""Pure, testable data-access and behavioral-label rules for stage one."""

import random
from pathlib import Path
import xml.etree.ElementTree as ET

GOALS = ("front", "left", "right", "back")


def normalization_stat_shape(shape):
    """Scalar statistics may serialize as (), (1,) or singleton image axes."""
    shape = tuple(shape)
    return (1,) if all(dimension == 1 for dimension in shape) else shape


def new_goal_development_gate(rows, goal, expected, required):
    """Use only the new goal, retaining failures and rejecting repeated initial states."""
    if goal not in GOALS or not 0 < required <= expected:
        raise ValueError('Invalid development gate')
    selected = [row for row in rows if row['requested_goal'] == goal]
    if len(selected) != expected or len({row['init_index'] for row in selected}) != expected:
        raise ValueError('Development requires exactly the expected unique initial states')
    ever = sum(row['correct_ever'] for row in selected)
    terminal = sum(row['terminal_category'] == 'correct' for row in selected)
    return {'goal':goal,'total':expected,'success_ever':ever,'terminal_correct':terminal,
            'required':required,'passed':ever >= required and terminal >= required}


def observation_action_indices(length, alignment):
    """Pair within a single trajectory; never borrow a terminal label from another."""
    if alignment not in ("same_index_legacy", "post_action_obs_to_next_action"):
        raise ValueError(f"Unknown observation/action alignment: {alignment}")
    offset = int(alignment == "post_action_obs_to_next_action")
    if length <= offset:
        raise ValueError("Trajectory has no usable observation/action pairs")
    return range(length-offset), range(offset, length)


def remap_demo_xml(xml, assets_root, robosuite_root):
    tree = ET.fromstring(xml)
    for element in tree.findall(".//asset/*"):
        value = element.get("file")
        if not value:
            continue
        if "/robosuite/" in value:
            value = str(Path(robosuite_root) / value.rsplit("/robosuite/", 1)[1])
        elif "/assets/" in value:
            value = str(Path(assets_root) / value.split("/assets/", 1)[1])
        if not Path(value).is_file():
            raise FileNotFoundError(value)
        element.set("file", value)
    return ET.tostring(tree, encoding="unicode")


def balanced_indices(groups, per_goal, seed, step):
    if not groups or per_goal < 1 or any(not values for values in groups.values()):
        raise ValueError("Each goal needs nonempty support")
    rng = random.Random((seed << 32) + step)
    indices = [rng.choice(values) for values in groups.values() for _ in range(per_goal)]
    rng.shuffle(indices)
    return indices


def split_trajectories(names, train_count, seed):
    if len(set(names)) != len(names) or not 0 < train_count < len(names):
        raise ValueError("Expected unique trajectories and nonempty train/validation splits")
    shuffled = sorted(names, key=lambda x: int(x.split("_")[-1]))
    random.Random(seed).shuffle(shuffled)
    return {"train": shuffled[:train_count], "validation": shuffled[train_count:]}


def summarize_events(requested, events):
    if requested not in GOALS or not events:
        raise ValueError("Expected a known request and initial/terminal events")
    if any(events[0].get(g, False) for g in GOALS):
        raise ValueError("Initial state already satisfies a goal")
    first_times = {g: next((i for i, e in enumerate(events) if e.get(g, False)), None) for g in GOALS}
    active = [g for g in GOALS if events[-1].get(g, False)]
    earliest = min((t for t in first_times.values() if t is not None), default=None)
    first = [g for g, t in first_times.items() if t is not None and t == earliest]
    terminal = active[0] if len(active) == 1 else None
    category = "ambiguous" if len(active) > 1 else "unfinished" if not active else "correct" if terminal == requested else "wrong_goal"
    return {"requested_goal": requested, "first_goal": first[0] if len(first) == 1 else None,
            "first_goal_times": first_times, "terminal_goal": terminal, "terminal_category": category,
            "correct_ever": first_times[requested] is not None,
            "correct_then_destroyed": first_times[requested] is not None and not events[-1].get(requested, False)}


def qualification(rows, goals, expected_count, required_correct):
    counts = {}
    for goal in goals:
        selected = [r for r in rows if r["requested_goal"] == goal]
        counts[goal] = {"total": len(selected), "correct": sum(r["terminal_category"] == "correct" for r in selected)}
    return {"qualified": all(c["total"] == expected_count and c["correct"] >= required_correct for c in counts.values()), "counts": counts}


def instruction_response_summary(rows, trained_goal):
    """All four requests on matched states; failures stay in every denominator."""
    if trained_goal not in GOALS or not rows:raise ValueError('Known trained goal and nonempty rows required')
    keyed={}
    for row in rows:
        key=(row['init_index'],row['requested_goal'])
        if key in keyed:raise ValueError('Repeated initial-state/instruction pair')
        keyed[key]=row
    indices=sorted({r['init_index'] for r in rows})
    for idx in indices:
        if {g for i,g in keyed if i==idx}!=set(GOALS):raise ValueError('Every state needs four instructions')
        for field in ('initial_state_sha256','scene_xml_sha256','seed'):
            if len({keyed[idx,g][field] for g in GOALS})!=1:raise ValueError('Instructions have mismatched reset or inference seed')
    columns=(*GOALS,'unfinished','ambiguous')
    matrix={g:{c:0 for c in columns} for g in GOALS}
    for r in rows:
        outcome=r['terminal_category'] if r['terminal_category'] in ('unfinished','ambiguous') else r['terminal_goal']
        if outcome not in columns:raise ValueError('Invalid terminal outcome')
        matrix[r['requested_goal']][outcome]+=1
    novel=[r for r in rows if r['requested_goal']!=trained_goal]
    control_indices=[idx for idx in indices if keyed[idx,trained_goal]['terminal_category']=='correct']
    conditioned=[r for r in novel if r['init_index'] in control_indices]
    intrusion=sum(r['terminal_goal']==trained_goal for r in novel)
    conditional_intrusion=sum(r['terminal_goal']==trained_goal for r in conditioned)
    return {'trained_goal':trained_goal,'initial_states':len(indices),'episodes':len(rows),'terminal_matrix':matrix,
            'trained_instruction_correct':len(control_indices),
            'other_instructions':{'total':len(novel),'requested_goal_correct':sum(r['terminal_category']=='correct' for r in novel),
                                  'trained_goal_terminal':intrusion,'trained_goal_terminal_rate':intrusion/len(novel)},
            'conditioned_on_trained_instruction_success':{'control_indices':control_indices,'total':len(conditioned),
                      'trained_goal_terminal':conditional_intrusion,
                      'trained_goal_terminal_rate':conditional_intrusion/len(conditioned) if conditioned else None}}
