"""Find when the goal predicate first holds inside each demonstration by restoring stored sim states.

Read-only: restores demo["states"][k] (no control replay), evaluates all four compartment predicates,
and reports first-success step, frames kept afterwards, and gripper command/opening around that step.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from stage1_smoke import sources  # noqa: E402  (sets LIBERO_CONFIG_PATH before libero is imported)
from stage1_protocol import GOALS, remap_demo_xml  # noqa: E402
from stage1_r0_eval import events, make_env, task_resources  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import robosuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402


def main(output, seed):
    if output.exists():
        raise FileExistsError(output)
    _, paths, predicates, _ = task_resources()
    report = {}
    for goal, path in sources().items():
        env = make_env(paths[goal])
        rows = []
        try:
            with h5py.File(path) as h:
                names = sorted(h["data"], key=lambda x: int(x.split("_")[-1]))
                for name in names:
                    d = h["data"][name]
                    xml = remap_demo_xml(d.attrs["model_file"], get_libero_path("assets"), Path(robosuite.__file__).parent)
                    env.seed(seed); env.reset(); env.reset_from_xml_string(xml); env.sim.reset()
                    states = d["states"][:]
                    flags = []
                    for s in states:
                        env.set_init_state(s)
                        flags.append(events(env, predicates))
                    hit = [i for i, f in enumerate(flags) if f[goal]]
                    other = sorted({g for f in flags for g in GOALS if g != goal and f[g]})
                    grip = d["actions"][:, -1]
                    fingers = d["obs/gripper_states"][:]
                    width = np.abs(fingers[:, 0] - fingers[:, 1])
                    first = hit[0] if hit else None
                    rows.append({
                        "demo": name, "frames": len(states), "first_success": first,
                        "frames_after_first_success": None if first is None else len(states) - 1 - first,
                        "success_frames": len(hit), "terminal_success": bool(flags[-1][goal]),
                        "other_goal_predicates_seen": other,
                        "gripper_cmd_closed_at_first_success": None if first is None else bool(grip[first] > 0),
                        "any_open_cmd_from_first_success": None if first is None else bool((grip[first:] < 0).any()),
                        "width_at_first_success": None if first is None else float(width[first]),
                        "width_at_end": float(width[-1]),
                    })
        finally:
            env.close()
        after = np.array([r["frames_after_first_success"] for r in rows if r["first_success"] is not None])
        report[goal] = {
            "summary": {
                "demos": len(rows), "with_success": int(len(after)),
                "frames_after_first_success_min_median_max": [int(after.min()), float(np.median(after)), int(after.max())],
                "gripper_closed_at_first_success": int(sum(bool(r["gripper_cmd_closed_at_first_success"]) for r in rows)),
                "any_open_cmd_after_first_success": int(sum(bool(r["any_open_cmd_from_first_success"]) for r in rows)),
                "demos_seeing_other_goal": int(sum(bool(r["other_goal_predicates_seen"]) for r in rows)),
            },
            "demos": rows,
        }
        print(goal, json.dumps(report[goal]["summary"]), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261021)
    a = p.parse_args()
    main(a.output, a.seed)
