"""Compare raw obs[t] with rendering/kinematics of saved state[t] and state[t+1].

No action replay is used, so missing controller history cannot confound this
particular timing comparison. Uses only trajectories in the old training split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from stage1_control_audit import image_error, proprio, proprio_error, selected_indices
from stage1_protocol import remap_demo_xml
from stage1_r0_eval import make_env, task_resources
from stage1_smoke import ROOT, dump, sources


def main():
    import h5py
    import numpy as np
    import robosuite
    from libero.libero import get_libero_path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--per-goal", type=int, default=2)
    args = p.parse_args()
    if args.per_goal < 1:
        p.error("per-goal must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    audit = json.loads(args.audit.read_text())
    _, paths, _, _ = task_resources()
    raw_paths = sources()
    report = {"status": "running", "purpose": "saved_state_timing_without_control_replay",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "split_sha256": audit["split_sha256"], "cases": []}
    dump(args.output / "summary.json", report)
    started = time.monotonic()
    for goal in audit["config"]["old_goals"]:
        env = make_env(paths[goal])
        try:
            with h5py.File(raw_paths[goal]) as handle:
                for name in audit["splits"][goal]["train"][:args.per_goal]:
                    demo = handle["data"][name]
                    states = demo["states"][:]
                    actions = demo["actions"][:]
                    stored = np.concatenate((demo["obs/ee_states"][:], demo["obs/gripper_states"][:]), axis=1)
                    xml = remap_demo_xml(demo.attrs["model_file"], get_libero_path("assets"), Path(robosuite.__file__).parent)
                    env.seed(audit["config"]["seed"])
                    env.reset()
                    env.reset_from_xml_string(xml)
                    env.sim.reset()
                    indices = [t for t in selected_indices(actions, stored) if t+1 < len(states)]
                    rows = []
                    for t in indices:
                        row = {"index": t, "saved_time_delta_s": float(states[t+1, 0]-states[t, 0])}
                        for label, index in (("state_t", t), ("state_t_plus_1", t+1)):
                            obs = env.set_init_state(states[index])
                            row[label] = {"proprio_error": proprio_error(proprio(obs), stored[t]),
                                          "image_mae": image_error(obs, demo, t)}
                        rows.append(row)
                    dump(args.output / f"{goal}_{name}.json", rows)
                    medians = {label: {key: statistics.median(row[label]["proprio_error"][key] for row in rows)
                                       for key in rows[0][label]["proprio_error"]}
                               for label in ("state_t", "state_t_plus_1")}
                    images = {label: {key: statistics.median(row[label]["image_mae"][key] for row in rows)
                                      for key in rows[0][label]["image_mae"]}
                              for label in ("state_t", "state_t_plus_1")}
                    result = {"goal": goal, "demo": name, "samples": len(rows),
                              "median_proprio_error": medians, "median_image_mae": images,
                              "saved_time_delta_s_range": [min(r["saved_time_delta_s"] for r in rows), max(r["saved_time_delta_s"] for r in rows)]}
                    report["cases"].append(result)
                    dump(args.output / "summary.json", report)
                    print(json.dumps(result), flush=True)
        finally:
            env.close()
    report.update(status="complete", seconds=time.monotonic()-started,
                  limitations=["This checks temporal correspondence of stored observations and simulator states, not whether a one-step offset explains low policy success.",
                               "It does not change training examples or decide a new data protocol automatically."])
    dump(args.output / "summary.json", report)
    dump(args.output / "COMPLETE.json", {"status": "complete", "seconds": report["seconds"],
                                         "trajectories": len(report["cases"]), "snapshot_pairs": sum(c["samples"] for c in report["cases"])})


if __name__ == "__main__":
    main()
