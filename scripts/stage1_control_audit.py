"""No-training P0 audit of saved-demo replay and observation/action timing.

Writes new diagnostic artifacts only. It does not shift labels, change the R0
protocol, or certify the controller from a small set of successful replays.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from stage1_protocol import remap_demo_xml, summarize_events
from stage1_r0_eval import events, make_env, task_resources
from stage1_smoke import ROOT, dump, sources


CASES = (("front", "demo_1"), ("right", "demo_1"),
         ("right", "demo_0"), ("right", "demo_2"))
IMAGE_KEYS = {"agentview_rgb": "agentview_image", "eye_in_hand_rgb": "robot0_eye_in_hand_image"}


def proprio(obs):
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle

    return np.concatenate((obs["robot0_eef_pos"],
                           quat2axisangle(obs["robot0_eef_quat"].copy()),
                           obs["robot0_gripper_qpos"])).astype(float)


def proprio_error(actual, reference):
    import numpy as np
    from robosuite.utils.transform_utils import axisangle2quat, quat2mat

    a = quat2mat(axisangle2quat(actual[3:6]))
    b = quat2mat(axisangle2quat(reference[3:6]))
    cosine = np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)
    return {"eef_position_l2_m": float(np.linalg.norm(actual[:3] - reference[:3])),
            "eef_rotation_angle_rad": float(np.arccos(cosine)),
            "gripper_position_l2": float(np.linalg.norm(actual[6:] - reference[6:]))}


def sim_error(actual, reference, model):
    import numpy as np

    nq, nv = model.nq, model.nv
    return {"time_abs_s": float(abs(actual[0] - reference[0])),
            "qpos_l2": float(np.linalg.norm(actual[1:1+nq] - reference[1:1+nq])),
            "qvel_l2": float(np.linalg.norm(actual[1+nq:1+nq+nv] - reference[1+nq:1+nq+nv]))}


def image_error(obs, demo, index):
    import numpy as np

    return {raw: float(np.abs(np.asarray(obs[live], dtype=float)
                             - demo[f"obs/{raw}"][index].astype(float)).mean())
            for raw, live in IMAGE_KEYS.items()}


def selected_indices(actions, stored):
    """Cover the trajectory, maximum lift, and gripper transitions without labels."""
    import numpy as np

    length = len(actions)
    indices = {0, 1, length//4, length//2, 3*length//4, length-2, length-1,
               int(np.argmax(stored[:, 2]))}
    transitions = np.flatnonzero(np.diff(actions[:, -1] > 0)) + 1
    for index in transitions:
        indices.update((int(index)-1, int(index), int(index)+1))
    return sorted(i for i in indices if 0 <= i < length)


def reset_saved(env, xml, state, seed):
    env.seed(seed)
    env.reset()
    env.reset_from_xml_string(xml)
    env.sim.reset()
    return env.set_init_state(state)


def replay(args, paths, predicates, raw_paths, goal, name, repetition):
    import h5py
    import numpy as np
    import robosuite
    from libero.libero import get_libero_path
    from PIL import Image, ImageDraw

    folder = args.output / f"{goal}_{name}_repeat{repetition}"
    folder.mkdir()
    env = make_env(paths[goal])
    started = time.monotonic()
    try:
        with h5py.File(raw_paths[goal]) as stream:
            demo = stream["data"][name]
            xml = remap_demo_xml(demo.attrs["model_file"], get_libero_path("assets"),
                                 Path(robosuite.__file__).parent)
            actions = demo["actions"][:]
            states = demo["states"][:]
            stored = np.concatenate((demo["obs/ee_states"][:], demo["obs/gripper_states"][:]), axis=1)
            assert len(actions) == len(states) == len(stored)
            # Match the successful historical verify flow, including its terminal
            # state check. No extra settle controls are inserted before replay.
            reset_saved(env, xml, states[-1], args.seed)
            saved_terminal = events(env, predicates)
            assert saved_terminal[goal] and sum(saved_terminal.values()) == 1
            env.reset_from_xml_string(xml)
            env.sim.reset()
            obs = env.set_init_state(states[0])
            initial = np.asarray(env.get_sim_state()).copy()
            trace = [events(env, predicates)]
            first_divergence = None
            samples = selected_indices(actions, stored)
            for index, action in enumerate(actions):
                before = proprio(obs)
                sim_before = np.asarray(env.get_sim_state()).copy()
                obs, _, _, _ = env.step(action)
                after = proprio(obs)
                sim_after = np.asarray(env.get_sim_state()).copy()
                flags = events(env, predicates)
                trace.append(flags)
                next_error = sim_error(sim_after, states[index+1], env.sim.model) if index+1 < len(states) else None
                # This is a numerical diagnostic threshold, not a task gate.
                if next_error and first_divergence is None and next_error["qpos_l2"] > args.qpos_threshold:
                    first_divergence = index
                row = {"index": index, "action": action.tolist(), "before_proprio": before.tolist(),
                       "after_proprio": after.tolist(), "predicates_after": flags,
                       "before_sim_vs_saved_t": sim_error(sim_before, states[index], env.sim.model),
                       "after_sim_vs_saved_t_plus_1": next_error,
                       "saved_obs_t_vs_before": proprio_error(before, stored[index]),
                       "saved_obs_t_vs_after": proprio_error(after, stored[index])}
                if index in samples:
                    row["images_after_vs_saved_t_mae"] = image_error(obs, demo, index)
                with (folder / "replay_trace.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            summary = summarize_events(goal, trace)
            replay_terminal = proprio(obs)

            # Restore each selected saved simulator state independently. This
            # avoids interpreting accumulated replay drift as a timing offset.
            timing = []
            for index in samples:
                obs_before = reset_saved(env, xml, states[index], args.seed)
                before = proprio(obs_before)
                images_before = {raw: np.asarray(obs_before[live]).copy() for raw, live in IMAGE_KEYS.items()}
                before_images_error = image_error(obs_before, demo, index)
                obs_after, _, _, _ = env.step(actions[index])
                row = {"index": index, "action": actions[index].tolist(),
                       "before_proprio": before.tolist(), "after_proprio": proprio(obs_after).tolist(),
                       "saved_proprio": stored[index].tolist(),
                       "saved_obs_t_vs_before": proprio_error(before, stored[index]),
                       "saved_obs_t_vs_after": proprio_error(proprio(obs_after), stored[index]),
                       "images_before_vs_saved_t_mae": before_images_error,
                       "images_after_vs_saved_t_mae": image_error(obs_after, demo, index),
                       "after_sim_vs_saved_t_plus_1": sim_error(np.asarray(env.get_sim_state()), states[index+1], env.sim.model)
                       if index+1 < len(states) else None}
                if index:
                    row["saved_obs_t_minus_1_vs_before"] = proprio_error(before, stored[index-1])
                timing.append(row)
                canvas = Image.new("RGB", (384, 286), "white")
                draw = ImageDraw.Draw(canvas)
                for column, label in enumerate(("saved obs[t]", "restore state[t]", "after action[t]")):
                    draw.text((column*128+2, 2), label, fill="black")
                for camera, (raw, live) in enumerate(IMAGE_KEYS.items()):
                    pictures = (demo[f"obs/{raw}"][index], images_before[raw], obs_after[live])
                    for column, picture in enumerate(pictures):
                        canvas.paste(Image.fromarray(np.ascontiguousarray(picture[::-1, ::-1])), (column*128, 24+camera*130))
                canvas.save(folder / f"timing_{index:04d}.png")
            dump(folder / "timing_samples.json", timing)
            medians = {}
            for phase in ("before", "after"):
                medians[phase] = {key: float(np.median([r[f"saved_obs_t_vs_{phase}"][key] for r in timing]))
                                  for key in timing[0][f"saved_obs_t_vs_{phase}"]}
            report = {"goal": goal, "demo": name, "repetition": repetition, "seed": args.seed,
                      "frames": len(actions), "saved_terminal": saved_terminal,
                      "action_replay": summary, "timing_indices": samples,
                      "timing_error_medians": medians,
                      "first_qpos_divergence_action_index": first_divergence,
                      "qpos_divergence_threshold": args.qpos_threshold,
                      "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                      "scene_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                      "replay_terminal_proprio": replay_terminal.tolist(),
                      "seconds": time.monotonic()-started}
            dump(folder / "summary.json", report)
            print(json.dumps(report), flush=True)
            return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--qpos-threshold", type=float, default=0.01)
    args = parser.parse_args()
    if args.repetitions < 1 or args.qpos_threshold <= 0:
        parser.error("positive repetitions and qpos threshold required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    _, paths, predicates, _ = task_resources()
    raw_paths = sources()
    script = Path(__file__)
    report = {"status": "running", "purpose": "P0_control_and_timing_no_training",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
              "raw_sources": {g: str(p) for g, p in raw_paths.items()},
              "settle_steps": 0, "cases": [],
              "limitations": ["Raw flattened simulator states do not restore all controller, observable, or contact-solver history.",
                              "Single-state timing probes reset controller history and are diagnostics, not exact expert continuation.",
                              "qpos/qvel L2 combine coordinates with different units; they are divergence diagnostics.",
                              "No labels or data conversion are changed by this audit."]}
    dump(args.output / "audit_summary.json", report)
    start = time.monotonic()
    for goal, name in CASES:
        for repetition in range(args.repetitions):
            report["cases"].append(replay(args, paths, predicates, raw_paths, goal, name, repetition))
            dump(args.output / "audit_summary.json", report)
    report.update(status="complete", seconds=time.monotonic()-start)
    dump(args.output / "audit_summary.json", report)
    dump(args.output / "COMPLETE.json", {"status": "complete", "episodes": len(report["cases"]), "seconds": report["seconds"]})


if __name__ == "__main__":
    main()
