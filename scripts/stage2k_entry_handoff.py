"""Development-only old-entrance state source x continuation-policy diagnostic.

Requires the scene-aware stage1 modules already present on the stage2 server.
This script never trains, changes checkpoints, or selects states by eventual success.
Fixed centres are supplied explicitly; prefix and continuation outcomes stay separate.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


OLD_GOALS = ("left", "front")


def entry_eligible(step, position, initial_height, centre, flags, reached_before,
                   action_steps=10, radius=0.08, lift=0.03, min_step=20, competing_centres=()):
    """First eligible *pre-action* planning boundary; no future outcome is used."""
    values = [*position, initial_height, *centre, *(v for c in competing_centres for v in c)]
    if any(not math.isfinite(float(x)) for x in values):
        raise ValueError("Nonfinite entrance geometry")
    if action_steps <= 0 or radius <= 0 or lift < 0 or min_step < 0:
        raise ValueError("Invalid entrance rule")
    distance = math.dist(position[:2], centre[:2])
    unique_nearest = all(distance + 1e-9 < math.dist(position[:2], c[:2]) for c in competing_centres)
    return (step >= min_step and step % action_steps == 0 and not reached_before
            and not any(flags.values()) and position[2] - initial_height > lift
            and distance < radius and unique_nearest)


def read_centres(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    centres = data.get("centres", data.get("compartment_centres"))
    if not isinstance(centres, dict):
        raise ValueError("Centres JSON requires centres or compartment_centres")
    if not set(OLD_GOALS) <= set(centres):
        raise ValueError("Fixed centres must include both old goals")
    for goal in centres:
        values = centres.get(goal, [])
        if len(values) < 2 or not all(math.isfinite(float(x)) for x in values):
            raise ValueError(f"Missing/nonfinite fixed centre: {goal}")
    return centres, data


def summarize_handoffs(prefix_rows, continuation_rows):
    """Never hide absent entrance states inside a selected-success denominator."""
    summary = {}
    for source in ("teacher", "student"):
        for goal in OLD_GOALS:
            prefixes = [r for r in prefix_rows if r["source"] == source and r["goal"] == goal]
            cell = {"prefixes": len(prefixes), "entrances": sum(r["entry_step"] is not None for r in prefixes),
                    "missing_entrance": sum(r["entry_step"] is None for r in prefixes), "continuations": {}}
            for policy in ("teacher", "student"):
                rows = [r for r in continuation_rows if r["source"] == source
                        and r["goal"] == goal and r["continuation_policy"] == policy]
                cell["continuations"][policy] = {"total": len(rows),
                    "correct_ever": sum(r["correct_ever"] for r in rows),
                    "first_goal_counts": {g: sum(r["first_goal"] == g for r in rows)
                                          for g in (*OLD_GOALS, "right", "back", None)},
                    "terminal_correct": sum(r["terminal_category"] == "correct" for r in rows),
                    "correct_then_lost": sum(r["correct_then_destroyed"] for r in rows)}
            summary[f"{source}:{goal}"] = cell
    return summary


def _write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _runtime_provenance(actors):
    """Record actual imported source, not the checkout that happens to launch us."""
    import stage1_smoke
    names = {"stage1_r0_eval", "stage1_protocol", "stage1_smoke",
             "lerobot.policies.common.flow_matching"}
    for actor in actors.values():
        names.update((actor.policy.__class__.__module__, actor.policy.model.__class__.__module__))
        expert = getattr(actor.policy.model, "vlm_with_expert", None)
        if expert is not None:
            names.add(expert.__class__.__module__)
    loaded = {}
    for name in sorted(names):
        module = sys.modules.get(name)
        source = getattr(module, "__file__", None)
        if source:
            path = Path(source).resolve()
            loaded[name] = {"path": str(path), "sha256": _sha(path)}
        else:
            loaded[name] = {"status": "not_loaded"}
    runtime_root = Path(stage1_smoke.__file__).resolve().parents[1]
    result = {"python": sys.executable, "runtime_root": str(runtime_root), "loaded_sources": loaded,
              "diagnostic_source": {"path": str(Path(__file__).resolve()), "sha256": _sha(__file__)}}
    try:
        result["runtime_commit"] = subprocess.check_output(
            ["git", "-C", str(runtime_root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(
            ["git", "-C", str(runtime_root), "status", "--short"], text=True)
        result["runtime_dirty"] = bool(status.strip())
        result["runtime_status"] = status.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["runtime_git_error"] = repr(exc)
    return result


def _numeric_attributes(obj):
    """Save numeric controller/interpolator state, excluding simulator references."""
    import numpy as np
    result = {}
    for key, value in vars(obj).items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            result[key] = value
        elif isinstance(value, np.ndarray) and value.dtype.kind in "biufc" and value.size <= 10000:
            result[key] = {"array": value.tolist(), "dtype": str(value.dtype)}
        elif isinstance(value, np.generic):
            result[key] = value.item()
    return result


def _restore_attributes(obj, saved):
    import numpy as np
    for key, value in saved.items():
        if isinstance(value, dict) and "array" in value:
            value = np.array(value["array"], dtype=value["dtype"])
        setattr(obj, key, copy.deepcopy(value))


def _control_state(env):
    out = {"environment": {}, "robots": []}
    for key in ("cur_time", "timestep", "done"):
        if hasattr(env.env, key):
            value = getattr(env.env, key)
            if isinstance(value, (bool, int, float)):
                out["environment"][key] = value
    for robot in env.robots:
        controller = robot.controller
        row = {"controller": _numeric_attributes(controller), "interpolators": {}}
        for name, value in vars(controller).items():
            if "interpolator" in name and value is not None and hasattr(value, "__dict__"):
                row["interpolators"][name] = _numeric_attributes(value)
        gripper = getattr(robot, "gripper", None)
        if gripper is not None and hasattr(gripper, "current_action"):
            row["gripper_current_action"] = gripper.current_action.tolist()
        out["robots"].append(row)
    return out


def _restore_control(env, saved):
    import numpy as np
    for key, value in saved["environment"].items():
        setattr(env.env, key, value)
    for robot, row in zip(env.robots, saved["robots"], strict=True):
        _restore_attributes(robot.controller, row["controller"])
        for name, attrs in row["interpolators"].items():
            _restore_attributes(getattr(robot.controller, name), attrs)
        if "gripper_current_action" in row:
            robot.gripper.current_action = np.array(row["gripper_current_action"])


def _snapshot(env, obs):
    import numpy as np
    import torch
    arrays = {"sim_state": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
              "torch_cpu_rng": torch.get_rng_state().cpu().numpy().copy(),
              "torch_cuda_rng": torch.cuda.get_rng_state().cpu().numpy().copy()}
    for key in ("ctrl", "mocap_pos", "mocap_quat", "act", "qacc_warmstart", "userdata"):
        if hasattr(env.sim.data, key):
            arrays[f"data_{key}"] = np.asarray(getattr(env.sim.data, key)).copy()
    for key, value in obs.items():
        if hasattr(value, "shape"):
            arrays[f"obs_{key}"] = np.asarray(value).copy()
    rng = np.random.get_state()
    meta = {"control": _control_state(env), "numpy_rng": [rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]]}
    return arrays, meta


def _restore_snapshot(env, scene_xml, arrays, meta, seed):
    import numpy as np
    import torch
    env.seed(seed)
    env.reset()
    env.reset_from_xml_string(scene_xml)
    env.sim.reset()
    env.set_state(arrays["sim_state"])
    for key, value in arrays.items():
        if key.startswith("data_"):
            target = getattr(env.sim.data, key[5:])
            target[...] = value
    _restore_control(env, meta["control"])
    env.sim.forward()
    if "data_qacc_warmstart" in arrays:
        env.sim.data.qacc_warmstart[...] = arrays["data_qacc_warmstart"]
    env.check_success()
    env._post_process()
    env._update_observables(force=True)
    obs = env.env._get_observations()
    state_error = float(np.max(np.abs(np.asarray(env.get_sim_state()) - arrays["sim_state"])))
    obs_errors = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat",
                "robot0_gripper_qpos"):
        reference = arrays[f"obs_{key}"]
        error = float(np.max(np.abs(np.asarray(obs[key]).astype(np.float64) - reference.astype(np.float64))))
        obs_errors[key] = error
        tolerance = 1.0 if "image" in key else 1e-6
        if error > tolerance:
            raise RuntimeError(f"Snapshot observation mismatch: {key} max={error} tol={tolerance}")
    if state_error > 1e-8:
        raise RuntimeError(f"Snapshot physical state mismatch: {state_error}")
    torch.set_rng_state(torch.from_numpy(arrays["torch_cpu_rng"].copy()))
    torch.cuda.set_rng_state(torch.from_numpy(arrays["torch_cuda_rng"].copy()))
    rng = meta["numpy_rng"]
    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
    return obs, {"sim_state_max_abs": state_error, "observation_max_abs": obs_errors}


class Actor:
    def __init__(self, checkpoint, model_config, load_policy):
        from lerobot.policies import make_pre_post_processors
        self.policy = load_policy(checkpoint)
        self.policy.config.n_action_steps = model_config["n_action_steps"]
        self.policy.config.num_steps = model_config["flow_steps"]
        if int(getattr(self.policy.config, "n_obs_steps", 1)) != 1:
            raise ValueError("Snapshot queue reset currently supports n_obs_steps=1 only")
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)

    def reset(self):
        self.policy.reset()

    def action(self, obs, instruction):
        import numpy as np
        import torch
        from robosuite.utils.transform_utils import quat2axisangle
        state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()),
                                obs["robot0_gripper_qpos"])).astype(np.float32)
        batch = {"observation.state": torch.from_numpy(state), "task": instruction}
        for key, raw in (("observation.images.image", "agentview_image"),
                         ("observation.images.image2", "robot0_eye_in_hand_image")):
            image = np.ascontiguousarray(obs[raw][::-1, ::-1])
            batch[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            action = self.post(self.policy.select_action(self.pre(batch))).float().cpu().numpy().reshape(-1)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError("Invalid policy action")
        return action


def main(args):
    if args.runtime_scripts:
        sys.path.insert(0, str(args.runtime_scripts.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from stage1_r0_eval import canonical_scene, events, make_env, reset_common, task_resources
    from stage1_smoke import DEFAULT_SCENE, load_policy
    from stage1_protocol import summarize_events

    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; failed attempts are retained")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    centres, centre_source = read_centres(args.centres)
    e = config["evaluation"]
    if not set(args.indices) <= set(e["development_indices"]):
        raise ValueError("This pilot uses development initial states only")
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("Repeated initial states")
    tasks, paths, predicates, states = task_resources(config.get("scene", DEFAULT_SCENE))
    common = states[e["common_state_source_goal"]]
    scene_xml = canonical_scene(paths[e["common_state_source_goal"]], config["seed"])
    geometry_scene_hash = centre_source.get("canonical_scene_xml_sha256")
    if geometry_scene_hash and geometry_scene_hash != hashlib.sha256(scene_xml.encode()).hexdigest():
        raise ValueError("Fixed geometry belongs to a different canonical scene")
    (args.output / "scene.xml").write_text(scene_xml)
    models = {"teacher": args.teacher, "student": args.student}
    manifest = {"status": "running", "scope": "development entrance handoff, not training or formal qualification",
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "centres_source": centre_source, "centres_sha256": _sha(args.centres),
                "scene_xml_sha256": hashlib.sha256(scene_xml.encode()).hexdigest(),
                "model_sha256": {k: _sha(v / "model.safetensors") for k, v in models.items()},
                "snapshot_rng_rule": "both continuations restore the same source-boundary torch and numpy RNG",
                "entry_rule": "first pre-action planning boundary: fixed radius/lift, requested goal uniquely nearest among all supplied centres, before any goal",
                "natural_self_check": "source-policy first action and first physical next state compared after full restore",
                "prefix_rows": [], "continuation_rows": []}
    _write(args.output / "manifest.json", manifest)
    actors = {name: Actor(path, config["model"], load_policy) for name, path in models.items()}
    manifest["runtime"] = _runtime_provenance(actors)
    _write(args.output / "manifest.json", manifest)
    snapshots = []
    action_steps = config["model"]["n_action_steps"]

    try:
        for source, actor in actors.items():
            for goal in OLD_GOALS:
                env = make_env(paths[goal])
                try:
                    for index in args.indices:
                        if time.monotonic() - started > args.max_minutes * 60:
                            raise TimeoutError("Development experiment walltime limit")
                        seed = config["seed"] + index
                        torch.manual_seed(seed); np.random.seed(seed)
                        obs, _ = reset_common(env, common[index], seed, e["settle_steps"], scene_xml)
                        actor.reset()
                        initial_height = float(obs[f"{args.object}_pos"][2])
                        trace = [events(env, predicates)]
                        reached = any(trace[0].values())
                        snapshot = None
                        for step in range(args.horizon):
                            position = np.asarray(obs[f"{args.object}_pos"])
                            if snapshot is None and entry_eligible(step, position, initial_height, centres[goal],
                                    trace[-1], reached, action_steps, args.entry_radius, args.min_lift, args.min_step,
                                    competing_centres=[c for g, c in centres.items() if g != goal]):
                                arrays, metadata = _snapshot(env, obs)
                                ident = f"{source}_{goal}_{index:02d}"
                                snapshot = {"id": ident, "source": source, "goal": goal, "index": index,
                                            "step": step, "seed": seed, "arrays": arrays, "metadata": metadata}
                                for raw in ("agentview_image", "robot0_eye_in_hand_image"):
                                    Image.fromarray(np.ascontiguousarray(obs[raw][::-1, ::-1])).save(
                                        args.output / f"{ident}_{raw}.png")
                            action = actor.action(obs, tasks[goal].language)
                            obs, _, _, _ = env.step(np.clip(action, -1, 1))
                            if snapshot is not None and snapshot["step"] == step:
                                snapshot["natural_first_action"] = action.copy()
                                snapshot["natural_first_next_state"] = np.asarray(env.get_sim_state()).copy()
                            flags = events(env, predicates)
                            trace.append(flags)
                            reached = reached or any(flags.values())
                        row = {"source": source, "goal": goal, "init_index": index,
                               "entry_step": snapshot["step"] if snapshot else None,
                               **summarize_events(goal, trace)}
                        manifest["prefix_rows"].append(row)
                        if snapshot:
                            snapshot["arrays"]["natural_first_action"] = snapshot["natural_first_action"]
                            snapshot["arrays"]["natural_first_next_state"] = snapshot["natural_first_next_state"]
                            np.savez_compressed(args.output / f"{snapshot['id']}_snapshot.npz", **snapshot["arrays"])
                            _write(args.output / f"{snapshot['id']}_snapshot.json", {
                                k: v for k, v in snapshot.items() if k not in
                                ("arrays", "natural_first_action", "natural_first_next_state")})
                            snapshots.append(snapshot)
                        _write(args.output / "manifest.json", manifest)
                        print(json.dumps({"phase": "prefix", **row}), flush=True)
                finally:
                    env.close()

        for snapshot in snapshots:
            goal, seed = snapshot["goal"], snapshot["seed"]
            for name, actor in actors.items():
                if time.monotonic() - started > args.max_minutes * 60:
                    raise TimeoutError("Development experiment walltime limit")
                env = make_env(paths[goal])
                try:
                    obs, restore_check = _restore_snapshot(env, scene_xml, snapshot["arrays"],
                                                          snapshot["metadata"], seed)
                    actor.reset()
                    trace = [events(env, predicates)]
                    first_action_error = first_state_error = None
                    closest = {g: float("inf") for g in OLD_GOALS}
                    for step in range(args.continuation):
                        action = actor.action(obs, tasks[goal].language)
                        obs, _, _, _ = env.step(np.clip(action, -1, 1))
                        trace.append(events(env, predicates))
                        position = np.asarray(obs[f"{args.object}_pos"])
                        for g in OLD_GOALS:
                            closest[g] = min(closest[g], float(np.linalg.norm(position[:2] - np.asarray(centres[g])[:2])))
                        if step == 0 and name == snapshot["source"]:
                            first_action_error = float(np.max(np.abs(action - snapshot["natural_first_action"])))
                            first_state_error = float(np.max(np.abs(np.asarray(env.get_sim_state())
                                                         - snapshot["natural_first_next_state"])))
                    self_ok = (first_action_error is None or
                               (first_action_error <= args.self_action_tolerance and first_state_error <= 1e-5))
                    row = {"source": snapshot["source"], "goal": goal, "init_index": snapshot["index"],
                           "entry_step": snapshot["step"], "continuation_policy": name,
                           "steps": args.continuation, "restore_check": restore_check,
                           "self_first_action_max_abs": first_action_error,
                           "self_first_next_state_max_abs": first_state_error,
                           "self_first_step_passed": self_ok, "closest_old_target_xy_m": closest,
                           **summarize_events(goal, trace)}
                    manifest["continuation_rows"].append(row)
                    _write(args.output / "manifest.json", manifest)
                    print(json.dumps({"phase": "continuation", **row}), flush=True)
                    if not self_ok:
                        raise RuntimeError("Self-continuation first-step mismatch; retained evidence, stop interpretation")
                finally:
                    env.close()
        manifest["summary"] = summarize_handoffs(manifest["prefix_rows"], manifest["continuation_rows"])
        manifest.update(status="complete", seconds=time.monotonic() - started,
                        prefix_episodes=len(manifest["prefix_rows"]),
                        continuation_episodes=len(manifest["continuation_rows"]))
        _write(args.output / "manifest.json", manifest)
        _write(args.output / "COMPLETE.json", {"status": "complete", "seconds": manifest["seconds"]})
        print("ENTRY_HANDOFF_COMPLETE", json.dumps(manifest["summary"]), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc), seconds=time.monotonic() - started)
        _write(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-scripts", type=Path, help="Use the audited scene-aware server stage1 modules")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--centres", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--continuation", type=int, default=120)
    parser.add_argument("--entry-radius", type=float, default=0.08)
    parser.add_argument("--min-lift", type=float, default=0.03)
    parser.add_argument("--min-step", type=int, default=20)
    parser.add_argument("--object", default="black_book_1")
    parser.add_argument("--max-minutes", type=float, default=100)
    parser.add_argument("--self-action-tolerance", type=float, default=1e-5)
    main(parser.parse_args())
