"""P0/P1: replay sealed entrance states with shared continuation-noise schedules.

No prefixes are recollected and no policy is trained. Each saved state is tested
with both original policies and the same predeclared CPU Gaussian schedules.
Native post-step and planning-boundary-refreshed goal events are both recorded.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time


POLICIES = ("teacher", "student")
GOALS = ("left", "front", "right", "back")


def jobs_for(snapshot_ids, seeds):
    if not snapshot_ids or len(set(snapshot_ids)) != len(snapshot_ids):
        raise ValueError("Need nonempty unique snapshot IDs")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 for s in seeds):
        raise ValueError("Need nonnegative unique torch-compatible noise seeds")
    # No source/goal/state/model offsets: each seed is the SAME schedule everywhere.
    return [(snapshot, policy, seed) for snapshot in snapshot_ids for seed in seeds for policy in POLICIES]


def summarize_rows(rows, snapshot_ids, seeds):
    expected = set(jobs_for(snapshot_ids, seeds))
    keyed = {(r["snapshot_id"], r["continuation_policy"], r["noise_seed"]): r for r in rows}
    if len(keyed) != len(rows) or set(keyed) != expected:
        raise ValueError("Incomplete or duplicate continuation matrix")
    cells, snapshots = {}, {}
    for source in POLICIES:
        for goal in ("left", "front"):
            cell = {}
            for policy in POLICIES:
                part = [r for r in rows if r["source"] == source and r["goal"] == goal
                        and r["continuation_policy"] == policy]
                cell[policy] = {"continuations": len(part),
                    "distinct_snapshots": len({r["snapshot_id"] for r in part}),
                    "correct_ever": sum(r["correct_ever"] for r in part),
                    "terminal_correct": sum(r["terminal_category"] == "correct" for r in part),
                    "first_goal_counts": dict(Counter(r["first_goal"] or "none_or_tied" for r in part))}
            cells[f"{source}:{goal}"] = cell
    for snapshot in snapshot_ids:
        wins = Counter()
        entry = {policy: sum(keyed[snapshot, policy, seed]["correct_ever"] for seed in seeds)
                 for policy in POLICIES}
        for seed in seeds:
            t, s = (bool(keyed[snapshot, policy, seed]["correct_ever"]) for policy in POLICIES)
            wins["both" if t and s else "teacher_only" if t else "student_only" if s else "neither"] += 1
        snapshots[snapshot] = {"seeds": len(seeds), "correct_ever": entry, "paired_outcomes": dict(wins)}
    return {"cells": cells, "per_snapshot": snapshots, "distinct_snapshots": len(snapshot_ids),
            "noise_repeats_per_snapshot_policy": len(seeds), "continuations": len(rows)}


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class SharedNoise:
    """Override only the Gaussian draw, using explicit CPU PRNG and recorded bytes."""
    def __init__(self, actor, seed):
        import torch
        self.model = actor.policy.model
        self.original = self.model.sample_noise
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.records = []

    def draw(self, shape, device):
        import torch
        noise = torch.randn(tuple(shape), generator=self.generator, dtype=torch.float32, device="cpu")
        self.records.append({"chunk": len(self.records), "shape": list(shape),
                             "sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest()})
        return noise.to(device=device)

    def __enter__(self):
        self.model.sample_noise = self.draw
        return self

    def __exit__(self, *unused):
        self.model.sample_noise = self.original


def _plain(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def observation_fields(obs):
    """Sensor values exactly as exposed to policy; do not perform extra forward."""
    result = {}
    for key, value in obs.items():
        if "image" not in key and (hasattr(value, "shape") or isinstance(value, (bool, int, float))):
            result[key] = _plain(value)
    return result


def physical_fields(env, object_name):
    import numpy as np
    result = {"sim_state": np.asarray(env.get_sim_state()).tolist(),
              "qpos": np.asarray(env.sim.data.qpos).tolist(),
              "qvel": np.asarray(env.sim.data.qvel).tolist()}
    try:
        joint = env.env.objects_dict[object_name].joints[0]
        position = np.asarray(env.sim.data.get_joint_qpos(joint))
        velocity = np.asarray(env.sim.data.get_joint_qvel(joint))
        if position.shape != (7,):
            raise ValueError(f"Expected 7D free joint, got {position.shape}")
        result["book_free_joint"] = {"name": joint, "position_xyz": position[:3].tolist(),
            "quaternion_wxyz": position[3:].tolist(), "velocity": velocity.tolist(),
            "source": "integrated qpos, no extra kinematic forward"}
    except (AttributeError, KeyError, ValueError) as error:
        result["book_free_joint"] = {"available": False, "reason": repr(error)}
    return result


def relative_pose(obs, object_name):
    import numpy as np
    q = np.asarray(obs["robot0_eef_quat"], dtype=float)
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    relative = rotation.T @ (np.asarray(obs[f"{object_name}_pos"])-np.asarray(obs["robot0_eef_pos"]))
    # Quaternion product inv(eef) * book, both sensor quaternions use xyzw.
    a = np.array([-x, -y, -z, w]); b = np.asarray(obs[f"{object_name}_quat"])
    vector = a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3])
    scalar = a[3]*b[3] - np.dot(a[:3], b[:3])
    return {"position_in_eef_frame": relative.tolist(),
            "quaternion_xyzw": np.r_[vector, scalar].tolist(), "source": "as-seen sensor observation"}


def contact_records(env):
    """Read native solver contacts, trying supported force APIs without forward."""
    import numpy as np
    sim = env.sim
    contacts = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        row = {"geom1_id": int(contact.geom1), "geom2_id": int(contact.geom2),
               "geom1": sim.model.geom_id2name(contact.geom1),
               "geom2": sim.model.geom_id2name(contact.geom2),
               "distance": float(contact.dist), "position": np.asarray(contact.pos).tolist()}
        force = np.zeros(6, dtype=float)
        failures = []
        try:
            import mujoco
            mujoco.mj_contactForce(getattr(sim.model, "_model", sim.model),
                                  getattr(sim.data, "_data", sim.data), index, force)
            row.update(force=force.tolist(), force_api="mujoco.mj_contactForce")
        except (ImportError, AttributeError, TypeError) as error:
            failures.append(type(error).__name__)
            try:
                from mujoco_py import functions
                functions.mj_contactForce(sim.model, sim.data, index, force)
                row.update(force=force.tolist(), force_api="mujoco_py.functions.mj_contactForce")
            except (ImportError, AttributeError, TypeError) as error2:
                failures.append(type(error2).__name__)
                row.update(force=None, force_unavailable_reason=", ".join(failures))
        contacts.append(row)
    return contacts


def resource_check(output, started, minutes, max_bytes):
    if time.monotonic() - started > minutes * 60:
        raise TimeoutError("Registered walltime limit")
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; stop without deleting prior data")
    used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if used > max_bytes:
        raise RuntimeError("Registered output-data budget exceeded")


def main(args):
    if args.runtime_scripts:
        sys.path.insert(0, str(args.runtime_scripts.resolve()))
    import numpy as np
    import torch
    from stage1_r0_eval import events, make_env, task_resources
    from stage1_smoke import DEFAULT_SCENE, load_policy
    from stage1_protocol import summarize_events
    from stage2k_entry_handoff import (Actor, _fresh_observation, _restore_snapshot,
                                      _runtime_provenance, _sha)

    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; failures are never overwritten")
    source_path = args.snapshots / "manifest.json"
    source = json.loads(source_path.read_text())
    if source["status"] != "complete":
        raise ValueError("Snapshot source experiment is incomplete")
    helper_path = Path(sys.modules[Actor.__module__].__file__).resolve()
    helper_hash = _sha(helper_path)
    original_helper_hash = source.get("runtime", {}).get("diagnostic_source", {}).get("sha256")
    if not original_helper_hash or helper_hash != original_helper_hash:
        raise ValueError("Snapshot restore/observation helper differs from sealed source implementation")
    all_ids = [f"{r['source']}_{r['goal']}_{r['init_index']:02d}" for r in source["prefix_rows"]
               if r["entry_step"] is not None]
    selected = args.snapshot_ids or all_ids
    if not set(selected) <= set(all_ids):
        raise ValueError("Snapshot filter includes an unsealed state")
    jobs = jobs_for(selected, args.seeds)
    if args.horizon != 120:
        raise ValueError("This registered P1 version uses 120 steps only")
    config = json.loads(args.config.read_text())
    for key, value in (("chunk_size", 50), ("n_action_steps", 10), ("flow_steps", 10)):
        if config["model"][key] != value:
            raise ValueError(f"Registered source action protocol requires {key}={value}")
    model_paths = {name: Path(getattr(args, name) or source["arguments"][name]).resolve() for name in POLICIES}
    model_hashes = {name: _sha(path / "model.safetensors") for name, path in model_paths.items()}
    if model_hashes != source["model_sha256"]:
        raise ValueError("Policy checkpoint changed from sealed source experiment")
    xml_path = args.snapshots / "scene.xml"
    scene_xml = xml_path.read_text()
    if hashlib.sha256(scene_xml.encode()).hexdigest() != source["scene_xml_sha256"]:
        raise ValueError("Sealed scene XML changed")
    sealed = {}
    snapshots = {}
    for ident in selected:
        meta_path, arrays_path = (args.snapshots / f"{ident}_snapshot{suffix}" for suffix in (".json", ".npz"))
        metadata = json.loads(meta_path.read_text())
        if metadata["id"] != ident:
            raise ValueError("Snapshot identity mismatch")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            snapshots[ident] = ({key: arrays[key].copy() for key in arrays.files}, metadata)
        sealed[ident] = {"metadata_sha256": _sha(meta_path), "arrays_sha256": _sha(arrays_path)}
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    manifest = {"status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "P0/P1 sealed-state development confirmation, no new prefixes or training",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_manifest_sha256": _sha(source_path), "source_scene_sha256": _sha(xml_path),
        "source_prefix_rows": source["prefix_rows"], "sealed_snapshots": sealed,
        "model_sha256": model_hashes, "script_sha256": _sha(__file__),
        "source_helper_path": str(helper_path), "source_helper_sha256": helper_hash,
        "config_sha256": _sha(args.config), "selected_snapshot_ids": selected,
        "distinct_snapshot_count": len(selected), "continuation_seeds": args.seeds,
        "planned_continuations": len(jobs), "primary_horizon": 120,
        "primary_events": "native post-step except at subsequent planning boundaries, where refreshed events replace them; both saved",
        "observation_rule": "initial restored/fresh, then fresh before each new plan every 10 actions; native step observations between plans",
        "noise_rule": "explicit float32 CPU Gaussian seed, identical across source/state/goal/model, copied to GPU",
        "noise_protocol_difference": "P1 uses CPU PRNG; 2K used saved native CUDA RNG; scores are not exact reruns",
        "source_native_self_checks": [], "rows": [], "noise_schedules": {}}
    write_json(args.output / "manifest.json", manifest)
    actors = {}
    try:
        resource_check(args.output, started, args.max_minutes, args.max_output_bytes)
        actors = {name: Actor(path, config["model"], load_policy) for name, path in model_paths.items()}
        manifest["runtime"] = _runtime_provenance(actors)
        for field in ("chunk_size", "max_action_dim", "n_action_steps", "num_steps"):
            if getattr(actors["teacher"].policy.config, field) != getattr(actors["student"].policy.config, field):
                raise ValueError(f"Policy action/noise dimensions differ: {field}")
        tasks, paths, predicates, _ = task_resources(config.get("scene", DEFAULT_SCENE))
        centres = source["centres_source"]["centres"]
        action_steps = config["model"]["n_action_steps"]
        if 120 % action_steps:
            raise ValueError("Horizon must end on a complete action chunk")

        if args.self_check:
            for ident in selected:
                arrays, meta = snapshots[ident]
                actor = actors[meta["source"]]
                env = make_env(paths[meta["goal"]])
                try:
                    obs, restore = _restore_snapshot(env, scene_xml, arrays, meta["metadata"], meta["seed"])
                    actor.reset()
                    action = actor.action(obs, tasks[meta["goal"]].language)
                    env.step(np.clip(action, -1, 1))
                    action_error = float(np.max(np.abs(action - arrays["natural_first_action"])))
                    state_error = float(np.max(np.abs(np.asarray(env.get_sim_state()) - arrays["natural_first_next_state"])))
                    row = {"snapshot_id": ident, "source_policy": meta["source"], "restore_check": restore,
                           "first_action_max_abs": action_error, "first_next_state_max_abs": state_error,
                           "passed": action_error <= 1e-5 and state_error <= 1e-5}
                    manifest["source_native_self_checks"].append(row)
                    write_json(args.output / "manifest.json", manifest)
                    if not row["passed"]:
                        raise RuntimeError("Native-source self-step failed; stop before P1 scoring")
                finally:
                    env.close()

        manifest["status"] = "running"
        for ident, policy_name, seed in jobs:
            resource_check(args.output, started, args.max_minutes, args.max_output_bytes)
            arrays, meta = snapshots[ident]
            goal, actor = meta["goal"], actors[policy_name]
            name = f"{ident}__{policy_name}__seed{seed}"
            folder = args.output / name
            folder.mkdir()
            env = make_env(paths[goal])
            image_frames = {key: [] for key in ("agentview_image", "robot0_eye_in_hand_image")}
            row = {"snapshot_id": ident, "source": meta["source"], "goal": goal,
                   "init_index": meta["index"], "entry_step": meta["step"],
                   "continuation_policy": policy_name, "noise_seed": seed, "rollout": name}
            rollout_started = time.monotonic()
            try:
                obs, restore = _restore_snapshot(env, scene_xml, arrays, meta["metadata"], meta["seed"])
                actor.reset()
                torch.manual_seed(seed); np.random.seed(seed % 2**32); random.seed(seed)
                initial = events(env, predicates)
                if any(initial.values()):
                    raise ValueError("Sealed entrance already satisfies a goal")
                primary_events, native_events = [initial], [initial]
                write_json(folder / "initial.json", {"events": initial, "observations": observation_fields(obs),
                    "physical": physical_fields(env, args.object), "restore_check": restore,
                    "rgb_frame": 0, "raw_rgb_orientation": "native sensor orientation; policy flips both spatial axes"})
                for key in image_frames:
                    image_frames[key].append(np.asarray(obs[key], dtype=np.uint8).copy())
                with SharedNoise(actor, seed) as noise, (folder / "trace.jsonl").open("w", encoding="utf-8", newline="\n") as trace:
                    for step in range(120):
                        before = observation_fields(obs)
                        physical_before = physical_fields(env, args.object)
                        calls_before = len(noise.records)
                        action = actor.action(obs, tasks[goal].language)
                        executed = np.clip(action, -1, 1)
                        obs, _, _, _ = env.step(executed)
                        native = events(env, predicates)
                        native_events.append(native)
                        refreshed = None
                        if (step + 1) % action_steps == 0 and step + 1 < 120:
                            obs = _fresh_observation(env)
                            refreshed = events(env, predicates)
                        primary = refreshed if refreshed is not None else native
                        primary_events.append(primary)
                        for key in image_frames:
                            image_frames[key].append(np.asarray(obs[key], dtype=np.uint8).copy())
                        position = np.asarray(obs[f"{args.object}_pos"])
                        record = {"step": step + 1, "input_obs_frame_index": step,
                            "replanning_chunk": step // action_steps,
                            "new_noise_chunk": noise.records[-1] if len(noise.records) > calls_before else None,
                            "observation_before": before, "physical_before": physical_before,
                            "action_unclipped": action.tolist(), "action_executed": executed.tolist(),
                            "clipped_dimensions": (action != executed).tolist(),
                            "native_post_step_predicates": native, "boundary_refresh_predicates": refreshed,
                            "primary_predicates_after": primary,
                            "observation_after": observation_fields(obs), "physical_after": physical_fields(env, args.object),
                            "book_relative_to_eef": relative_pose(obs, args.object),
                            "target_xy_distances_m": {g: float(np.linalg.norm(position[:2] - np.asarray(c)[:2])) for g, c in centres.items()},
                            "contacts_after": contact_records(env)}
                        trace.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        if (step + 1) % action_steps == 0:
                            trace.flush()
                            resource_check(args.output, started, args.max_minutes, args.max_output_bytes)
                    expected_chunks = 120 // action_steps
                    if len(noise.records) != expected_chunks:
                        raise RuntimeError(f"Expected {expected_chunks} Gaussian chunks, got {len(noise.records)}")
                    schedule = [record["sha256"] for record in noise.records]
                    previous = manifest["noise_schedules"].setdefault(str(seed), schedule)
                    if schedule != previous:
                        raise RuntimeError("Common-noise schedule differs across a source/state/model")
                    row.update(status="complete", restore_check=restore, steps=120,
                        noise_chunks=noise.records, native_post_step_summary=summarize_events(goal, native_events),
                        **summarize_events(goal, primary_events))
            except Exception as error:
                row.update(status="failed", error=repr(error))
                raise
            finally:
                # These are policy-as-seen frames, not rerenders from another state.
                if image_frames["agentview_image"]:
                    np.savez_compressed(folder / "observations.npz", **{
                        key: np.stack(value) for key, value in image_frames.items()})
                row["seconds"] = time.monotonic() - rollout_started
                for filename in ("trace.jsonl", "initial.json", "observations.npz"):
                    path = folder / filename
                    if path.exists():
                        row.setdefault("files", {})[filename] = {"bytes": path.stat().st_size, "sha256": _sha(path)}
                write_json(folder / "result.json", row)
                manifest["rows"].append(row)
                manifest["seconds"] = time.monotonic() - started
                write_json(args.output / "manifest.json", manifest)
                env.close()
            print(json.dumps({"completed": len(manifest["rows"]), "planned": len(jobs),
                              **{k: row[k] for k in ("snapshot_id", "continuation_policy", "noise_seed", "correct_ever")}}), flush=True)
        manifest["summary"] = summarize_rows(manifest["rows"], selected, args.seeds)
        manifest.update(status="complete", seconds=time.monotonic() - started,
                        finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "COMPLETE.json", {"status": "complete", "continuations": len(jobs),
            "distinct_snapshots": len(selected), "seconds": manifest["seconds"]})
    except Exception as error:
        manifest.update(status="failed", error=repr(error), seconds=time.monotonic() - started)
        write_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--runtime-scripts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026102801, 2026102802, 2026102803])
    parser.add_argument("--snapshot-ids", nargs="+")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--object", default="black_book_1")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--max-output-bytes", type=int, default=1610612736)
    main(parser.parse_args())
