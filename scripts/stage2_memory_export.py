"""Export the self-contained ordinary-ER memory: uniform left/front training frames under a byte budget.

The memory is one uncompressed .npz plus a manifest; every byte of both files is charged. ER training reads
only this directory, never the full old videos. Verification reloads the exported arrays and compares them with
the training dataset exactly as the old model saw it (decoded video frames, 50-step chunk, padding mask).
"""

import argparse
import hashlib
import json
from pathlib import Path

from stage1_smoke import ROOT, CAMERAS, dump
from stage2_protocol import OLD_GOALS, check_budget, memory_record_bytes, nested_subset, select_memory

ARRAYS = ("image", "image2", "state", "action", "action_is_pad", "goal_index", "frame_index", "episode_index")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_groups(conversion):
    groups = {goal: [] for goal in OLD_GOALS}
    offset = 0
    for episode in conversion["episodes"]:
        groups[episode["goal"]].extend(range(offset, offset + episode["frames"]))
        offset += episode["frames"]
    return groups, offset


def export(config, output):
    import numpy as np
    from lerobot.datasets import LeRobotDataset

    m = config["memory"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite memory: {output}")
    source = ROOT / m["source_dataset"]
    conversion = json.loads((source / "r0_conversion.json").read_text())
    assert conversion["spec"]["split"] == "train" and tuple(conversion["spec"]["goals"]) == OLD_GOALS
    assert conversion["spec"]["action_alignment"] == config["data"]["action_alignment"]
    groups, total = old_groups(conversion)
    chunk = config["model"]["chunk_size"]
    dataset = LeRobotDataset("local/cl_binding_r0_train", root=source, video_backend="pyav",
                             delta_timestamps={"action": [i / config["data"]["fps"] for i in range(chunk)]})
    assert total == len(dataset)
    per_goal = m["per_goal"]
    estimate = 2 * per_goal * memory_record_bytes((3, 128, 128), len(CAMERAS), chunk, 7, 8)
    if estimate > m["budget_bytes"]:
        raise ValueError(f"Raw payload {estimate} already exceeds the budget; lower per_goal before export")
    selected = select_memory(groups, per_goal, config["seeds"]["memory"])
    instructions = {e["goal"]: e["instruction"] for e in conversion["episodes"]}
    rows = {key: [] for key in ARRAYS}
    for goal_index, goal in enumerate(OLD_GOALS):
        for index in selected[goal]:
            sample = dataset[index]
            assert sample["task"] == instructions[goal]
            for key, name in (("image", "observation.images.image"), ("image2", "observation.images.image2")):
                pixels = (sample[name].numpy() * 255).round()
                assert pixels.min() >= 0 and pixels.max() <= 255
                rows[key].append(pixels.astype(np.uint8))
            rows["state"].append(sample["observation.state"].numpy().astype(np.float32))
            rows["action"].append(sample["action"].numpy().astype(np.float32))
            rows["action_is_pad"].append(sample["action_is_pad"].numpy().astype(bool))
            rows["goal_index"].append(goal_index)
            rows["frame_index"].append(int(sample["frame_index"]))
            rows["episode_index"].append(int(sample["episode_index"]))
    arrays = {key: np.stack(value) if key not in ("goal_index", "frame_index", "episode_index")
              else np.asarray(value, dtype=np.int32) for key, value in rows.items()}
    output.mkdir(parents=True)
    np.savez(output / "memory.npz", **arrays)
    episodes = [{"goal": e["goal"], "demo": e["demo"]} for e in conversion["episodes"]]
    manifest = {"purpose": "stage2a_ordinary_er_self_contained_memory", "source_dataset": m["source_dataset"],
                "source_split_sha256": conversion["spec"]["split_sha256"], "goals": list(OLD_GOALS),
                "instructions": instructions, "per_goal": per_goal, "selection": "uniform_without_replacement_per_goal",
                "memory_seed": config["seeds"]["memory"], "dataset_indices": selected,
                "records": [{"goal": OLD_GOALS[g], "dataset_index": i, "demo": episodes[e]["demo"], "frame_index": f}
                            for g, i, e, f in zip(arrays["goal_index"].tolist(), selected["left"] + selected["front"],
                                                  arrays["episode_index"].tolist(), arrays["frame_index"].tolist(),
                                                  strict=True)],
                "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
                "memory_npz_sha256": sha256(output / "memory.npz")}
    dump(output / "memory_manifest.json", manifest)
    # Charge both files; the manifest is written before measuring so the reported total is final.
    budget = check_budget({p.name: p.stat().st_size for p in output.iterdir()}, m["budget_bytes"])
    if not budget["within_budget"]:
        raise ValueError(f"Memory exceeds budget: {budget}")
    verify(config, output, dataset, budget)


def export_subset(config, output):
    """Smaller self-contained memory made only from a parent memory's records (no access to old videos)."""
    import numpy as np
    m = config["memory"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite memory: {output}")
    parent_dir = ROOT / m["subset_of"]
    arrays, parent_manifest, _ = load_memory(parent_dir, m["budget_bytes"])
    positions = {goal: [int(i) for i in np.flatnonzero(arrays["goal_index"] == g)] for g, goal in enumerate(OLD_GOALS)}
    chosen = nested_subset(positions, m["per_goal"], m["subset_seed"])
    order = chosen["left"] + chosen["front"]
    output.mkdir(parents=True)
    np.savez(output / "memory.npz", **{key: value[order] for key, value in arrays.items()})
    records = [parent_manifest["records"][i] for i in order]
    manifest = {"purpose": "stage2a_er_budget_sweep_nested_subset", "source_dataset": parent_manifest["source_dataset"],
                "source_split_sha256": parent_manifest["source_split_sha256"], "goals": list(OLD_GOALS),
                "instructions": parent_manifest["instructions"], "per_goal": m["per_goal"],
                "selection": "prefix_of_fixed_per_goal_permutation_of_parent_memory", "subset_seed": m["subset_seed"],
                "parent_memory": m["subset_of"], "parent_memory_npz_sha256": parent_manifest["memory_npz_sha256"],
                "parent_positions": chosen, "records": records,
                "dataset_indices": {goal: [r["dataset_index"] for r in records if r["goal"] == goal] for goal in OLD_GOALS},
                "memory_npz_sha256": sha256(output / "memory.npz")}
    dump(output / "memory_manifest.json", manifest)
    budget = check_budget({p.name: p.stat().st_size for p in output.iterdir()}, m["budget_bytes"])
    reloaded, _, _ = load_memory(output, m["budget_bytes"])
    exact = all(np.array_equal(reloaded[k], arrays[k][order]) for k in ARRAYS)
    report = {"status": "pass" if exact and budget["within_budget"] else "fail", "records": len(order),
              "records_per_goal": {goal: len(chosen[goal]) for goal in OLD_GOALS}, "budget": budget,
              "exact_copy_of_parent_records": exact, "parent_memory": m["subset_of"]}
    dump(output.parent / f"{output.name}_verification.json", report)
    print(json.dumps(report), flush=True)
    if report["status"] != "pass":
        raise RuntimeError("Subset memory failed verification")


def load_memory(directory, budget_bytes):
    """Read only the memory directory; refuse anything over budget or not matching its manifest."""
    import numpy as np
    directory = Path(directory)
    manifest = json.loads((directory / "memory_manifest.json").read_text())
    if sha256(directory / "memory.npz") != manifest["memory_npz_sha256"]:
        raise ValueError("Memory payload does not match its manifest")
    budget = check_budget({p.name: p.stat().st_size for p in directory.iterdir() if p.name in
                           ("memory.npz", "memory_manifest.json")}, budget_bytes)
    if not budget["within_budget"]:
        raise ValueError(f"Memory exceeds budget: {budget}")
    with np.load(directory / "memory.npz") as data:
        arrays = {key: data[key] for key in ARRAYS}
    return arrays, manifest, budget


def memory_sample(arrays, manifest, index):
    """One record in the exact raw-sample layout produced by LeRobotDataset for training."""
    import torch
    goal = OLD_GOALS[int(arrays["goal_index"][index])]
    return {"observation.images.image": torch.from_numpy(arrays["image"][index]).float() / 255,
            "observation.images.image2": torch.from_numpy(arrays["image2"][index]).float() / 255,
            "observation.state": torch.from_numpy(arrays["state"][index]),
            "action": torch.from_numpy(arrays["action"][index]),
            "action_is_pad": torch.from_numpy(arrays["action_is_pad"][index]),
            "task": manifest["instructions"][goal]}


def verify(config, output, dataset, budget):
    import numpy as np
    arrays, manifest, reloaded_budget = load_memory(output, config["memory"]["budget_bytes"])
    order = manifest["dataset_indices"]["left"] + manifest["dataset_indices"]["front"]
    worst = {"image": 0.0, "image2": 0.0, "state": 0.0, "action": 0.0}
    mismatched_masks = 0
    for position, index in enumerate(order):
        expected = dataset[index]
        restored = memory_sample(arrays, manifest, position)
        for key, name in (("image", "observation.images.image"), ("image2", "observation.images.image2"),
                          ("state", "observation.state"), ("action", "action")):
            worst[key] = max(worst[key], float((restored[name] - expected[name]).abs().max()))
        mismatched_masks += int(not bool((restored["action_is_pad"] == expected["action_is_pad"]).all()))
        assert restored["task"] == expected["task"]
    passed = all(v < 1e-6 for v in worst.values()) and mismatched_masks == 0
    per_goal = {goal: int((arrays["goal_index"] == i).sum()) for i, goal in enumerate(OLD_GOALS)}
    episodes = {goal: len({int(e) for e, g in zip(arrays["episode_index"], arrays["goal_index"]) if OLD_GOALS[g] == goal})
                for goal in OLD_GOALS}
    report = {"status": "pass" if passed else "fail", "records": len(order), "records_per_goal": per_goal,
              "distinct_source_episodes_per_goal": episodes, "max_abs_difference_vs_training_dataset": worst,
              "mismatched_padding_masks": mismatched_masks, "budget": reloaded_budget,
              "padded_chunk_records": int(np.asarray(arrays["action_is_pad"]).any(axis=1).sum()),
              "reads_full_old_videos_during_training": False,
              "scope": "Verification is outside the memory directory and is not charged; ER reads memory only."}
    dump(output.parent / f"{output.name}_verification.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("Exported memory differs from the training dataset")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/stage2a_replay_v1.json")
    p.add_argument("--output", type=Path, help="Defaults to config memory.output")
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    (export_subset if c["memory"].get("subset_of") else export)(c, a.output or ROOT / c["memory"]["output"])
