"""Audit full raw files, freeze trajectory splits, and convert old-only R0 data."""

import argparse
import hashlib
import json
from pathlib import Path

from stage1_protocol import observation_action_indices, split_trajectories
from stage1_smoke import CAMERAS, ROOT, dump, sources


def prepare(config, output):
    import h5py
    import numpy as np

    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    output.mkdir(parents=True)
    result = {"config": config, "files": {}, "splits": {}}
    for goal, path in sources().items():
        with h5py.File(path) as h:
            data = h["data"]
            names = sorted(data, key=lambda x: int(x.split("_")[-1]))
            language = json.loads(data.attrs["problem_info"])["language_instruction"]
            env_args = json.loads(data.attrs["env_args"])
            assert env_args["env_kwargs"]["control_freq"] == config["data"]["fps"]
            assert env_args["env_kwargs"]["controller_configs"]["type"] == "OSC_POSE"
            assert f"{goal} compartment" in language
            rows = []
            for name in names:
                d = data[name]
                actions = d["actions"][:]
                state = np.concatenate((d["obs/ee_states"][:], d["obs/gripper_states"][:]), axis=1)
                assert actions.shape == (len(state), 7) and state.shape[1] == 8
                assert np.isfinite(actions).all() and np.isfinite(state).all()
                for key in CAMERAS.values():
                    image = d[f"obs/{key}"]
                    assert image.shape == (len(state), 128, 128, 3) and image.dtype == np.uint8
                    for index in (0, len(state)//2, len(state)-1):
                        assert float(image[index].std()) > 0
                rows.append({"demo": name, "frames": len(state), "first_sim_state": d["states"][0].tolist(),
                             "last_reward": float(d["rewards"][-1])})
            result["files"][goal] = {"path": str(path), "bytes": path.stat().st_size,
                                     "instruction": language, "trajectories": rows}
            if goal in config["old_goals"]:
                result["splits"][goal] = split_trajectories(names, config["data"]["train_per_goal"], config["seed"])
        print(json.dumps({"goal": goal, "audited_trajectories": len(rows), "frames": sum(r["frames"] for r in rows)}), flush=True)
    encoded = json.dumps(result["splits"], sort_keys=True).encode()
    result["split_sha256"] = hashlib.sha256(encoded).hexdigest()
    dump(output / "audit.json", result)
    for split in ("train", "validation"):
        destination = ROOT / config["data"][f"{split}_output"]
        convert(result, split, destination)
    dump(output / "COMPLETE.json", {"status": "pass", "split_sha256": result["split_sha256"]})


def convert(audit, split, output):
    import h5py
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    spec = {"purpose": "R0_old_only", "split": split, "split_sha256": audit["split_sha256"],
            "goals": audit["config"]["old_goals"], "fps": audit["config"]["data"]["fps"], "image_transform": "rotate_180"}
    alignment = audit["config"]["data"].get("action_alignment", "same_index_legacy")
    observation_action_indices(2, alignment)  # Reject an unknown mode before creating a dataset.
    if alignment != "same_index_legacy":
        spec.update(action_alignment=alignment, observation_offset=0, action_offset=1,
                    terminal_observations_dropped_per_trajectory=1)
    marker = output / "r0_conversion.json"
    if output.exists():
        if marker.is_file() and json.loads(marker.read_text()).get("spec") == spec:
            print("REUSE", output, flush=True)
            return
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    features = {k: {"dtype": "video", "shape": (128,128,3), "names": ["height","width","channels"]} for k in CAMERAS}
    features.update({"observation.state": {"dtype":"float32","shape":(8,),"names":None},
                     "action": {"dtype":"float32","shape":(7,),"names":None}})
    dataset = LeRobotDataset.create(repo_id=f"local/cl_binding_r0_{split}", root=output, fps=spec["fps"],
            robot_type="panda", features=features, use_videos=True, video_backend="pyav",
            image_writer_threads=4, encoder_threads=2,
            rgb_encoder=RGBEncoderConfig(vcodec="h264",crf=18,preset="fast",g=2))
    episodes = []
    try:
        for goal in spec["goals"]:
            source = audit["files"][goal]
            with h5py.File(source["path"]) as h:
                for name in audit["splits"][goal][split]:
                    d=h["data"][name]; a=d["actions"][:].astype(np.float32)
                    state=np.concatenate((d["obs/ee_states"][:],d["obs/gripper_states"][:]),axis=1).astype(np.float32)
                    obs_indices, action_indices = observation_action_indices(len(a), alignment)
                    for i,j in zip(obs_indices, action_indices, strict=True):
                        frame={k:np.ascontiguousarray(d[f"obs/{raw}"][i][::-1,::-1]) for k,raw in CAMERAS.items()}
                        frame.update({"observation.state":state[i],"action":a[j],"task":source["instruction"]})
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    row={"goal":goal,"demo":name,"frames":len(obs_indices),"instruction":source["instruction"]}
                    if alignment != "same_index_legacy":row.update(source_frames=len(a),observation_offset=0,action_offset=1)
                    episodes.append(row)
                    print(json.dumps({"split":split,**row}),flush=True)
    finally:
        dataset.finalize()
    dump(marker,{"status":"complete","spec":spec,"episodes":episodes})


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=ROOT/"configs/stage1_r0.json")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    prepare(json.loads(a.config.read_text()), a.output)
