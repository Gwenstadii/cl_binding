"""Offline, local-only LIBERO/SmolVLA engineering smoke checks (not research training)."""

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".server" / "libero")

from download_stage1 import FILENAMES, GOALS, complete_model, find_demo, find_model

CAMERAS = {"observation.images.image": "agentview_rgb", "observation.images.image2": "eye_in_hand_rgb"}
REPO_ID = "local/cl_binding_smoke"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


DEFAULT_SCENE = "STUDY_SCENE2"


def task_name(goal, scene=DEFAULT_SCENE):
    return f"{scene}_pick_up_the_book_and_place_it_in_the_{goal}_compartment_of_the_caddy"


def sources(goals=GOALS, scene=DEFAULT_SCENE):
    result = {goal: find_demo(f"{task_name(goal, scene)}_demo.hdf5") for goal in GOALS if goal in goals}
    missing = [goal for goal, path in result.items() if path is None]
    if missing:
        raise RuntimeError(f"Missing raw demonstrations: {missing}. Finish download_stage1.py --data first.")
    return result


def inspect_raw():
    import h5py
    import numpy as np

    report = {}
    for goal, path in sources().items():
        with h5py.File(path, "r") as stream:
            data = stream["data"]
            demos = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))
            demo = data[demos[0]]
            obs = demo["obs"]
            actions = demo["actions"][:]
            state = np.concatenate((obs["ee_states"][:], obs["gripper_states"][:]), axis=1)
            if actions.shape != (len(state), 7) or state.shape[1] != 8:
                raise RuntimeError(f"Unexpected state/action shape in {path}")
            if not np.isfinite(actions).all() or not np.isfinite(state).all():
                raise RuntimeError(f"Non-finite data in {path}")
            info = json.loads(data.attrs["problem_info"])
            language = info["language_instruction"]
            if f"{goal} compartment" not in language:
                raise RuntimeError(f"Instruction/file mismatch: {goal}: {language}")
            for key in CAMERAS.values():
                if obs[key].shape[0] != len(actions) or obs[key].shape[-1] != 3 or obs[key].dtype != np.uint8:
                    raise RuntimeError(f"Invalid image array: {path}:{key}")
            report[goal] = {
                "file": str(path), "bytes": path.stat().st_size, "demos": len(demos),
                "first_demo": demos[0], "frames": len(actions), "instruction": language,
                "state_shape": list(state.shape), "action_shape": list(actions.shape),
                "image_shape": list(obs["agentview_rgb"].shape),
                "action_min": actions.min(axis=0).tolist(), "action_max": actions.max(axis=0).tolist(),
            }
    dump(ROOT / ".server" / "smoke" / "raw_inspection.json", report)
    print(json.dumps(report, indent=2))


def convert(args):
    import h5py
    import numpy as np
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset
    from PIL import Image

    raw = sources(args.goals)
    output = local_path(args.output)
    spec = {"goals": args.goals, "episodes_per_task": args.episodes_per_task, "fps": 20,
            "image_transform": "rotate_180", "state": "ee_states(6)+gripper_states(2)",
            "actions": "original_OSC_POSE_relative_7D", "sources": {goal: str(raw[goal]) for goal in args.goals}}
    marker = output / "smoke_conversion.json"
    if output.exists():
        if marker.is_file():
            previous = json.loads(marker.read_text())
            if previous.get("spec") == spec and previous.get("status") == "complete":
                print(f"REUSE completed smoke dataset: {output}")
                return
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new --output for a fresh conversion")
    with h5py.File(raw[args.goals[0]], "r") as stream:
        first = stream["data"][sorted(stream["data"].keys())[0]]
        height, width, _ = first["obs"]["agentview_rgb"].shape[1:]
    features = {key: {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]} for key in CAMERAS}
    features.update({"observation.state": {"dtype": "float32", "shape": (8,), "names": None},
                     "action": {"dtype": "float32", "shape": (7,), "names": None}})
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID, root=output, fps=20, robot_type="panda", features=features,
        use_videos=True, video_backend="pyav", image_writer_threads=4, encoder_threads=2,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="fast", g=2),
    )
    episodes = []
    try:
        for goal in args.goals:
            with h5py.File(raw[goal], "r") as stream:
                data = stream["data"]
                demos = sorted(data.keys(), key=lambda name: int(name.split("_")[-1]))[:args.episodes_per_task]
                if len(demos) != args.episodes_per_task:
                    raise RuntimeError(f"Too few demos for {goal}")
                language = json.loads(data.attrs["problem_info"])["language_instruction"]
                for name in demos:
                    demo = data[name]
                    obs = demo["obs"]
                    actions = demo["actions"][:].astype(np.float32)
                    state = np.concatenate((obs["ee_states"][:], obs["gripper_states"][:]), axis=1).astype(np.float32)
                    for index, action in enumerate(actions):
                        frame = {key: np.ascontiguousarray(obs[source][index][::-1, ::-1]) for key, source in CAMERAS.items()}
                        if index == 0:
                            for key, image in frame.items():
                                preview = ROOT / ".server" / "smoke" / "previews" / f"{goal}_{key.split('.')[-1]}.png"
                                preview.parent.mkdir(parents=True, exist_ok=True)
                                Image.fromarray(image).save(preview)
                        frame.update({"observation.state": state[index], "action": action, "task": language})
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    episodes.append({"goal": goal, "demo": name, "frames": len(actions), "instruction": language})
                    print(json.dumps(episodes[-1]), flush=True)
    finally:
        dataset.finalize()
    report = {"status": "complete", "spec": spec, "episodes": episodes,
              "purpose": "engineering_smoke_only_not_R0_or_success_rate_estimation"}
    dump(marker, report)
    print(f"CONVERSION PASS: {output}")


def metadata_path():
    record = json.loads((ROOT / ".server" / "resources.json").read_text())
    path = Path(record["HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]["metadata_path"])
    if not (path / "config.json").is_file():
        raise RuntimeError("Missing VLM metadata; finish download_stage1.py --models")
    return path


def load_policy(checkpoint=None, image_shape=(3, 128, 128)):
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = local_path(checkpoint) if checkpoint else find_model("lerobot/smolvla_base")
    if path is None or not complete_model(path):
        raise RuntimeError("Missing/incomplete policy checkpoint")
    config = SmolVLAConfig.from_pretrained(path, local_files_only=True)
    config.vlm_model_name = str(metadata_path())
    config.load_vlm_weights = False
    config.push_to_hub = False
    config.device = "cuda"
    config.use_amp = True
    config.n_action_steps = 1
    # Physical dimensions are padded internally to unchanged 32D projections.
    config.input_features = {key: PolicyFeature(type=FeatureType.VISUAL, shape=image_shape) for key in CAMERAS}
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    config.pretrained_path = path
    policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True, local_files_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    return policy


def check_load():
    import torch

    torch.set_num_threads(4)
    policy = load_policy()
    report = {"status": "pass", "strict_loading": True, "checkpoint": str(find_model("lerobot/smolvla_base")),
              "total_parameters": sum(value.numel() for value in policy.parameters()),
              "trainable_parameters": sum(value.numel() for value in policy.parameters() if value.requires_grad),
              "gpu": torch.cuda.get_device_name(), "allocated_mib": torch.cuda.memory_allocated() / 2**20,
              "vlm_initialization": "config_only_then_full_policy_checkpoint_strictly_loaded",
              "inputs": "2 RGB cameras + 8D state", "outputs": "7D action", "max_projection_dims": [32, 32]}
    dump(ROOT / ".server" / "smoke" / "model_load.json", report)
    print(json.dumps(report, indent=2))


def train(args):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors

    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    run = local_path(args.run_dir)
    if run.exists():
        raise FileExistsError(f"Choose a new --run-dir; refusing to overwrite {run}")
    dataset_root = local_path(args.dataset)
    conversion = json.loads((dataset_root / "smoke_conversion.json").read_text())
    if conversion["spec"]["goals"] != ["front", "left", "right"]:
        raise RuntimeError("This smoke route trains on three old goals only; back/N1 must stay separate")
    meta = LeRobotDatasetMetadata(REPO_ID, root=dataset_root)
    image_shape = tuple(meta.features[next(iter(CAMERAS))]["shape"])
    # Dataset features are HWC; policy features are CHW.
    if image_shape[-1] == 3:
        image_shape = (3, *image_shape[:2])
    policy = load_policy(image_shape=image_shape)
    dataset = LeRobotDataset(REPO_ID, root=dataset_root, video_backend="pyav",
                             delta_timestamps={"action": [i / meta.fps for i in range(policy.config.chunk_size)]})
    preprocess, postprocess = make_pre_post_processors(policy.config, dataset_stats=meta.stats)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    batch_iter = iter(loader)
    # Record decoder results separately from model-input normalization.
    raw_batch = next(batch_iter)
    decoded = {key: {"shape": list(raw_batch[key].shape), "std": float(raw_batch[key].std())} for key in CAMERAS}
    if not all(value["std"] > 0 for value in decoded.values()):
        raise RuntimeError("Decoded an empty/constant camera image")
    optimizer = torch.optim.AdamW([value for value in policy.parameters() if value.requires_grad], lr=1e-5)
    tracked = policy.model.action_out_proj.weight
    before = tracked.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    logs = []
    start = time.monotonic()
    policy.train()
    for step in range(args.steps):
        batch = preprocess(copy.deepcopy(raw_batch))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([value for value in policy.parameters() if value.requires_grad], 10.0, error_if_nonfinite=True)
        if norm.item() <= 0:
            raise RuntimeError("No nonzero trainable gradient")
        optimizer.step()
        logs.append({"step": step + 1, "loss": loss.item(), "gradient_norm": norm.item()})
        print(json.dumps(logs[-1]), flush=True)
        if step + 1 < args.steps:
            try:
                raw_batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                raw_batch = next(batch_iter)
    change = (tracked.detach() - before).abs().max().item()
    if change == 0 or not np.isfinite(change):
        raise RuntimeError("Action projection did not update correctly")
    policy.eval()
    policy.reset()
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")
    reference_batch = preprocess(copy.deepcopy(raw_batch))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = policy.predict_action_chunk(reference_batch, noise=noise).float().cpu()
    checkpoint = run / "checkpoint"
    policy.save_pretrained(checkpoint)
    preprocess.save_pretrained(checkpoint)
    postprocess.save_pretrained(checkpoint)
    peak = torch.cuda.max_memory_allocated() / 2**20
    del tracked, before, optimizer, policy, preprocess, postprocess
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_policy(checkpoint, image_shape=image_shape)
    pre2, post2 = make_pre_post_processors(restored.config, pretrained_path=checkpoint)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = restored.predict_action_chunk(pre2(copy.deepcopy(raw_batch)), noise=noise)
        physical = post2(actual).float().cpu()
    difference = (expected - actual.float().cpu()).abs().max().item()
    if not torch.allclose(expected, actual.float().cpu(), atol=1e-3, rtol=1e-3):
        raise RuntimeError(f"Save/reload changed fixed-noise predictions: {difference}")
    if list(physical.shape) != [1, restored.config.chunk_size, 7] or not torch.isfinite(physical).all():
        raise RuntimeError("Invalid denormalized action chunk")
    report = {"status": "pass", "purpose": "engineering_smoke_not_R0", "steps": logs,
              "decoded_cameras": decoded, "updated_projection_max_delta": change,
              "save_reload_max_difference": difference, "action_shape": list(physical.shape),
              "peak_training_allocated_mib": peak, "seconds": time.monotonic() - start,
              "checkpoint": str(checkpoint), "normalization_source": str(dataset_root)}
    dump(run / "train_smoke.json", report)
    print(json.dumps(report, indent=2))


def rollout(args):
    import numpy as np
    import torch
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.transform_utils import quat2axisangle
    from lerobot.policies import make_pre_post_processors

    torch.set_num_threads(4)
    torch.manual_seed(0)
    checkpoint = local_path(args.checkpoint)
    output = local_path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output; refusing to overwrite {output}")
    policy = load_policy(checkpoint)
    preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    report = {"purpose": "engineering_smoke_not_task_success_evaluation", "tasks": []}
    for goal in GOALS:
        task_id = next(index for index, task in enumerate(suite.tasks) if task.name == task_name(goal))
        task = suite.tasks[task_id]
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                                 camera_heights=128, camera_widths=128)
        actions = []
        try:
            env.seed(0)
            env.reset()
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file, weights_only=False)
            obs = env.set_init_state(states[0])
            for _ in range(10):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            policy.reset()
            for step in range(args.steps):
                batch = {"observation.state": torch.from_numpy(np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])).astype(np.float32)),
                         "task": task.language}
                for key, raw_key in zip(CAMERAS, ("agentview_image", "robot0_eye_in_hand_image"), strict=True):
                    image = np.ascontiguousarray(obs[raw_key][::-1, ::-1])
                    batch[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
                    if step in (0, args.steps - 1):
                        image_path = output / f"{goal}_step{step}_{key.split('.')[-1]}.png"
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(image).save(image_path)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    action = postprocess(policy.select_action(preprocess(batch))).float().cpu().numpy().reshape(-1)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise RuntimeError("Invalid simulation action")
                clipped = np.clip(action, -1, 1)
                actions.append({"raw": action.tolist(), "executed": clipped.tolist()})
                obs, _, done, _ = env.step(clipped)
                if done:
                    break
            result = {"goal": goal, "task_id": task_id, "status": "pass", "policy_steps": len(actions),
                      "task_success": bool(env.check_success()), "actions": actions}
            report["tasks"].append(result)
            dump(output / "rollout_smoke.json", report)
            print(json.dumps({key: value for key, value in result.items() if key != "actions"}), flush=True)
        finally:
            env.close()
    report["status"] = "pass"
    dump(output / "rollout_smoke.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect")
    commands.add_parser("load")
    conversion = commands.add_parser("convert")
    conversion.add_argument("--goals", nargs="+", choices=GOALS, required=True)
    conversion.add_argument("--episodes-per-task", type=int, default=1)
    conversion.add_argument("--output", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", default="datasets/stage1_smoke_old")
    training.add_argument("--run-dir", default="outputs/smoke/train_001")
    training.add_argument("--steps", type=int, default=5)
    evaluation = commands.add_parser("rollout")
    evaluation.add_argument("--checkpoint", default="outputs/smoke/train_001/checkpoint")
    evaluation.add_argument("--output", default="outputs/smoke/rollout_001")
    evaluation.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if getattr(args, "steps", 1) < 1 or getattr(args, "episodes_per_task", 1) < 1:
        parser.error("steps and episodes-per-task must be positive")
    {"inspect": lambda: inspect_raw(), "load": lambda: check_load(), "convert": lambda: convert(args),
     "train": lambda: train(args), "rollout": lambda: rollout(args)}[args.command]()


if __name__ == "__main__":
    main()
