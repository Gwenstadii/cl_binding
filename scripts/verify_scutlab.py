"""Check the local environment without fetching model weights or datasets."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".server"
STATE.mkdir(exist_ok=True)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

# The user's legacy ~/.libero points at a removed installation. Isolate ours.
libero_spec = importlib.util.find_spec("libero")
if libero_spec is None or not libero_spec.submodule_search_locations:
    raise RuntimeError("The selected LIBERO runtime is not installed")
libero_outer = Path(next(iter(libero_spec.submodule_search_locations)))
libero_root = libero_outer / "libero"
config_dir = STATE / "libero"
config_dir.mkdir(exist_ok=True)
os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
config_paths = {
    "benchmark_root": str(libero_root),
    "bddl_files": str(libero_root / "bddl_files"),
    "init_states": str(libero_root / "init_files"),
    "assets": str(libero_root / "assets"),
    "datasets": str(ROOT / "datasets" / "raw"),
}
import yaml

(config_dir / "config.yaml").write_text(yaml.safe_dump(config_paths), encoding="utf-8")

import torch
import torchvision
import av
import lerobot
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero.libero import benchmark

versions = {}
for distribution in ["lerobot", "torch", "torchvision", "torchcodec", "transformers", "accelerate", "huggingface-hub", "hf-libero", "mujoco", "robosuite", "av", "h5py"]:
    versions[distribution] = importlib.metadata.version(distribution)

devices = []
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        value = (torch.ones(8, device=f"cuda:{index}") * 2).sum().item()
    if value != 16:
        raise RuntimeError(f"CUDA numerical check failed on device {index}")
    devices.append({"index": index, "name": torch.cuda.get_device_name(index)})

suite = benchmark.get_benchmark_dict()["libero_90"]()
tasks = []
for goal in ["front", "left", "right", "back"]:
    stem = f"STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_{goal}_compartment_of_the_caddy"
    matches = [(index, task) for index, task in enumerate(suite.tasks) if task.name == stem]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one registered task for {stem}, got {len(matches)}")
    index, task = matches[0]
    definition = Path(config_paths["bddl_files"]) / task.problem_folder / task.bddl_file
    initial_state = Path(config_paths["init_states"]) / task.problem_folder / task.init_states_file
    tasks.append({"goal": goal, "task_id": index, "definition_exists": definition.is_file(), "initial_state_exists": initial_state.is_file()})

try:
    from torchcodec.decoders import VideoDecoder
    torchcodec_status = "import_ok"
except Exception as error:
    torchcodec_status = f"unavailable: {type(error).__name__}: {error}"

report = {
    "python": sys.version,
    "versions": versions,
    "devices": devices,
    "lerobot_import": str(Path(lerobot.__file__).resolve()),
    "libero_paths": config_paths,
    "task_checks": tasks,
    "pyav_import": "ok",
    "torchcodec": torchcodec_status,
    "model_weights_downloaded_by_check": False,
    "smolvla_import": SmolVLAPolicy.__name__,
}
(STATE / "environment_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
if not all(task["definition_exists"] and task["initial_state_exists"] for task in tasks):
    raise RuntimeError("LIBERO task definitions or initial states are incomplete; see environment_check.json")
if not Path(config_paths["assets"]).is_dir():
    raise RuntimeError("The installed LIBERO asset directory is missing")
