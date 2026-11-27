"""Compare the K18 replay against the historical Front training prefix."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


RUN = Path(__file__).resolve().parent
HISTORY = RUN.parent / "r1_then_front_v1/train"
PARENT = RUN.parent / "r1_left_right_20261104_01/train_r1/checkpoints/024000"
SAVED = (10, 30, 100, 300, 1000, 3000, 6000)
KEYS = ("step", "loss", "gradient_norm", "lr", "batch_indices", "samples_per_goal")


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def log_rows(root):
    return [json.loads(line) for line in (root / "train_steps.jsonl").read_text().splitlines()]


def identical(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    return bool(a == b)


def model_equal(a, b):
    with safe_open(a, framework="pt") as left, safe_open(b, framework="pt") as right:
        return left.keys() == right.keys() and all(identical(left.get_tensor(k), right.get_tensor(k)) for k in left.keys())


def main():
    torch.set_num_threads(2)
    replay = RUN / "train"
    a, b = log_rows(replay), log_rows(HISTORY)
    assert len(a) == 6000 and len(b) == 12000
    assert [r["step"] for r in a] == list(range(1, 6001))
    assert [r["step"] for r in b] == list(range(1, 12001))
    mismatches = {k: [x["step"] for x, y in zip(a, b) if x[k] != y[k]] for k in KEYS}
    checkpoints = [
        ("R1", PARENT),
        ("Front6000_historical", HISTORY / "checkpoints/006000"),
        ("Front12000_historical", HISTORY / "checkpoints/012000"),
        *((f"Front{step}_replay", replay / "checkpoints" / f"{step:06d}") for step in SAVED),
    ]
    digests = {name: file_digest(path / "model.safetensors") for name, path in checkpoints}
    out = replay / "checkpoints/006000"
    ref = HISTORY / "checkpoints/006000"
    model_same_hash = digests["Front6000_replay"] == digests["Front6000_historical"]
    left = torch.load(out / "training_state.pt", map_location="cpu", weights_only=False)
    right = torch.load(ref / "training_state.pt", map_location="cpu", weights_only=False)
    keys = ("step", "config", "optimizer", "scheduler", "torch_rng", "cuda_rng")
    match = {k: identical(left[k], right[k]) for k in keys}
    report = {
        "replay_steps": len(a),
        "historical_steps": len(b),
        "mismatches": mismatches,
        "gpu_step_time_sum_seconds": sum(r["step_seconds"] for r in a),
        "model_sha256": digests,
        "6000_model_sha256_equal": model_same_hash,
        "6000_model_tensors_equal": model_same_hash or model_equal(out / "model.safetensors", ref / "model.safetensors"),
        "6000_training_state_equal": match,
        "saved_checkpoint_markers": {
            name: json.loads((path / "CHECKPOINT_COMPLETE.json").read_text()) for name, path in checkpoints
        },
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
