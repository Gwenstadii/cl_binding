"""Offline parent queries for stage 2E (no training).

pseudo: for every frame of the NEW-goal training split, the sealed parent's action chunk under each old instruction
        (fixed per-frame noise, 10-step sampler, physical units) -> counterfactual pseudo-labels for CID.
score:  for every frame of the OLD training split, how much the parent's chunk changes between the two old
        instructions under identical noise (RMS in normalized units) -> decision-point scores for DPR memory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from stage1_smoke import ROOT, dump, load_policy
from stage2_protocol import OLD_GOALS, stream_seed

KEYS = ("observation.images.image", "observation.images.image2", "observation.state")


def main(args):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies import make_pre_post_processors

    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = ROOT / config["parent"]["checkpoint"]
    sha = hashlib.sha256((parent / "model.safetensors").read_bytes()).hexdigest()
    assert sha == config["parent"]["model_sha256"], "Parent weights changed"
    root = ROOT / (config["data"]["train_output"] if args.mode == "pseudo" else config["memory"]["source_dataset"])
    conversion = json.loads((root / "r0_conversion.json").read_text())
    instructions = {}
    for rel in (config["memory"]["source_dataset"],):
        for e in json.loads((ROOT / rel / "r0_conversion.json").read_text())["episodes"]:
            instructions[e["goal"]] = e["instruction"]
    dataset = LeRobotDataset("local/cl_binding_r0_train", root=root, video_backend="pyav")
    torch.set_num_threads(4)
    policy = load_policy(parent)
    policy.config.n_action_steps = config["model"]["n_action_steps"]
    policy.config.num_steps = config["model"]["flow_steps"]
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=parent)
    total = len(dataset) if not args.limit else args.limit
    chunk, max_dim = config["model"]["chunk_size"], policy.config.max_action_dim
    out = {goal: [] for goal in OLD_GOALS}
    raw_out = {goal: [] for goal in OLD_GOALS}
    for start in range(0, total, args.batch_size):
        ids = list(range(start, min(total, start + args.batch_size)))
        frames = dataset.__getitems__(ids)
        generator = torch.Generator().manual_seed(stream_seed(f"teacher-{args.mode}", config["seeds"]["memory"], start))
        noise = torch.randn(len(ids), chunk, max_dim, generator=generator).cuda()
        for goal in OLD_GOALS:
            raw = {k: torch.stack([f[k] for f in frames]) for k in KEYS}
            raw["task"] = [instructions[goal]] * len(ids)
            policy.reset()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = policy.predict_action_chunk(pre(copy.deepcopy(raw)), noise=noise).float()
                physical = post(normalized).float()
            raw_out[goal].append(normalized[..., :7].cpu().numpy())
            out[goal].append(physical.cpu().numpy())
        if start % (args.batch_size * 20) == 0:
            print(json.dumps({"mode": args.mode, "done": min(total, start + args.batch_size), "total": total}), flush=True)
    arrays = {goal: np.concatenate(v).astype(np.float32) for goal, v in out.items()}
    normalized = {goal: np.concatenate(v) for goal, v in raw_out.items()}
    args.output.mkdir(parents=True)
    if args.mode == "pseudo":
        np.savez(args.output / "pseudo_labels.npz", frame_index=np.arange(total, dtype=np.int32), **arrays)
    else:
        score = np.sqrt(((normalized["left"] - normalized["front"]) ** 2).mean(axis=(1, 2)))
        np.savez(args.output / "decision_scores.npz", frame_index=np.arange(total, dtype=np.int32), score=score.astype(np.float32))
    dump(args.output / "manifest.json", {"mode": args.mode, "parent": str(parent), "parent_model_sha256": sha,
         "dataset": str(root), "split_sha256": conversion["spec"]["split_sha256"], "frames": total,
         "instructions": instructions, "flow_steps": policy.config.num_steps, "noise": "stream_seed(teacher-<mode>, memory seed, batch start)",
         "scope": "offline parent queries; no old demonstrations are stored by pseudo mode"})
    print("TEACHER_LABELS_COMPLETE", args.mode, total, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mode", choices=["pseudo", "score"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int)
    main(p.parse_args())
