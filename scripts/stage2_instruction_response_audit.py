"""Fixed-input instruction-response audit across the dual-goal sequential chain (forward passes only).

For validation frames of all four goals at several trajectory phases, query every checkpoint with each of the
four instructions under identical images, state and flow noise. Outputs are normalized action chunks from
(a) one Euler step (an affine image of the t=1 velocity) and (b) the full 10-step sampler.

Reported per checkpoint and phase:
- pairwise instruction separation  D(ci,cj) = mean_frames RMS(A(ci) - A(cj))
- old-condition drift decomposition over C={left,front} against the parent theta0:
      mean_c ||A(c)-A0(c)||^2 = ||mu-mu0||^2 + mean_c ||delta(c)-delta0(c)||^2
- nearest-reference goal assignment of A(c) among references
      left/front <- parent, right <- E007 endpoint, back <- E009 endpoint
- cosine between predicted cumulative end-effector displacement and each reference's displacement.
No training, no simulator, no checkpoint is modified.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from stage1_smoke import load_policy  # noqa: E402  (sets offline/LIBERO env before heavy imports)

import numpy as np  # noqa: E402
import torch  # noqa: E402

GOALS = ("left", "front", "right", "back")
CHECKPOINTS = {
    "parent_dual12k": "outputs/r0/dual_left_front_20261023_01/checkpoints/012000",
    **{f"right_{k}k": f"outputs/sequential/dual12000_then_right_20261023_01/checkpoints/{k:03d}000" for k in (3, 6, 9, 12)},
    **{f"back_{k}k": f"outputs/sequential/dual_right_then_back_20261023_01/checkpoints/{k:03d}000" for k in (3, 6, 9, 12)},
}
REFERENCE = {"left": "parent_dual12k", "front": "parent_dual12k", "right": "right_12k", "back": "back_12k"}
DATASETS = ("datasets/stage1_dual_left_front_validation_aligned_v1",
            "datasets/stage1_single_right_validation_aligned_v1",
            "datasets/stage1_single_back_validation_aligned_v1")


def collect_frames(phases):
    from lerobot.datasets import LeRobotDataset
    frames, instructions = [], {}
    for rel in DATASETS:
        root = REPO / rel
        conversion = json.loads((root / "r0_conversion.json").read_text())
        dataset = LeRobotDataset("local/cl_binding_r0_validation", root=root, video_backend="pyav")
        offset = 0
        for episode in conversion["episodes"]:
            instructions[episode["goal"]] = episode["instruction"]
            for phase in phases:
                index = offset + min(int(round(phase * (episode["frames"] - 1))), episode["frames"] - 1)
                sample = dataset[index]
                frames.append({"source_goal": episode["goal"], "demo": episode["demo"], "phase": phase,
                               "frame_in_episode": index - offset, "dataset": rel,
                               "observation.images.image": sample["observation.images.image"],
                               "observation.images.image2": sample["observation.images.image2"],
                               "observation.state": sample["observation.state"]})
            offset += episode["frames"]
        assert offset == len(dataset)
    assert set(instructions) == set(GOALS)
    return frames, instructions


def query(policy, pre, frames, instruction, noise, flow_steps, batch_size):
    policy.config.num_steps = flow_steps
    outputs = []
    for start in range(0, len(frames), batch_size):
        part = frames[start:start + batch_size]
        raw = {key: torch.stack([f[key] for f in part]) for key in
               ("observation.images.image", "observation.images.image2", "observation.state")}
        raw["task"] = [instruction] * len(part)
        batch = pre(copy.deepcopy(raw))
        policy.reset()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = policy.predict_action_chunk(batch, noise=noise[start:start + len(part)])
        outputs.append(chunk.float().cpu())
    return torch.cat(outputs).numpy()


def rms(x):
    return float(np.sqrt((x ** 2).mean()))


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    for item in args.extra or []:
        name, rel = item.split("=", 1)
        if name in CHECKPOINTS:
            raise ValueError(f"Duplicate checkpoint name {name}")
        CHECKPOINTS[name] = rel
    torch.set_num_threads(4)
    frames, instructions = collect_frames(args.phases)
    generator = torch.Generator().manual_seed(args.seed)
    chunk, max_dim = 50, 32
    noises = [torch.randn(len(frames), chunk, max_dim, generator=generator).cuda() for _ in range(args.noise_samples)]
    raw_outputs, identity, displacement = {}, {}, {}
    for name, rel in CHECKPOINTS.items():
        path = REPO / rel
        policy = load_policy(path)
        policy.eval()
        from lerobot.policies import make_pre_post_processors
        pre, post = make_pre_post_processors(policy.config, pretrained_path=path)
        identity[name] = {"path": str(path),
                          "model_sha256": hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest(),
                          "normalizer_sha256": hashlib.sha256(next(path.glob("*normalizer_processor.safetensors")).read_bytes()).hexdigest()}
        raw_outputs[name], displacement[name] = {}, {}
        for steps in args.flow_steps:
            for goal in GOALS:
                per_noise = [query(policy, pre, frames, instructions[goal], n, steps, args.batch_size) for n in noises]
                raw_outputs[name][(steps, goal)] = np.stack(per_noise)  # (noise, frames, 50, 7) normalized
                if steps == max(args.flow_steps):
                    physical = np.stack([post(torch.from_numpy(x).cuda()).float().cpu().numpy() for x in per_noise])
                    displacement[name][goal] = physical[..., :3].sum(axis=2)  # (noise, frames, 3)
        print(json.dumps({"checkpoint": name, **identity[name]}), flush=True)
        del policy, pre, post
        torch.cuda.empty_cache()

    phases = sorted(set(args.phases))
    phase_of = np.array([f["phase"] for f in frames])
    report = {"scope": "forward-only fixed-input audit; no training, simulator or checkpoint modification",
              "frames": [{k: f[k] for k in ("source_goal", "demo", "phase", "frame_in_episode", "dataset")} for f in frames],
              "instructions": instructions, "checkpoints": identity, "noise_samples": args.noise_samples,
              "seed": args.seed, "flow_steps": args.flow_steps, "references": REFERENCE, "results": {}}
    parent = "parent_dual12k"
    for steps in args.flow_steps:
        for name in CHECKPOINTS:
            out = raw_outputs[name]
            entry = {}
            for phase in ["all", *phases]:
                mask = np.ones(len(frames), bool) if phase == "all" else phase_of == phase
                A = {g: out[(steps, g)][:, mask] for g in GOALS}
                A0 = {g: raw_outputs[parent][(steps, g)][:, mask] for g in GOALS}
                sep = {f"{a}-{b}": rms(A[a] - A[b]) for i, a in enumerate(GOALS) for b in GOALS[i + 1:]}
                mu, mu0 = (A["left"] + A["front"]) / 2, (A0["left"] + A0["front"]) / 2
                d, d0 = (A["left"] - A["front"]) / 2, (A0["left"] - A0["front"]) / 2
                mean_drift = float(((mu - mu0) ** 2).mean())
                delta_drift = float(((d - d0) ** 2).mean())
                total = float(np.mean([((A[c] - A0[c]) ** 2).mean() for c in ("left", "front")]))
                assign = {}
                for g in GOALS:
                    refs = {r: raw_outputs[REFERENCE[r]][(steps, r)][:, mask] for r in GOALS}
                    dist = np.stack([((A[g] - refs[r]) ** 2).mean(axis=(2, 3)) for r in GOALS])  # (ref, noise, frames)
                    winner = dist.argmin(axis=0).reshape(-1)
                    assign[g] = {r: int((winner == i).sum()) for i, r in enumerate(GOALS)}
                row = {"separation_rms": sep,
                       "old_condition_drift": {"total": total, "mean_component": mean_drift, "delta_component": delta_drift,
                                               "parent_delta_power": float((d0 ** 2).mean()),
                                               "identity_check": total - mean_drift - delta_drift},
                       "nearest_reference_assignment": assign}
                if steps == max(args.flow_steps):
                    cos = {}
                    for g in GOALS:
                        v = displacement[name][g][:, mask].reshape(-1, 3)
                        cos[g] = {}
                        for r in GOALS:
                            ref = displacement[REFERENCE[r]][r][:, mask].reshape(-1, 3)
                            num = (v * ref).sum(1)
                            den = np.linalg.norm(v, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9
                            cos[g][r] = float((num / den).mean())
                    row["displacement_cosine_to_reference"] = cos
                entry[str(phase)] = row
            report["results"].setdefault(f"flow_steps_{steps}", {})[name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    np.savez_compressed(args.output.with_suffix(".npz"),
                        **{f"{n}__s{s}__{g}": v for n, d in raw_outputs.items() for (s, g), v in d.items()})
    for steps in args.flow_steps:
        print(f"== flow_steps={steps} (phase=all)")
        for name in CHECKPOINTS:
            r = report["results"][f"flow_steps_{steps}"][name]["all"]
            s, o = r["separation_rms"], r["old_condition_drift"]
            print(f"{name:15s} L-F {s['left-front']:.3f} L-R {s['left-right']:.3f} F-R {s['front-right']:.3f} "
                  f"L-B {s['left-back']:.3f} R-B {s['right-back']:.3f} | drift mu {o['mean_component']:.3f} "
                  f"delta {o['delta_component']:.3f} (parent delta {o['parent_delta_power']:.3f}) | "
                  f"L->{r['nearest_reference_assignment']['left']} F->{r['nearest_reference_assignment']['front']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phases", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--flow-steps", type=int, nargs="+", default=[1, 10])
    p.add_argument("--noise-samples", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--extra", nargs="+", help="Additional name=repo-relative-checkpoint entries, e.g. er_12k=outputs/...")
    main(p.parse_args())
