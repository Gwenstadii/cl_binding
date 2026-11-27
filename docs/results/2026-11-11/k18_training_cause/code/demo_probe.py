"""Fixed validation-demo inputs and teacher-forced action drafts."""
import copy
import json

from stage1_smoke import ROOT

import torch
from lerobot.datasets import LeRobotDataset
from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs

from measurement import RUN, Lab5, INSTR, BLOCKS, STEPS, checkpoint, load


def select_records():
    roots = ["stage1_dual_left_right_validation_aligned_v1", "stage1_single_front_validation_aligned_v1"]
    for name in roots:
        path = ROOT / "datasets" / name
        conv = json.loads((path / "r0_conversion.json").read_text())
        ds = LeRobotDataset("local/k18_validation", root=path, video_backend="pyav",
                            delta_timestamps={"action": [i / 20 for i in range(50)]})
        groups = {}
        offset = 0
        for ep in conv["episodes"]:
            groups.setdefault(ep["goal"], []).append((ep["demo"], offset, ep["frames"]))
            offset += ep["frames"]
        assert offset == len(ds)
        for goal, demos in groups.items():
            chosen = sorted({round((len(demos) - 1) * f) for f in (0., 1 / 3, 2 / 3, 1.)})
            for index in chosen:
                demo, start, length = demos[index]
                for part in (.2, .5, .8):
                    frame = start + min(length - 1, int(part * length))
                    raw = torch.utils.data.default_collate(ds.__getitems__([frame]))
                    yield {"goal": goal, "demo": demo, "frame": frame, "dataset": name, "position_fraction": part}, raw


def make_prefix(lab, role, raw, text):
    rt = lab.rt
    policy = rt.models[role]
    values = copy.deepcopy(raw)
    values["task"] = [text]
    batch = rt.pre[role](values)
    images, image_masks = policy.prepare_images(batch)
    inputs = (images, image_masks, batch[rt.constants.OBS_LANGUAGE_TOKENS],
              batch[rt.constants.OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_state(batch))
    return rt.make_prefix_fn(policy, inputs, rt.make_masks)


def main():
    lab = Lab5()
    bank = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, (identity, raw) in enumerate(select_records()):
            batch = lab.rt.pre["R"](copy.deepcopy(raw))
            actions = lab.rt.models["R"].prepare_action(batch)
            valid = ~batch["action_is_pad"][0]
            noise = torch.randn(actions.shape, generator=torch.Generator().manual_seed(20261111 + index)).cuda()
            prefix = {g: make_prefix(lab, "R", raw, INSTR[g]) for g in "ABN"}
            drafts = []
            for value in (.9, .5, .1):
                t = torch.tensor([value], device="cuda")
                x, target, _ = make_flow_matching_inputs(actions, noise, t, convention=FlowConvention.NOISE_AT_ONE)
                old = [lab.rt.query("R", prefix[g], x, t).float() for g in "ABN"]
                drafts.append((value, x, t, target, old))
            bank.append((identity, raw, valid, drafts))
    for step in STEPS:
        load(lab, checkpoint(lab, step))
        rows = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for identity, raw, valid, drafts in bank:
                prefix = {g: make_prefix(lab, "F", raw, INSTR[g]) for g in "ABN"}
                for time_value, x, t, target, old in drafts:
                    a, b, front = [lab.rt.query("F", prefix[g], x, t).float() for g in "ABN"]
                    ref = ((old[0] - old[1]) / 2)[0, valid, :7].flatten()
                    d = ((a - b) / 2)[0, valid, :7].flatten()
                    mu = (a + b) / 2
                    delta = mu - (old[0] + old[1]) / 2
                    actual = {"left": a, "right": b, "front": front}[identity["goal"]]
                    error = (actual - target)[0, valid, :7].square()
                    rows.append({**identity, "time": time_value, "gain": float((ref * d).sum()) / max(float(ref.square().sum()), 1e-12),
                                 "parent_contrast_rms": float(ref.square().mean().sqrt()),
                                 "mu_drift": {k: float(delta[0, valid, v].square().mean().sqrt()) for k, v in BLOCKS.items()},
                                 "mu_front_distance": float((mu - front)[0, valid, :7].square().mean().sqrt()),
                                 "matched_instruction_mse": float(error.mean()),
                                 "matched_instruction_channel_mse": error.mean(0).tolist()})
        if step == 0:
            assert max(abs(r["gain"] - 1) for r in rows) < 1e-6
        (RUN / f"demo_probe_{step:06d}.json").write_text(json.dumps(rows, allow_nan=False))
        print("DEMO_PROBE", step, len(rows), flush=True)
    lab.ctx.env.close()


if __name__ == "__main__":
    main()
