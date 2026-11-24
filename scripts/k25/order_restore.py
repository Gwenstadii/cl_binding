"""K25 single-policy repair across curricula, checkpoints and held-out wording.

Run from a new .server/k25_order_restore_*/ directory beside the archived K runs.
The frozen base, processors and simulator come from the existing project runtime.
No parent policy is loaded during inference. Goal labels enter scoring only.
"""

import argparse
from collections import Counter
import copy
from hashlib import file_digest
import json
from pathlib import Path
import random
import sys
from time import monotonic

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

RUN = Path(__file__).resolve().parent
SERVER = RUN.parent
sys.path[:0] = [str(SERVER / "k25_final_20261121_01")]
import self_restore as sr

from stage1_smoke import ROOT, load_policy
from stage1_protocol import balanced_indices
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors

LR_PARENT = SERVER / "r1_left_right_20261104_01/train_r1/checkpoints/024000"
LF_PARENT = Path("/home/lzx/cl_binding_stage2a/outputs/r0/dual_left_front_20261023_01/checkpoints/012000")
HISTORY = SERVER / "k18_training_cause_20261111_01/train/checkpoints"
DEV_IDS = list(range(200, 208))
TEST_IDS = list(range(800, 820))
STANDARD_GAIN = 5.294038631688923
DEV_GAINS = (1., 3., STANDARD_GAIN)
SEED = 20261021
LINES = {
    "lr_front_130": dict(goals=("left", "right", "front"), step=130, dev="lr_front_130"),
    "lr_front_300": dict(goals=("left", "right", "front"), step=300, dev="lr_front_1000"),
    "lr_front_1000": dict(goals=("left", "right", "front"), step=1000, dev="lr_front_1000"),
    "lf_right_1000": dict(goals=("left", "front", "right"), step=1000, dev="lf_right_1000"),
}
VARIANTS = {"move": "Move the book to the caddy compartment on the {goal}."}


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def names():
    return read(SERVER / "r1_then_front_v1/train/trainable_parameters.json")["names"]


def selected_weights(path):
    with safe_open(path, framework="pt") as source:
        return {n: source.get_tensor(n) for n in names()}


def train():
    """A new LF -> Right stream; same aligned data/optimizer recipe as LR -> Front."""
    from train_controls import frozen_hash

    dest = RUN / "lf_right_train"
    dest.mkdir(exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    policy = load_policy(LF_PARENT)
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=LF_PARENT)
    dataset_path = ROOT / "datasets/stage1_single_right_train_aligned_v1"
    conversion = read(dataset_path / "r0_conversion.json")
    assert conversion["spec"]["goals"] == ["right"]
    assert conversion["spec"]["split"] == "train" and conversion["spec"]["action_offset"] == 1
    assert len(conversion["episodes"]) == 40
    ds = LeRobotDataset("local/k25_order", root=dataset_path, video_backend="pyav",
                        delta_timestamps={"action": [i/20 for i in range(50)]})
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    assert [n for n, _ in named] == names() and len(named) == 155
    frozen = frozen_hash(policy)
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=policy.config.optimizer_betas,
                                 eps=policy.config.optimizer_eps, weight_decay=policy.config.optimizer_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: min(1., (k+1)/100))
    manifest = dict(curriculum="Left+Front -> Right", parent=str(LF_PARENT), parent_sha256=sha(LF_PARENT / "model.safetensors"),
                    seed=SEED, steps=1000, saves=[30, 300, 1000], learning_rate=1e-4, warmup=100,
                    batch_size=6, data=str(dataset_path), conversion=conversion["spec"],
                    frozen_hash=frozen, optimizer_reset=True, normalization="inherit LF parent", replay=False,
                    source_sha256=sha(Path(__file__)))
    write(dest / "manifest.json", manifest)
    policy.train()
    started = monotonic()
    with (dest / "train.jsonl").open("x", encoding="utf-8") as log:
        for k in range(1000):
            indices = balanced_indices({"right": list(range(len(ds)))}, 6, SEED, k)
            batch = pre(copy.deepcopy(torch.utils.data.default_collate(ds.__getitems__(indices))))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = policy(batch, reduction="none")[0].mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, policy.config.optimizer_grad_clip_norm, error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            row = dict(step=k+1, indices=indices, loss=float(loss.detach()), grad_norm=float(norm), lr=lr,
                       seconds=monotonic()-started)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if k+1 in manifest["saves"]:
                path = dest / f"{k+1:06d}.safetensors"
                state = {n: p.detach().cpu().contiguous() for n, p in named}
                save_file(state, str(path))
                restored = load_file(str(path))
                assert all(torch.equal(v, restored[n]) for n, v in state.items())
                write(dest / f"{k+1:06d}_READY.json", dict(step=k+1, sha256=sha(path), tensors=155))
            if k < 3 or (k+1) % 100 == 0:
                print("ORDER_TRAIN", row, flush=True)
    assert frozen_hash(policy) == frozen
    write(dest / "COMPLETE.json", dict(steps=1000, frozen_unchanged=True, seconds=monotonic()-started,
                                        manifest_sha256=sha(dest / "manifest.json")))


def source(line):
    if line == "lf_right_1000":
        return LF_PARENT, RUN / "lf_right_train/001000.safetensors", RUN / "lf_right_train/000030.safetensors"
    step = LINES[line]["step"]
    late = sr.LATE if step == 130 else HISTORY / f"{step:06d}/model.safetensors"
    return LR_PARENT, late, HISTORY / "000030/model.safetensors"


def weights(line):
    parent, late, anchor = source(line)
    full, early = selected_weights(late), selected_weights(anchor)
    _, chosen = sr.parameters_for("cross_v", full)
    deltas = {n: early[n].double()-full[n].double() for n in sorted(chosen)}
    assert len(deltas) == 8
    assert all(torch.equal((full[n].double()+d).to(full[n].dtype), early[n]) for n, d in deltas.items())
    return parent, full, deltas


class Lab(sr.SingleLab):
    def __init__(self, parent):
        self.ctx = sr.km.Ctx("scene2")
        self.torch, self.np = self.ctx.torch, self.ctx.np
        self.rt = sr.km.make_runtime(self.ctx, {"FT4": str(parent)}, ["F"])
        assert tuple(self.rt.models) == ("F",)


def configure(line):
    sr.LABEL = dict(zip("ABN", LINES[line]["goals"], strict=True))
    sr.INSTR = {g: f"pick up the book and place it in the {goal} compartment of the caddy" for g, goal in sr.LABEL.items()}


def prepare():
    assert not (RUN / "protocol.json").exists()
    # Reuse development resets only. Fresh test states have no grasp/outcome filter.
    candidates = read(SERVER / "k19_retention_20261111_01/candidates.json")["records"]
    dev = [r for r in candidates if r["init"] in DEV_IDS]
    assert [r["init"] for r in dev] == DEV_IDS
    ctx = sr.km.Ctx("scene2")
    test = []
    known = [np.asarray(r["state"])[1:] for r in candidates]
    known += [np.asarray(r["state"])[1:] for r in read(sr.HOLDOUT)["selected"]]
    try:
        for i, init in enumerate(TEST_IDS):
            seed = 995201+i
            np.random.seed(seed)
            ctx.env.seed(seed)
            ctx.env.reset()
            state = np.asarray(ctx.env.get_sim_state(), dtype=np.float64).copy()
            distance = min(float(np.linalg.norm(state[1:]-ref)) for ref in known)
            assert distance > 1e-6
            rec = dict(init=init, state=state.tolist(), reset_seed=seed, seed=15960041+107*i,
                       min_distance_previous_resets=distance)
            obs = sr.reset(ctx, rec)
            assert not any(ctx.flags().values())
            rec.update(signature=ctx.signature(obs), z0=float(obs[ctx.obj][2]), book_start=obs[ctx.obj].tolist())
            assert rec["signature"]["sim"] not in {r["signature"]["sim"] for r in test}
            test.append(rec)
    finally:
        ctx.env.close()
    write(RUN / "cases.json", dict(dev=dev, test=test, test_filter="geometry at reset only; no rollout screening"))
    checkpoints = {}
    for line in LINES:
        parent, late, anchor = source(line)
        assert parent.exists() and late.exists() and anchor.exists()
        checkpoints[line] = {"parent": str(parent), "late": str(late), "anchor": str(anchor),
                             "parent_sha256": sha(parent / "model.safetensors"),
                             "late_sha256": sha(late), "anchor_sha256": sha(anchor)}
    write(RUN / "protocol.json", dict(lines=LINES, window=[80, 180], horizon=400, dev_gains=DEV_GAINS,
          dev_lines=["lr_front_130", "lr_front_1000", "lf_right_1000"], variants=VARIANTS,
          test_ids=TEST_IDS, dev_ids=DEV_IDS, checkpoints=checkpoints, cases_sha256=sha(RUN / "cases.json"),
          standard_gain=STANDARD_GAIN, alpha_selection="maximize min(old goal first), joint first, old terminal sum, then smaller gain",
          main_arms=["native ABN", "forward AB", "reverse AB"],
          extra_standard_line="lr_front_130 if selected gain differs", phrase_lines=["lr_front_1000", "lf_right_1000"],
          source_sha256=sha(Path(__file__)), baseline_source_sha256=sha(Path(sr.__file__))))


def episode(lab, model, rec, text, arm, gain):
    started = monotonic()
    obs = sr.reset(lab.ctx, rec)
    assert lab.ctx.signature(obs) == rec["signature"]
    rows, decisions = [], []
    boundary_signature = None
    for k in range(40):
        if k == 8:
            boundary_signature = lab.ctx.signature(obs)
        if arm == "native" or not 8 <= k < 18:
            action = lab.native("F", text, obs, rec["seed"], k)
        else:
            action, decision = model.patch(text, obs, rec["seed"], k, "v_forward" if arm == "forward" else "v_reverse")
            decisions.append(decision)
        obs, events = lab.execute(obs, lab.rt.physical(action), k*10, rec["z0"])
        rows.extend(events)
    assert len(rows) == 400
    c = lab.ctx
    end = c.geo.movement("scene2", c.geometry, rows, rec["z0"], None)
    grasp = c.geo.first_run([r["step"] for r in rows if r["obj"][2]-rec["z0"] > .03 and r["c1"] and r["c2"]])
    first_mm = {}
    for goal in sr.LABEL.values():
        center = np.asarray(c.geometry[goal]["centre"])[:2]
        first_mm[goal] = 1000*float(np.linalg.norm(np.asarray(rows[79]["obj"])[:2]-center)
                                    - np.linalg.norm(np.asarray(rows[89]["obj"])[:2]-center))
    return dict(init=rec["init"], text=text, arm=arm, gain=gain, first=end["first_targets"], times=end["times"],
                first_step=end["first_step"], terminal=rows[-1]["pred"], first_grasp=grasp,
                ever={g: any(r["pred"][g] for r in rows) for g in c.geo.GOALS},
                boundary_signature=boundary_signature, first_10_approach_mm=first_mm,
                decisions=decisions, routed=dict(Counter(d["decoded"] for d in decisions)), seconds=monotonic()-started)


def gate(lab, model, full, deltas, rec):
    obs = sr.reset(lab.ctx, rec)
    rt = lab.rt
    x, t = rt.noise(rec["seed"], 0), torch.ones(1, device="cuda")
    pref = {g: rt.prefix("F", obs, sr.INSTR[g]) for g in "ABN"}
    base = {g: model.value(pref[g], x, t) for g in "ABN"}
    relay = model.relays[1.]
    projected = relay.cache({g: pref[g] for g in "AB"})
    changed = {}
    for g in "AB":
        with relay.whole(projected, g, False, verify=True):
            changed[g] = model.value(pref[g], x, t)
    params = dict(rt.models["F"].named_parameters())
    try:
        for n, v in model.overlay(1.).items():
            params[n].copy_(v)
        assert all(torch.equal(changed[g], model.value(pref[g], x, t)) for g in "AB")
    finally:
        for n in deltas:
            params[n].copy_(full[n])
    assert all(torch.equal(base[g], model.value(pref[g], x, t)) for g in "ABN")
    delta = sr.contrast(changed["A"], changed["B"])-sr.contrast(base["A"], base["B"])
    a = base["A"].double()+model.gain*delta.double()
    b = base["B"].double()-model.gain*delta.double()
    torch.testing.assert_close(a+b, base["A"].double()+base["B"].double(), atol=1e-12, rtol=1e-12)
    mean_cast_rms = sr.rms((a.to(base["A"].dtype).float()+b.to(base["B"].dtype).float())*.5
                            -(base["A"].float()+base["B"].float())*.5)
    native = lab.native("F", sr.INSTR["N"], obs, rec["seed"], 0)
    tested, decoded = model.patch(sr.INSTR["N"], obs, rec["seed"], 0, "v_forward")
    assert decoded["decoded"] == "N" and decoded["coefficient"] == 0. and torch.equal(native, tested)
    return dict(single_model_roles=list(rt.models), v_only_overlay_equal=True, weights_restored=True,
                mean_preserved_fp64=True, mean_cast_rms=mean_cast_rms, new_goal_bypass_bitwise=True)


def worker(suite, line, lane, lanes):
    protocol = read(RUN / "protocol.json")
    assert protocol["source_sha256"] == sha(Path(__file__))
    assert protocol["cases_sha256"] == sha(RUN / "cases.json")
    for kind in ("late", "anchor"):
        assert sha(Path(protocol["checkpoints"][line][kind])) == protocol["checkpoints"][line][kind+"_sha256"]
    configure(line)
    parent, full, deltas = weights(line)
    bank = read(RUN / "cases.json")["dev" if suite == "dev" else "test"][lane::lanes]
    assert bank
    if suite == "dev":
        jobs = [("canon", "native", 0., "AB")]+[("canon", "forward", g, "AB") for g in DEV_GAINS]
    else:
        choice = read(RUN / "selection.json")["selected"][LINES[line]["dev"]]["gain"]
        jobs = [("move" if suite == "phrase" else "canon", arm, 0. if arm == "native" else choice,
                 "ABN" if suite == "main" and arm == "native" else "AB") for arm in ("native", "forward", "reverse")]
        if suite == "main" and line == "lr_front_130" and choice != STANDARD_GAIN:
            jobs.append(("canon", "forward", STANDARD_GAIN, "AB"))
    path = RUN / f"{suite}_{line}_{lane}.jsonl"
    old = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    seen = {(r["variant"], r["arm"], r["gain"], r["init"], r["goal"]) for r in old}
    lab = Lab(parent)
    checks = []
    try:
        sr.assign(lab, "F", full)
        for variant, arm, gain, goals in jobs:
            # Relay caches only V tensors; this creates no second policy.
            model = sr.Restore(lab, full, deltas, gain or STANDARD_GAIN, sr.read(sr.MANIFEST)["own_d_gain"])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                checks.append(dict(gain=gain, **gate(lab, model, full, deltas, bank[0])))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                with path.open("a", encoding="utf-8") as out:
                    for rec in bank:
                        for goal in goals:
                            key = variant, arm, gain, rec["init"], goal
                            if key in seen:
                                continue
                            text = sr.INSTR[goal] if variant == "canon" else VARIANTS[variant].format(goal=sr.LABEL[goal])
                            row = episode(lab, model, rec, text, arm, gain)
                            row.update(suite=suite, line=line, variant=variant, goal=goal, target=sr.LABEL[goal])
                            out.write(json.dumps(row, allow_nan=False)+"\n")
                            out.flush()
                            print("ORDER_EVAL", suite, line, lane, gain, arm, rec["init"], goal,
                                  row["first"], row["terminal"][sr.LABEL[goal]], round(row["seconds"], 2), flush=True)
            del model
    finally:
        lab.ctx.env.close()
    write(RUN / f"{suite}_{line}_{lane}_COMPLETE.json", dict(suite=suite, line=line, lane=lane, lanes=lanes,
          ids=[r["init"] for r in bank], gates=checks, rows_sha256=sha(path),
          protocol_sha256=sha(RUN / "protocol.json"), source_sha256=sha(Path(__file__)),
          selection_sha256=sha(RUN / "selection.json") if suite != "dev" else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "prepare", "dev", "main", "phrase"))
    parser.add_argument("--line", choices=LINES)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "train":
        # The helper is imported after self_restore has initialized its dependency paths.
        sys.path.insert(0, str(SERVER / "k18_training_controls_20261111_01"))
        train()
    elif args.mode == "prepare":
        prepare()
    else:
        assert args.line and 0 <= args.lane < args.lanes
        worker(args.mode, args.line, args.lane, args.lanes)
