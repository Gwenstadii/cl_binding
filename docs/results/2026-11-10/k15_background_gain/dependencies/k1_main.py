"""K1 broad screen driver (design: K1广筛预登记方案_多方向严格泛化_20261101.md, prereg/k1_prereg.json).

Phases
  prefix       native prefix rollouts (400 steps) of R0 and every FT4, inits 4-9, prefix instruction A/B by parity
  plan         T1 / T2 handoffs, snapshot list, eligibility (prefix rollouts only; no continuation exists yet)
  engineering  gates: snapshot reconstruction x3, native / reconstruction identity, instruction-swap symmetry,
               zero-difference leak, geometry, timing
  worker       continuations: every snapshot x 18 K1 conditions + 3 natives x 2 eval instructions
  collect      merge worker files -> records.jsonl + gates.json, then k1_analyze.py

Snapshots are rebuilt by replaying the prefix rollout's executed actions from reset (controller state preserved),
refreshing observations at every planning boundary exactly as in the prefix (collaborator's stage2r protocol).
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
RUNTIME = Path("/home/lzx/cl_binding_stage2a")
FAMILY_CFG = {"scene2": "/home/lzx/cl_binding/.server/stage2h_gate_20261027/configs/gcid.json",
              "scene1b": "/home/lzx/cl_binding_stage2a/configs/stage3b_scene1_replay_v1.json",
              "goal": "/home/lzx/cl_binding_stage2a/configs/stage3g_goal_replay_v1.json"}
REF_GEOMETRY = "/home/lzx/cl_binding/.server/stage2p_movement_20261029_01/geometry.json"
SLOT = {"A": "left", "B": "front"}
INV = {"left": "A", "front": "B", "right": "new", "back": "unseen"}
MIN_FREE = 3 * 1024 ** 3
ENDPOINTS = ("first_goal", "first_goal_step", "native_success", "success_step", "any_target_reached", "grasp_retained", "eligible")


def jdump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prereg():
    return json.loads((OUT / "prereg/k1_prereg.json").read_text())


def noise_seed(family, init):
    return int(hashlib.sha256(f"k1|{family}|{init}".encode()).hexdigest()[:8], 16)


def guard_disk():
    if shutil.disk_usage(OUT).free < MIN_FREE:
        raise RuntimeError("Disk free below 3 GB; stopping without cleanup")


class Ctx:
    """Per-family simulator context and libraries."""

    def __init__(self, family):
        sys.path.insert(0, str(RUNTIME / "scripts"))
        import numpy as np
        import torch
        import lerobot.policies as policies
        from lerobot.utils import constants
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        from lerobot.policies.common.flow_matching import euler_integrate
        from robosuite.utils.transform_utils import quat2axisangle
        import stage1_r0_eval as evaluation
        from stage1_smoke import DEFAULT_SCENE
        from stage2k_entry_handoff import _fresh_observation, _control_state
        from stage2m_cross_attention import make_prefix
        import k1_geometry as geo
        self.np, self.torch, self.policies, self.constants = np, torch, policies, constants
        self.make_masks, self.integrate, self.q2a, self.make_prefix = make_att_2d_masks, euler_integrate, quat2axisangle, make_prefix
        self.evaluation, self.fresh, self.control, self.geo = evaluation, _fresh_observation, _control_state, geo
        self.family = family
        self.root = json.loads(Path(FAMILY_CFG[family]).read_text())
        tasks, paths, predicates, states = evaluation.task_resources(self.root.get("scene", DEFAULT_SCENE))
        self.texts = {"A": tasks["left"].language, "B": tasks["front"].language}
        self.all_texts = {g: tasks[g].language for g in tasks}
        self.predicates, self.states = predicates, states
        self.env = evaluation.make_env(paths["front"])
        self.xml = evaluation.canonical_scene(paths["front"], self.root["seed"])
        self.xml_sha = hashlib.sha256(self.xml.encode()).hexdigest()
        self.obj = geo.OBJECT[family]
        self.reset(4)
        self.geometry = geo.build(family, self.env, np)
        if family == "scene2":
            ref = json.loads(Path(REF_GEOMETRY).read_text())
            assert ref["canonical_scene_xml_sha256"] == self.xml_sha, "scene2 canonical scene differs from reference"
            geo.check_against(self.geometry, ref["sites"])

    def reset(self, init):
        np = self.np
        np.random.seed(self.root["seed"] + init)
        self.evaluation.reset_common(self.env, self.states["front"][init], self.root["seed"] + init,
                                     self.root["evaluation"]["settle_steps"], self.xml)
        obs = self.fresh(self.env)
        sha = hashlib.sha256(np.asarray(self.env.get_sim_state(), dtype=np.float64).tobytes()).hexdigest()
        return obs, sha

    def flags(self):
        return {g: bool(all(self.env.env._eval_predicate(p) for p in parts)) for g, parts in self.predicates.items()}

    def row(self, step, obs, action, executed):
        np = self.np
        out = {"step": step, "pred": self.flags(), "a": [float(x) for x in action],
               "clip": bool((action != executed).any()), "eef": [float(x) for x in obs["robot0_eef_pos"]],
               "obj": [float(x) for x in obs[self.obj]]}
        return out

    def signature(self, obs):
        np = self.np
        state = np.asarray(self.env.get_sim_state(), dtype=np.float64)
        return {"sim": hashlib.sha256(state.tobytes()).hexdigest(), "control": digest(self.control(self.env)),
                "obs": {k: hashlib.sha256(np.ascontiguousarray(obs[k]).tobytes()).hexdigest()
                        for k in ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}}

    def plate_xy(self, obs):
        return [float(x) for x in obs["plate_1_pos"][:2]] if self.family == "goal" else None

    def step(self, action):
        executed = self.np.clip(action, -1, 1)
        obs, _, _, _ = self.env.step(executed)
        return obs, executed


def make_runtime(ctx, pair, roles):
    import k1_runtime as K
    return K.Runtime(pair, RUNTIME, roles, ctx.evaluation, ctx.policies, ctx.constants, ctx.make_masks,
                     ctx.integrate, ctx.q2a, ctx.make_prefix, ctx.torch, ctx.np)


# ---------------------------------------------------------------- prefix
def prefix_path(family, key, init):
    return OUT / "prefix" / family / f"{key}_i{init}.json.gz"


def run_prefix(ctx, rt, cond, key, init, instr):
    path = prefix_path(ctx.family, key, init)
    if path.exists():
        return
    seed = noise_seed(ctx.family, init)
    obs, sha0 = ctx.reset(init)
    plate = ctx.plate_xy(obs)
    rows, sigs = [], {}
    t0 = time.monotonic()
    for r in range(40):
        chunk, _ = rt.sample(cond, instr, obs, ctx.texts, seed, r)
        controls = rt.physical(chunk)
        for k in range(10):
            step = r * 10 + k + 1
            obs, executed = ctx.step(controls[k])
            if step % 10 == 0:
                obs = ctx.fresh(ctx.env)
            rows.append(ctx.row(step, obs, controls[k], executed))
            if step % 10 == 0 and step <= 200:
                sigs[str(step)] = ctx.signature(obs)
    closes = [x["step"] for x in rows if x["a"][6] > 0]
    success = {g: next((x["step"] for x in rows if x["pred"][g]), None) for g in ctx.geo.GOALS}
    out = {"family": ctx.family, "model_key": key, "model_sha": rt.sha[next(iter(rt.sha))], "init": init,
           "instruction": instr, "noise_seed": seed, "initial_sha": sha0, "plate_xy": plate,
           "first_close": closes[0] if closes else None, "native_success_step": success, "signatures": sigs,
           "seconds": time.monotonic() - t0, "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path) + ".tmp", "wt") as f:
        json.dump(out, f)
    os.replace(str(path) + ".tmp", path)
    print(json.dumps({k: out[k] for k in ("family", "model_key", "init", "instruction", "first_close", "native_success_step", "seconds")}), flush=True)


def phase_prefix(args):
    P = prereg()
    family = args.family
    ctx = Ctx(family)
    inits = P["snapshots"]["physical_inits"]
    pairs = {p: c for p, c in P["pairs"].items() if c["family"] == family}
    first = next(iter(pairs))
    jobs = [("R0", pairs[first], "R", "NAT-R0")] + [(f"FT4_{p}", c, "F", "NAT-FT4") for p, c in pairs.items()]
    for key, pair, role, cond in jobs:
        rt = make_runtime(ctx, pair, {role})
        with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
            for init in inits:
                run_prefix(ctx, rt, cond, key, init, "A" if init % 2 == 0 else "B")
        del rt
        import gc
        gc.collect()
        ctx.torch.cuda.empty_cache()
    jdump(OUT / "prefix" / family / "geometry.json", {"family": family, "scene_xml_sha256": ctx.xml_sha, "regions": ctx.geometry,
                                                      "texts": ctx.texts, "all_texts": ctx.all_texts})


def load_prefix(family, key, init):
    with gzip.open(prefix_path(family, key, init), "rt") as f:
        return json.load(f)


# ---------------------------------------------------------------- plan
def phase_plan(args):
    import k1_geometry as geo
    P = prereg()
    inits = P["snapshots"]["physical_inits"]
    plan = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "T1": {}, "snapshots": [], "missing": []}
    fam_pairs = {}
    for p, c in P["pairs"].items():
        fam_pairs.setdefault(c["family"], []).append(p)
    for family, pairs in fam_pairs.items():
        G = json.loads((OUT / "prefix" / family / "geometry.json").read_text())["regions"]
        primary = next(p for p in pairs if p in P["primary_parents"])
        closes = [load_prefix(family, k, i)["first_close"] for i in inits for k in ("R0", f"FT4_{primary}")]
        closes = [c for c in closes if c is not None]
        med = statistics.median(closes) if closes else None
        T1 = 40 if med is None or med >= 40 else max(10, int((med - 1) // 10) * 10)
        plan["T1"][family] = {"step": T1, "median_first_close": med, "closes": closes}
    blocks = [tuple(b) for b in P["blocks"]]
    for pair_id, handoff in blocks:
        family = P["pairs"][pair_id]["family"]
        G = json.loads((OUT / "prefix" / family / "geometry.json").read_text())["regions"]
        for init in inits:
            for src, key in (("R0", "R0"), ("FT4", f"FT4_{pair_id}")):
                pre = load_prefix(family, key, init)
                if handoff == "T1":
                    h = plan["T1"][family]["step"]
                else:
                    fc = pre["first_close"]
                    h = None if fc is None or fc > 200 else int(-(-(fc + 10) // 10) * 10)
                    if h is not None and h > 200:
                        h = None
                if h is None:
                    plan["missing"].append({"pair_id": pair_id, "handoff_type": handoff, "init": init, "prefix_policy": src,
                                            "reason": "no gripper close by step 200"})
                    continue
                rows = pre["rows"]
                z0 = rows[0]["obj"][2]
                mv = geo.movement(family, G, rows[:h], z0, pre["plate_xy"])
                at = geo.hits(family, G, rows[h - 1]["obj"], z0, pre["plate_xy"])
                native_before = any(any(r["pred"].values()) for r in rows[:h])
                plan["snapshots"].append({
                    "snapshot_id": f"{pair_id}|{handoff}|i{init}|{src}", "pair_id": pair_id, "family": family,
                    "handoff_type": handoff, "handoff_step": h, "physical_init_id": init, "prefix_policy": src,
                    "prefix_key": key, "prefix_instruction": pre["instruction"], "noise_id": pre["noise_seed"],
                    "signature": pre["signatures"][str(h)], "snapshot_hash": pre["signatures"][str(h)]["sim"],
                    "prefix_first_targets": mv["first_targets"], "inside_at_handoff": at, "native_before": native_before,
                    "eligible": not mv["first_targets"] and not native_before and not at})
    jdump(OUT / "plan.json", plan)
    n_el = sum(s["eligible"] for s in plan["snapshots"])
    print(json.dumps({"T1": plan["T1"], "snapshots": len(plan["snapshots"]), "eligible": n_el, "missing": len(plan["missing"])}, ensure_ascii=False))


# ---------------------------------------------------------------- continuation
def rebuild(ctx, snap, pre):
    obs, sha0 = ctx.reset(snap["physical_init_id"])
    if sha0 != pre["initial_sha"]:
        raise AssertionError("Initial state differs from prefix rollout")
    err = 0.0
    for row in pre["rows"][:snap["handoff_step"]]:
        a = ctx.np.asarray(row["a"])
        obs, executed = ctx.step(a)
        if row["step"] % 10 == 0:
            obs = ctx.fresh(ctx.env)
        e = max(max(abs(x - y) for x, y in zip(obs["robot0_eef_pos"], row["eef"])),
                max(abs(x - y) for x, y in zip(obs[ctx.obj], row["obj"])))
        err = max(err, float(e))
    if err > 1e-7:
        raise AssertionError(f"Prefix replay drifted {err}")
    if ctx.signature(obs) != snap["signature"]:
        raise AssertionError("Snapshot signature differs from prefix rollout")
    return obs, err


def continuation(ctx, rt, snap, pre, cond, instr, G):
    geo = ctx.geo
    obs, replay_err = rebuild(ctx, snap, pre)
    h = snap["handoff_step"]
    rows, metas = [], []
    t0 = time.monotonic()
    for r in range(h // 10, 40):
        chunk, meta = rt.sample(cond, instr, obs, ctx.texts, snap["noise_id"], r)
        metas.append(meta)
        controls = rt.physical(chunk)
        for k in range(10):
            step = r * 10 + k + 1
            obs, executed = ctx.step(controls[k])
            if step % 10 == 0:
                obs = ctx.fresh(ctx.env)
            rows.append(ctx.row(step, obs, controls[k], executed))
    z0 = pre["rows"][0]["obj"][2]
    mv = geo.movement(ctx.family, G, rows, z0, pre["plate_xy"], h + 1)
    ft = mv["first_targets"]
    first = None if not ft else (INV[ft[0]] if len(ft) == 1 else "tie")
    goal = SLOT[instr]
    succ = next((x["step"] for x in rows if x["pred"][goal]), None)
    identity = None
    native_like = {"R0": ("NAT-R0", "K1-08"), "FT4": ("NAT-FT4", "K1-01")}[snap["prefix_policy"]]
    if instr == snap["prefix_instruction"] and cond in native_like:
        ref = pre["rows"][h:]
        identity = max(max(abs(x - y) for x, y in zip(a["a"], b["a"])) for a, b in zip(rows, ref))
    flat = [m for m in metas]
    def med(key):
        v = [x for m in flat for x in m.get(key, [])]
        return statistics.median(v) if v else None
    deg = [x for m in flat for x in m.get("degenerate", [])]
    nerr = [x for m in flat for x in m.get("norm_err", [])]
    rec = {"run_id": f"{snap['snapshot_id']}|{cond}|{instr}", "pair_id": snap["pair_id"], "family": snap["family"],
           "checkpoint_pair_id": snap["pair_id"], "handoff_type": snap["handoff_type"], "handoff_step": h,
           "physical_init_id": snap["physical_init_id"], "prefix_policy": snap["prefix_policy"],
           "prefix_instruction": snap["prefix_instruction"], "noise_id": snap["noise_id"], "condition_id": cond,
           "eval_instruction": instr, "first_goal": first, "first_goal_step": mv["first_step"], "first_goal_times": mv["times"],
           "native_success": succ is not None, "success_step": succ,
           "any_target_reached": first is not None, "grasp_retained": geo.lifted_run(rows, z0, h + 1),
           "clipping_rate": sum(x["clip"] for x in rows) / len(rows),
           "norm_match_error": max(nerr) if nerr else 0.0, "degenerate_rate": (sum(deg) / len(deg)) if deg else 0.0,
           "eligible": snap["eligible"], "snapshot_hash": snap["snapshot_hash"], "replay_error": replay_err,
           "identity_error": identity, "action_seq_sha": hashlib.sha256(json.dumps([x["a"] for x in rows]).encode()).hexdigest(),
           "terminal_predicates": rows[-1]["pred"], "seconds": time.monotonic() - t0,
           "readout_median": {k: med(k) for k in ("a_R", "a_F", "a_E", "cos_uR_uF", "mu_rel_F_R", "cos_uR_uE")},
           "trajectory_path": None}
    return rec, rows, metas


def conditions(P, handoff):
    conds = [c for c in P["conditions"] if c.startswith("K1-")] + P["conditions"]["native"]
    return conds


def phase_worker(args):
    P = prereg()
    plan = json.loads((OUT / "plan.json").read_text())
    snaps = plan["snapshots"]
    order = {tuple(b): i for i, b in enumerate(P["blocks"])}
    snaps = sorted(snaps, key=lambda s: (order[(s["pair_id"], s["handoff_type"])], s["physical_init_id"], s["prefix_policy"]))
    if args.only_blocks:
        keep = {tuple(x.split("/")) for x in args.only_blocks.split(",")}
        snaps = [s for s in snaps if (s["pair_id"], s["handoff_type"]) in keep]
    mine = [s for i, s in enumerate(snaps) if i % args.workers == args.worker]
    rundir = OUT / "runs"
    rundir.mkdir(parents=True, exist_ok=True)
    tag = f"w{args.worker:02d}" + (f"s{args.sub}" if args.subn > 1 else "")
    summ = rundir / f"{tag}.jsonl"
    done = set()
    for old in rundir.glob(f"w{args.worker:02d}*.jsonl"):  # all files of this worker, incl. earlier sub-splits
        done |= {json.loads(l)["run_id"] for l in old.read_text().splitlines() if l.strip()}
    if args.subn > 1:  # resume helper: split this worker's unfinished snapshots over several processes
        pending = [s for s in mine if any(f"{s['snapshot_id']}|{c}|{i}" not in done for c in conditions(P, s["handoff_type"]) for i in "AB")]
        mine = [s for j, s in enumerate(pending) if j % args.subn == args.sub]
    ctxs, rt, rt_pair = {}, None, None
    status = {"worker": args.worker, "workers": args.workers, "snapshots": len(mine), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "continuations_done": len(done), "pid": os.getpid()}
    for snap in mine:
        fam = snap["family"]
        if fam not in ctxs:
            ctxs[fam] = Ctx(fam)
        ctx = ctxs[fam]
        G = json.loads((OUT / "prefix" / fam / "geometry.json").read_text())["regions"]
        pre = load_prefix(fam, snap["prefix_key"], snap["physical_init_id"])
        todo = [(c, i) for c in conditions(P, snap["handoff_type"]) for i in ("A", "B")
                if f"{snap['snapshot_id']}|{c}|{i}" not in done]
        if not todo:
            continue
        if rt_pair != snap["pair_id"]:
            rt = None
            import gc
            gc.collect()
            ctx.torch.cuda.empty_cache()
            rt = make_runtime(ctx, P["pairs"][snap["pair_id"]], {"R", "F", "E"})
            rt_pair = snap["pair_id"]
        for cond, instr in todo:
            guard_disk()
            with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
                rec, rows, metas = continuation(ctx, rt, snap, pre, cond, instr, G)
            ev = rundir / f"{tag}_events.jsonl.gz"
            rec["trajectory_path"] = str(ev)
            with gzip.open(ev, "at") as f:
                f.write(json.dumps({"run_id": rec["run_id"], "rows": rows, "replans": metas}) + "\n")
            with summ.open("a") as f:
                f.write(json.dumps(rec, allow_nan=False) + "\n")
            status["continuations_done"] += 1
            status["last"] = rec["run_id"]
            status["last_seconds"] = rec["seconds"]
            status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            jdump(rundir / f"{tag}_status.json", status)
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(rundir / f"{tag}_status.json", status)


# ---------------------------------------------------------------- engineering
def phase_engineering(args):
    import k1_runtime as K
    P = prereg()
    plan = json.loads((OUT / "plan.json").read_text())
    pair_id = args.pair
    fam = P["pairs"][pair_id]["family"]
    ctx = Ctx(fam)
    G = json.loads((OUT / "prefix" / fam / "geometry.json").read_text())["regions"]
    rt = make_runtime(ctx, P["pairs"][pair_id], {"R", "F", "E"})
    report = {"pair_id": pair_id, "family": fam, "model_sha": {K.ROLE_NAME[r]: s for r, s in rt.sha.items()},
              "processor_check": rt.processor_check, "gpu_memory_allocated_gb": ctx.torch.cuda.memory_allocated() / 1e9,
              "checks": [], "timing": {}}
    np, torch = ctx.np, ctx.torch
    snaps = [s for s in plan["snapshots"] if s["pair_id"] == pair_id and s["handoff_type"] == "T1"][:2]
    ok = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for snap in snaps:
            pre = load_prefix(fam, snap["prefix_key"], snap["physical_init_id"])
            chk = {"snapshot_id": snap["snapshot_id"]}
            sigs = []
            for _ in range(3):
                obs, err = rebuild(ctx, snap, pre)
                sigs.append(ctx.signature(obs))
            chk["rebuild_x3_identical"] = all(s == snap["signature"] for s in sigs)
            r = snap["handoff_step"] // 10
            seed = snap["noise_id"]
            chunks, times = {}, {}
            for cond in ("NAT-R0", "NAT-FT4", "NAT-ER", "K1-01", "K1-08", "K1-09", "K1-10", "K1-04", "K1-13", "K1-17", "K1-18"):
                for instr in ("A", "B"):
                    t0 = time.monotonic()
                    c, meta = rt.sample(cond, instr, obs, ctx.texts, seed, r)
                    torch.cuda.synchronize()
                    times.setdefault(cond, []).append(time.monotonic() - t0)
                    chunks[cond, instr] = c.float().cpu().numpy()
            mx = lambda a, b: float(np.max(np.abs(chunks[a] - chunks[b])))
            chk["K1-01_vs_NAT-FT4"] = max(mx(("K1-01", i), ("NAT-FT4", i)) for i in "AB")
            chk["K1-08_vs_NAT-R0"] = max(mx(("K1-08", i), ("NAT-R0", i)) for i in "AB")
            chk["K1-09_A_vs_B"] = mx(("K1-09", "A"), ("K1-09", "B"))
            chk["K1-10_A_vs_B"] = mx(("K1-10", "A"), ("K1-10", "B"))
            src = "NAT-R0" if snap["prefix_policy"] == "R0" else "NAT-FT4"
            phys = rt.physical(torch.from_numpy(chunks[src, snap["prefix_instruction"]]).cuda())[:10]
            ref = np.asarray([x["a"] for x in pre["rows"][snap["handoff_step"]:snap["handoff_step"] + 10]])
            chk["first_chunk_vs_prefix_rollout"] = float(np.max(np.abs(phys - ref)))
            f_ab = rt.fields({"R", "F"}, obs, ctx.texts, seed, r, ("A", "B"))
            f_ba = rt.fields({"R", "F"}, obs, ctx.texts, seed, r, ("B", "A"))
            chk["swap_mu_max_abs"] = max(float(np.max(np.abs(f_ab[m][0] - f_ba[m][0]))) for m in f_ab)
            chk["swap_d_plus_max_abs"] = max(float(np.max(np.abs(f_ab[m][1] + f_ba[m][1]))) for m in f_ab)
            chk["a_R_first_flow"] = float(np.sqrt((f_ab["R"][1][..., :7] ** 2).sum()))
            chk["a_F_first_flow"] = float(np.sqrt((f_ab["F"][1][..., :7] ** 2).sum()))
            chk["padding_mu_abs_max"] = max(float(np.max(np.abs(f_ab[m][0][..., 7:]))) for m in f_ab)
            chk["passed"] = (chk["rebuild_x3_identical"] and chk["K1-01_vs_NAT-FT4"] <= 1e-5 and chk["K1-08_vs_NAT-R0"] <= 1e-5
                             and chk["K1-09_A_vs_B"] == 0.0 and chk["K1-10_A_vs_B"] == 0.0
                             and chk["first_chunk_vs_prefix_rollout"] <= 1e-5 and chk["swap_mu_max_abs"] <= 1e-6
                             and chk["swap_d_plus_max_abs"] <= 1e-6)
            ok = ok and chk["passed"]
            report["checks"].append(chk)
            report["timing"][snap["snapshot_id"]] = {k: statistics.mean(v) for k, v in times.items()}
        # one full continuation to time it end to end
        snap = snaps[0]
        pre = load_prefix(fam, snap["prefix_key"], snap["physical_init_id"])
        t0 = time.monotonic()
        rec, rows, metas = continuation(ctx, rt, snap, pre, "K1-04", "A", G)
        report["full_continuation_K1-04_seconds"] = time.monotonic() - t0
        t0 = time.monotonic()
        rec2, _, _ = continuation(ctx, rt, snap, pre, "NAT-R0", "A", G)
        report["full_continuation_NAT-R0_seconds"] = time.monotonic() - t0
        report["example_records"] = [rec, rec2]
    report["gpu_max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    report["passed"] = ok
    jdump(OUT / "engineering" / f"{pair_id}.json", report)
    print(json.dumps({"pair": pair_id, "passed": ok, "timing": report["timing"],
                      "K1-04_full_s": report["full_continuation_K1-04_seconds"], "NAT_full_s": report["full_continuation_NAT-R0_seconds"],
                      "gpu_gb": report["gpu_max_memory_allocated_gb"], "checks": [{k: v for k, v in c.items()} for c in report["checks"]]},
                     ensure_ascii=False, default=str))


def phase_geometry_gate(args):
    """Gate 7: on prefix rollouts, every native success's first region entry is the succeeded goal, before success."""
    import k1_geometry as geo
    P = prereg()
    out = {}
    for family in ("scene2", "scene1b", "goal"):
        G = json.loads((OUT / "prefix" / family / "geometry.json").read_text())["regions"]
        rows_out = []
        for f in sorted((OUT / "prefix" / family).glob("*.json.gz")):
            with gzip.open(f, "rt") as h:
                pre = json.load(h)
            rows = pre["rows"]
            z0 = rows[0]["obj"][2]
            mv = geo.movement(family, G, rows, z0, pre["plate_xy"])
            succ = {g: s for g, s in pre["native_success_step"].items() if s is not None}
            first_succ = min(succ, key=succ.get) if succ else None
            agree = None
            if first_succ:
                agree = mv["first_targets"] == [first_succ] and mv["first_step"] is not None and mv["first_step"] <= succ[first_succ]
            rows_out.append({"file": f.name, "first_success_goal": first_succ, "first_success_step": succ.get(first_succ),
                             "geometry_first": mv["first_targets"], "geometry_step": mv["first_step"], "agree": agree})
        n = [r for r in rows_out if r["agree"] is not None]
        out[family] = {"success_episodes": len(n), "agree": sum(r["agree"] for r in n),
                       "passed": len(n) > 0 and all(r["agree"] for r in n), "rows": rows_out}
    jdump(OUT / "engineering" / "geometry_gate.json", out)
    print(json.dumps({f: {k: v for k, v in x.items() if k != "rows"} for f, x in out.items()}, ensure_ascii=False))


# ---------------------------------------------------------------- collect
def phase_collect(args):
    P = prereg()
    recs = []
    for f in sorted((OUT / "runs").glob("w*.jsonl")):
        recs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    seen, dup = set(), 0
    uniq = []
    for r in recs:
        if r["run_id"] in seen:
            dup += 1
            continue
        seen.add(r["run_id"])
        uniq.append(r)
    by = {}
    for r in uniq:
        by.setdefault((r["pair_id"], r["handoff_type"], r["physical_init_id"], r["prefix_policy"]), {})[r["condition_id"], r["eval_instruction"]] = r
    # amendment 2: criterion 6 uses arm-dimension clipping (gripper saturates by design)
    arm = {}
    for ev in sorted((OUT / "runs").glob("w*_events.jsonl.gz")):
        with gzip.open(ev, "rt") as h:
            for line in h:
                e = json.loads(line)
                rows = e["rows"]
                arm[e["run_id"]] = sum(any(abs(x) > 1 for x in r["a"][:6]) for r in rows) / len(rows)
    for r in uniq:
        r["clipping_rate_incl_gripper"] = r["clipping_rate"]
        r["clipping_rate"] = arm[r["run_id"]]
    geo_gate = json.loads((OUT / "engineering" / "geometry_gate.json").read_text())
    gates = {}
    for p, c in P["pairs"].items():
        eng = OUT / "engineering" / f"{p}.json"
        e = json.loads(eng.read_text()) if eng.exists() else {"passed": False}
        g = {"engineering_passed": bool(e.get("passed")), "geometry_passed": geo_gate[c["family"]]["passed"],
             "identity_failures": [], "reconstruction_failures": [], "zero_leak_failures": [], "degenerate_conditions": []}
        for key, conds in by.items():
            if key[0] != p:
                continue
            for (cond, instr), r in conds.items():
                if r["identity_error"] is not None and r["identity_error"] > 1e-5:
                    g["identity_failures"].append(r["run_id"])
            for a, b in (("K1-01", "NAT-FT4"), ("K1-08", "NAT-R0")):
                for i in "AB":
                    if (a, i) in conds and (b, i) in conds and conds[a, i]["action_seq_sha"] != conds[b, i]["action_seq_sha"]:
                        x, y = conds[a, i], conds[b, i]
                        same = all(x[k] == y[k] for k in ENDPOINTS) and abs(arm[x["run_id"]] - arm[y["run_id"]]) <= 0.01
                        (g.setdefault("late_action_divergence", []) if same else g["reconstruction_failures"]).append(f"{key}|{a}|{i}")
            for z in ("K1-09", "K1-10"):
                if (z, "A") in conds and (z, "B") in conds and conds[z, "A"]["action_seq_sha"] != conds[z, "B"]["action_seq_sha"]:
                    g["zero_leak_failures"].append(f"{key}|{z}")
        g["passed"] = (g["engineering_passed"] and g["geometry_passed"] and not g["identity_failures"]
                       and not g["reconstruction_failures"] and not g["zero_leak_failures"])
        gates[p] = g
    with (OUT / "records.jsonl").open("w") as f:
        for r in uniq:
            f.write(json.dumps(r) + "\n")
    jdump(OUT / "gates.json", gates)
    print(json.dumps({"records": len(uniq), "duplicates": dup, "gates": {p: g["passed"] for p, g in gates.items()}}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prefix", "plan", "engineering", "geometry_gate", "worker", "collect"])
    ap.add_argument("--family")
    ap.add_argument("--pair")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only-blocks")
    ap.add_argument("--sub", type=int, default=0)
    ap.add_argument("--subn", type=int, default=1)
    a = ap.parse_args()
    {"prefix": phase_prefix, "plan": phase_plan, "engineering": phase_engineering, "geometry_gate": phase_geometry_gate,
     "worker": phase_worker, "collect": phase_collect}[a.phase](a)
