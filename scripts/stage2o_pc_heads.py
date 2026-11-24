"""Registered open-loop P/C factorization and true pre-W_O head screening.

Only FT4 receives patches; R0 supplies paired reference tensors. No training,
parameter mutation, generated control, adaptive head selection or closed loop.
"""

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from stage2m_cross_attention import (
    Observer, denoise, make_inputs, make_prefix, parameter_audit, rms, sha,
    tensor_sha, velocity_tensor, write,
)
from stage2n_cross_checkpoint import (
    check_prior, input_hashes, json_line, load_prior, matched_random_message,
    norm_blocks, opposite, prefix_diagnostics, random_seed_for,
)

ROLES = ("R0", "FT4")
GOALS = ("left", "front")
MODES = ("self_pc", "old_p_same", "old_c_same", "old_pc_same", "old_p_wrong",
         "old_c_wrong", "old_pc_wrong", "old_m_same", "old_m_wrong", "random_pc")
HEAD_MODES = ("head_same", "head_wrong", "head_random")


def head_span(head, heads, width):
    if width % heads or not 0 <= head < heads:
        raise ValueError("Invalid actual pre-projection head coordinates")
    return slice(head*(width//heads), (head+1)*(width//heads))


def expected_counts():
    return {"baseline": 144, "factor_patches": 2160, "head_patches": 3240,
            "patches": 5400, "norm_records": 2592, "reference_denoise": 120}


def readout(probability, content, heads, torch):
    """Native GQA expansion; PC uses actual heads, not post-W_O coordinates."""
    batch, length, kv_heads, dim = content.shape
    if heads % kv_heads or probability.shape != (batch, heads, 50, length):
        raise ValueError("P/C/head/token coordinate mismatch")
    expanded = content[:, :, :, None, :].expand(batch, length, kv_heads, heads//kv_heads, dim)
    expanded = expanded.reshape(batch, length, heads, dim)
    return torch.matmul(probability, expanded.permute(0, 2, 1, 3)).permute(
        0, 2, 1, 3).reshape(batch, 50, heads*dim)


class PCObserver(Observer):
    def __init__(self, policy, torch):
        self.sites = (1, 3, 9)
        self.evidence = {}
        self.expected_u = {}
        self.witness = {}
        self.random_seed = None
        super().__init__(policy, torch)
        for row in self.layer_map:
            if row["vlm_layer"] in self.sites:
                module = self.expert.lm_expert.layers[row["expert_layer"]].self_attn.o_proj
                self.handles.append(module.register_forward_pre_hook(self._u_witness(row["vlm_layer"])))

    def begin(self, patch=None):
        self.evidence, self.expected_u, self.witness = {}, {}, {}
        super().begin(patch)

    def _u_witness(self, index):
        def hook(module, inputs):
            if self.active and index in self.expected_u:
                error = float((inputs[0].float().cpu()-self.expected_u[index]).abs().max())
                if error > 1e-5:
                    raise AssertionError("Actual o_proj input differs from injected pre-head readout")
                self.witness[index] = error
        return hook

    def _message_hook(self, index):
        def hook(module, inputs, output):
            if not self.active:
                return None
            self.record.setdefault(index, {})["M"] = output.detach().clone().float().cpu()
            if self.patch is not None and self.patch["layer"] == index and self.patch["mode"].startswith("old_m"):
                source = self.patch["donor"]["M"].to(device=output.device, dtype=output.dtype).clone()
                if source.shape != output.shape:
                    raise ValueError("Post-projection message shape changed")
                self.evidence[index] = {"message_delta_rms": rms(source.float()-output.float()),
                                        "patch_interface": "post_W_O_M"}
                return source
            return None
        return hook

    def _eager(self, mask, batch_size, head_dim, query, key, value):
        # Retain the already verified native P/C reconstruction path.
        output = super()._eager(mask, batch_size, head_dim, query, key, value)
        layer = self.current_layer
        if not self.active or layer not in self.sites:
            return output
        record = self.record[layer]
        record["mask"] = mask.detach().cpu().clone()
        if output.shape[-1] != self.expert.num_attention_heads*head_dim:
            raise AssertionError("True pre-projection width is not heads * actual head_dim")
        record["head_dim"] = head_dim
        patch = self.patch
        if patch is None or patch["layer"] != layer or patch["mode"].startswith("old_m"):
            return output
        torch = self.torch
        mode, donor, heads = patch["mode"], patch.get("donor", record), self.expert.num_attention_heads
        # Token columns and legal masks must align; do not renormalize donor P.
        if donor["P"].shape != record["P"].shape or not torch.equal(donor["mask"], record["mask"]):
            raise ValueError("Donor/recipient token or mask coordinates do not align")
        if mode == "self_pc":
            result = readout(record["P"].to(output.device, value.dtype),
                             record["C"].to(output.device, value.dtype), heads, torch)
        elif mode.startswith("old_p_"):
            result = readout(donor["P"].to(output.device, value.dtype), value, heads, torch)
        elif mode.startswith("old_c_"):
            result = readout(record["P"].to(output.device, value.dtype),
                             donor["C"].to(output.device, value.dtype), heads, torch)
        else:
            result = donor["PC"].to(output.device, output.dtype).clone()
            if mode.startswith("head_"):
                span = head_span(patch["head"], heads, output.shape[-1])
                selected = output.clone()
                selected[..., span] = result[..., span]
                result = selected
            if mode in ("random_pc", "head_random"):
                planned, norm = matched_random_message(
                    output, result, heads, patch["random_seed"], output.dtype, torch)
                norm["block_semantics"] = "true pre-W_O readout heads; GQA expansion retained"
                self.evidence[layer] = {"norm": norm}
                result = planned.to(output.device, output.dtype)
        if result.shape != output.shape:
            raise ValueError("Injected PC shape differs from native readout")
        if mode.startswith("head_"):
            span = head_span(patch["head"], heads, output.shape[-1])
            unchanged = torch.ones(output.shape[-1], device=output.device, dtype=torch.bool)
            unchanged[span] = False
            if not torch.equal(result[..., unchanged], output[..., unchanged]):
                raise AssertionError("Head patch leaked into other actual head readouts")
        self.expected_u[layer] = result.detach().float().cpu().clone()
        if mode in ("old_pc_same", "head_same"):
            self.evidence.setdefault(layer, {})["correct_norm"] = {
                "effective_block_l2": norm_blocks(result.float()-output.float(), heads).tolist(),
                "block_semantics": "actual pre-W_O head delta norms"}
        self.evidence.setdefault(layer, {}).update(
            readout_delta_rms=rms(result.float()-output.float()),
            patch_interface="pre_W_O_true_head_readout", heads=heads, head_dim=head_dim,
            untouched_heads_max_abs=0.0 if mode.startswith("head_") else None)
        return result


def patch_query(models, observers, prepared, x_t, timestep, layer, mode, donor, head=None, seed=0):
    descriptor = {"layer": layer, "mode": mode, "donor": donor, "head": head, "random_seed": seed}
    observer = observers["FT4"]
    output, _ = denoise(models["FT4"], observer, prepared, x_t, timestep, record=True, patch=descriptor)
    evidence = dict(observer.evidence.get(layer, {}))
    if not mode.startswith("old_m"):
        if layer not in observer.witness:
            raise AssertionError("Pre-projection intervention witness did not execute")
        evidence["actual_input_witness_max_abs"] = observer.witness[layer]
    return velocity_tensor(output), evidence


def head_observations(common, captures, groups, heads):
    """Descriptive head tensors are not causal evidence by themselves."""
    rows = []
    for layer in (1, 3, 9):
        for role in ROLES:
            a, b = (captures[role][goal][layer] for goal in GOALS)
            width = a["PC"].shape[-1]
            for head in range(heads):
                span = head_span(head, heads, width)
                rows.append({**common, "model": role, "layer": layer, "head": head,
                             "head_dim": a["head_dim"],
                             "P_LF_rms": rms(a["P"][:, head, :10]-b["P"][:, head, :10]),
                             "U_LF_rms": rms(a["PC"][:, :10, span]-b["PC"][:, :10, span]),
                             "source_mass": {goal: {name: float(captures[role][goal][layer]["P"][:, head, :10, lo:hi].sum(-1).mean())
                                                   for name, (lo, hi) in groups.items()} for goal in GOALS}})
    return rows


def run_engineering(args, cfg, report, models, observers, initial, prefixes, noise, prior, guard, torch):
    apply_protocol(report, cfg)
    prepared = prefixes(initial(0))
    x_t, timestep = noise(report["seeds"][0]), torch.ones(1, device="cuda")
    captures, velocities, checks = {}, {}, []
    common = {"anchor_id": "i00_s2026102801_f00", "state_index": 0, "noise_seed": 2026102801,
              "flow_index": 0, "time": 1.0, "x_sha256": tensor_sha(x_t),
              "state_sha256": report["physical_state_sha256"]["0"]}
    for role in ROLES:
        captures[role], velocities[role] = {}, {}
        for goal in GOALS:
            native, _ = denoise(models[role], observers[role], prepared[role][goal][1], x_t, timestep)
            value, capture = denoise(models[role], observers[role], prepared[role][goal][1], x_t, timestep, record=True)
            error = float((native-value).abs().max())
            check_prior(common, role, goal, velocity_tensor(value), prior, 1e-5)
            if error > 1e-5:
                raise AssertionError("Recording changed native velocity")
            checks.append({"kind": "record", "model": role, "goal": goal, "max_abs": error})
            captures[role][goal], velocities[role][goal] = capture, velocity_tensor(value)
            report["counts"]["baseline"] += 1
    heads = observers["FT4"].expert.num_attention_heads
    if heads != cfg["expected_heads"]:
        raise ValueError("Real attention-head count differs from preregistration")
    # Exercise all factor arms and each true head before the full matrix.
    for goal in GOALS:
        for layer in report["sites"]:
            for mode in MODES:
                donor_goal = opposite(goal) if mode.endswith("wrong") else goal
                donor = captures["FT4"][goal][layer] if mode == "self_pc" else captures["R0"][donor_goal][layer]
                value, evidence = patch_query(models, observers, prepared["FT4"][goal][1],
                                              x_t, timestep, layer, mode, donor, seed=2026102901)
                if mode == "self_pc":
                    error = float((value-velocities["FT4"][goal]).abs().max())
                    if error > 1e-5:
                        raise AssertionError("P/C self-reconstruction changes velocity")
                    checks.append({"kind": mode, "goal": goal, "layer": layer, "max_abs": error})
                report["counts"]["patches"] += 1
            for head in range(heads):
                value, evidence = patch_query(models, observers, prepared["FT4"][goal][1], x_t, timestep,
                                              layer, "head_same", captures["FT4"][goal][layer], head)
                error = float((value-velocities["FT4"][goal]).abs().max())
                if error > 1e-5:
                    raise AssertionError("Individual head self-replacement changes velocity")
                checks.append({"kind": "head_self", "goal": goal, "layer": layer, "head": head, "max_abs": error})
                report["counts"]["patches"] += 1
            guard()
    gate = {"passed": True, "checks": checks, "config_sha256": report["config_sha256"],
            "script_sha256": report["script_sha256"], "sources": report["sources"],
            "heads": heads, "head_dim": captures["FT4"]["left"][1]["head_dim"],
            "no_generated_actions": True}
    write(args.output/"engineering.json", gate)


def apply_protocol(report, cfg):
    report["post_projection_hidden_width"] = report.pop("block_width")*report.pop("block_count")
    report.pop("block_semantics")
    report.update(scope="step4 P/C factorial and true pre-W_O head screening; open loop only",
                  script_sha256=sha(__file__), expected_counts=expected_counts(),
                  head_screen_seeds=[2026102801], factor_modes=list(MODES), head_modes=list(HEAD_MODES),
                  head_semantics="actual GQA-expanded readout head before output projection, not post-O coordinates")
    report["counts"].update(factor_patches=0, head_patches=0, norm_records=0, head_observations=0)


def run_analysis(args, cfg, report, models, observers, initial, prefixes, noise, prior, euler_integrate, guard, torch):
    apply_protocol(report, cfg)
    gate_path = args.output.parent/"engineering/engineering.json"
    gate = json.loads(gate_path.read_text())
    if not gate["passed"] or any(gate[k] != report[k] for k in ("config_sha256", "script_sha256", "sources")):
        raise ValueError("Engineering gate or provenance differs")
    write(args.output/"engineering.json", gate)
    report["engineering_source"] = {"path": str(gate_path), "sha256": sha(gate_path)}
    heads = observers["FT4"].expert.num_attention_heads
    report["actual_heads"], report["actual_head_dim"] = heads, gate["head_dim"]
    # Prior post-W_O interventions are positive implementation controls, not re-selection.
    if sha(cfg["prior_patch_path"]) != cfg["prior_patch_sha256"]:
        raise ValueError("Prior message-patch source changed")
    old_rows = [json.loads(line) for line in Path(cfg["prior_patch_path"]).read_text().splitlines()]
    old_lookup = {(r["anchor_id"], r["recipient"], r["layer"], r["mode"]):r["velocity"]
                  for r in old_rows if r["model"] == "FT4"}
    max_prior, max_m_prior = 0.0, 0.0
    with gzip.open(args.output/"norms.jsonl.gz", "wt", encoding="utf-8") as norms:
        for state in report["indices"]:
            prepared = prefixes(initial(state))
            for seed_position, seed in enumerate(report["seeds"]):
                references, counter = {}, [0]
                reference_goal = GOALS[(state+seed_position)%2]
                def reference_velocity(action_x, timestep):
                    index = counter[0]
                    if index in report["flow_indices"]:
                        references[index] = (action_x.detach().clone(), timestep.detach().clone())
                    value, _ = denoise(models["R0"], observers["R0"], prepared["R0"][reference_goal][1], action_x, timestep)
                    counter[0] += 1
                    report["counts"]["reference_denoise"] += 1
                    return value
                euler_integrate(reference_velocity, noise(seed), 10)
                if set(references) != set(report["flow_indices"]):
                    raise ValueError("Reference flow grid changed")
                for flow in report["flow_indices"]:
                    x_t, timestep = references[flow]
                    anchor = f"i{state:02d}_s{seed}_f{flow:02d}"
                    common = {"anchor_id": anchor, "state_index": state, "noise_seed": seed,
                              "flow_index": flow, "time": float(timestep[0]), "reference_goal": reference_goal,
                              "x_sha256": tensor_sha(x_t), "state_sha256": report["physical_state_sha256"][str(state)]}
                    captures = {}
                    for role in ROLES:
                        captures[role] = {}
                        for goal in GOALS:
                            output, capture = denoise(models[role], observers[role], prepared[role][goal][1], x_t, timestep, record=True)
                            velocity = velocity_tensor(output)
                            max_prior = max(max_prior, check_prior(common, role, goal, velocity, prior, 1e-5))
                            captures[role][goal] = capture
                            append_row = {**common, "model": role, "goal": goal, "velocity": velocity.tolist()}
                            with (args.output/"baseline.jsonl").open("a") as stream:
                                json_line(stream, append_row)
                            report["counts"]["baseline"] += 1
                    for row in head_observations(common, captures, prepared["FT4"]["left"][1][2], heads):
                        with (args.output/"heads.jsonl").open("a") as stream:
                            json_line(stream, row)
                        report["counts"]["head_observations"] += 1
                    for layer in report["sites"]:
                        for goal in GOALS:
                            jobs = [(mode, None) for mode in MODES]
                            if seed == 2026102801:
                                jobs += [(mode, head) for head in range(heads) for mode in HEAD_MODES]
                            for mode, head in jobs:
                                donor_goal = opposite(goal) if mode.endswith("wrong") else goal
                                donor_role = "FT4" if mode == "self_pc" else "R0"
                                donor = captures[donor_role][donor_goal][layer]
                                random_seed = random_seed_for(anchor, "FT4", goal+str(head), layer, cfg["random_seed_salt"])
                                value, evidence = patch_query(models, observers, prepared["FT4"][goal][1],
                                                              x_t, timestep, layer, mode, donor, head, random_seed)
                                if mode.startswith("old_m"):
                                    previous_mode = "cross_wrong" if mode.endswith("wrong") else "cross_same"
                                    old = torch.tensor(old_lookup[(anchor, goal, layer, previous_mode)])
                                    error = float((value-old).abs().max())
                                    if error > 1e-5:
                                        raise AssertionError("Prior full-message effect did not reproduce")
                                    max_m_prior = max(max_m_prior, error)
                                row = {**common, "model": "FT4", "recipient": goal, "layer": layer,
                                       "mode": mode, "head": head, "donor_goal": donor_goal,
                                       "velocity": value.tolist(), **{k:v for k,v in evidence.items()
                                                                      if k not in ("norm", "correct_norm")}}
                                with (args.output/"patches.jsonl").open("a") as stream:
                                    json_line(stream, row)
                                if "norm" in evidence or "correct_norm" in evidence:
                                    json_line(norms, {**common, "recipient": goal, "layer": layer,
                                                     "mode": mode, "head": head,
                                                     **evidence.get("norm", evidence.get("correct_norm"))})
                                    report["counts"]["norm_records"] += 1
                                report["counts"]["patches"] += 1
                                report["counts"]["head_patches" if head is not None else "factor_patches"] += 1
                    guard()
    for key, expected in expected_counts().items():
        if report["counts"][key] != expected:
            raise AssertionError(f"Incomplete registered matrix: {key}")
    report["prior_velocity_max_abs"], report["prior_full_message_max_abs"] = max_prior, max_m_prior



def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    indices = cfg.get("indices", cfg.get("development_indices"))
    seeds = cfg.get("seeds", cfg.get("noise_seeds"))
    flows = cfg.get("flow_indices", cfg.get("flow_step_indices"))
    sites = cfg["sites"]
    if indices != [0, 1, 2, 3] or seeds != [2026102801, 2026102802, 2026102803] or flows != [0, 4, 8]:
        raise ValueError("P/C screening accepts only the original sealed input matrix")
    if sorted(sites) != [1, 3, 9] or len(sites) != 3:
        raise ValueError("Only the two selected sites and the predeclared control site are allowed")
    if cfg.get("candidate_sites", [3, 9]) != [3, 9] or cfg.get("control_site", 1) != 1:
        raise ValueError("Unexpected selected/control site definition")
    if tuple(cfg.get("goals", GOALS)) != GOALS or tuple(cfg.get("modes", MODES)) != MODES:
        raise ValueError("The registered goals or intervention modes changed")
    tolerance = cfg.get("identity_tolerance", 1e-5)
    if tolerance != 1e-5:
        raise ValueError("The registered engineering tolerance cannot be changed")
    report = {
        "status": "initializing", "phase": args.phase,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "step4 P/C factorization and true pre-W_O head screening; no parameter mutation or generated actions",
        "indices": indices, "seeds": seeds, "flow_indices": flows, "layers": sites, "sites": sites,
        "candidate_sites": [3, 9], "control_site": 1, "roles": list(ROLES),
        "goals": list(GOALS), "modes": list(MODES), "identity_tolerance": tolerance,
        "counts": {"baseline": 0, "patches": 0, "norm_records": 0,
                   "reference_denoise": 0, "engineering_self_queries": 0},
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "configuration": cfg,
        "random_seed_salt": cfg.get("random_seed_salt", cfg.get("random_seed", 2026102903)),
        "effective_phase_inputs": {"indices": [indices[0]], "seeds": [seeds[0]], "flow_indices": [0]}
            if args.phase == "engineering" else {"indices": indices, "seeds": seeds, "flow_indices": flows},
    }
    observers, env = {}, None
    try:
        sys.path.insert(0, str(args.runtime_scripts.resolve()))
        import numpy as np
        import torch
        import lerobot.policies as policies
        from lerobot.utils import constants
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        from lerobot.policies.common.flow_matching import euler_integrate
        from robosuite.utils.transform_utils import quat2axisangle
        import stage1_r0_eval as evaluation
        import stage2m_cross_attention as prior_helper
        from stage1_smoke import DEFAULT_SCENE
        from stage2k_entry_handoff import _fresh_observation
        from stage2k_response_swap import verify_processors

        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("The registered inference runtime requires CUDA")
        root_cfg = json.loads(Path(cfg["base_config"]).read_text(encoding="utf-8"))
        if not set(indices) <= set(root_cfg["evaluation"]["development_indices"]):
            raise ValueError("These states are not registered development states")
        prior_path = Path(cfg["prior_baseline_path"])
        if sha(prior_path) != cfg["prior_baseline_sha256"]:
            raise ValueError("Prior baseline evidence bytes changed")
        prior = load_prior(prior_path)
        report["prior_baseline_path"] = str(prior_path)
        report["prior_baseline_sha256"] = sha(prior_path)
        expected_prior = {(i, s, f, role, goal) for i in indices for s in seeds for f in flows
                          for role in ROLES for goal in GOALS}
        if set(prior) != expected_prior:
            raise ValueError("Prior baseline does not contain exactly the registered old-condition matrix")
        candidates_path = Path(cfg["prior_candidates_path"])
        if sha(candidates_path) != cfg["prior_candidates_sha256"]:
            raise ValueError("Prior candidate-selection evidence changed")
        if json.loads(candidates_path.read_text(encoding="utf-8"))["selected"] != [3, 9]:
            raise ValueError("The two fixed sites do not match the prior selection")
        report["prior_candidates_sha256"] = sha(candidates_path)
        model_paths = {"R0": Path(cfg["teacher"]), "FT4": Path(cfg["student"])}
        report["processor_hashes"] = verify_processors(model_paths["R0"], model_paths["FT4"])
        report["model_sha256"] = {role: sha(path / "model.safetensors") for role, path in model_paths.items()}
        for role, key in (("R0", "teacher"), ("FT4", "student")):
            if report["model_sha256"][role] != cfg[key + "_model_sha256"]:
                raise ValueError(f"Model identity changed: {role}")
        models = {role: evaluation.load_policy(path) for role, path in model_paths.items()}
        preprocessors = {}
        for role, policy in models.items():
            policy.eval()
            policy.config.n_action_steps = 10
            policy.config.num_steps = 10
            if (policy.config.adapt_to_pi_aloha or policy.model._rtc_enabled()
                    or policy.config.n_obs_steps != 1 or not policy.model.config.use_cache):
                raise ValueError("Unexpected runtime action/observation/cache settings")
            if policy.config.chunk_size != 50 or policy.config.action_feature.shape[0] != 7:
                raise ValueError("Unexpected action shape")
            observers[role] = PCObserver(policy, torch)
            preprocessors[role], _ = policies.make_pre_post_processors(
                policy.config, pretrained_path=model_paths[role])
        report["layer_map"] = {role: observer.layer_map for role, observer in observers.items()}
        report["ca_layers"] = {role: observer.ca_layers for role, observer in observers.items()}
        if any(not set(sites) <= set(observer.ca_layers) for observer in observers.values()):
            raise ValueError("Registered intervention site is not an actual CA block")
        actual_control = min(x for x in observers["R0"].ca_layers if x not in (3, 9))
        if actual_control != cfg.get("control_site", 1):
            raise ValueError("The registered smallest noncandidate CA control changed")
        report["parameter_audit"] = parameter_audit(models["R0"], models["FT4"], torch)
        heads = [observer.expert.num_attention_heads for observer in observers.values()]
        widths = [policy.model.vlm_with_expert.expert_hidden_size for policy in models.values()]
        if len(set(heads)) != 1 or len(set(widths)) != 1 or widths[0] % heads[0]:
            raise ValueError("Post-output-projection bookkeeping coordinates are incompatible")
        report["block_count"], report["block_width"] = heads[0], widths[0] // heads[0]
        report["block_semantics"] = "contiguous post-output-projection coordinates; not original attention heads"
        report["model_configuration"] = {
            role: {name: getattr(policy.config, name, None) for name in
                   ("attention_mode", "num_vlm_layers", "num_expert_layers", "self_attn_every_n_layers",
                    "train_expert_only", "train_state_proj", "prefix_length", "pad_language_to",
                    "num_steps", "n_action_steps", "chunk_size", "max_action_dim", "use_cache")}
            for role, policy in models.items()
        }
        report["sources"] = {}
        for name, source in (
                ("evaluation", evaluation.__file__), ("stage2m_helper", prior_helper.__file__),
                ("model", sys.modules[type(models["R0"].model).__module__].__file__),
                ("expert", sys.modules[type(models["R0"].model.vlm_with_expert).__module__].__file__)):
            report["sources"][name] = {"path": source, "sha256": sha(source)}
        for source, expected in cfg.get("expected_model_sources", {}).items():
            if sha(source) != expected:
                raise ValueError(f"Model source changed: {source}")
        tasks, paths, _, physical_states = evaluation.task_resources(root_cfg.get("scene", DEFAULT_SCENE))
        instructions = {goal: tasks[goal].language for goal in GOALS}
        report["instructions"] = instructions
        env = evaluation.make_env(paths["front"])
        scene_xml = evaluation.canonical_scene(paths["front"], root_cfg["seed"])
        report["scene_xml_sha256"] = hashlib.sha256(scene_xml.encode()).hexdigest()

        def guard():
            if time.monotonic() - started > args.max_minutes * 60:
                raise TimeoutError("Registered walltime budget")
            size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if size > args.max_output_bytes or shutil.disk_usage(args.output).free < 2 * 1024**3:
                raise RuntimeError("Registered disk/output budget; no data are deleted")
            write(args.output / "progress.json",
                  {**report, "elapsed_seconds": time.monotonic() - started})

        def initial(index):
            evaluation.reset_common(env, physical_states["front"][index], root_cfg["seed"] + index,
                                    root_cfg["evaluation"]["settle_steps"], scene_xml)
            observation = _fresh_observation(env)
            report.setdefault("physical_state_sha256", {})[str(index)] = hashlib.sha256(
                np.asarray(env.get_sim_state()).tobytes()).hexdigest()
            return observation

        def prefixes(observation):
            prepared, hashes = {}, {}
            for role in ROLES:
                prepared[role], hashes[role] = {}, {}
                for goal in GOALS:
                    inputs = make_inputs(models[role], preprocessors[role], observation,
                                         instructions[goal], torch, np, quat2axisangle, constants)
                    hashes[role][goal] = input_hashes(inputs)
                    prepared[role][goal] = (inputs, make_prefix(models[role], inputs, make_att_2d_masks))
            report.setdefault("prefix_input_hashes", []).append(hashes)
            for goal in GOALS:
                if hashes["R0"][goal] != hashes["FT4"][goal]:
                    raise AssertionError(f"Cross-checkpoint preprocessed input differs: {goal}")
            for name in ("images", "image_masks", "state_before_projection"):
                if hashes["R0"]["left"][name] != hashes["R0"]["front"][name]:
                    raise AssertionError(f"Physical input changed across instructions: {name}")
            prefix_diagnostics(prepared, report, tolerance, torch)
            return prepared

        def noise(seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return torch.randn((1, 50, models["R0"].config.max_action_dim), generator=generator,
                               dtype=torch.float32, device="cpu").cuda()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.phase == "engineering":
                run_engineering(args, cfg, report, models, observers, initial, prefixes,
                                noise, prior, guard, torch)
            else:
                run_analysis(args, cfg, report, models, observers, initial, prefixes,
                             noise, prior, euler_integrate, guard, torch)
        report.update(
            status="complete", elapsed_seconds=time.monotonic() - started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            cache_isolation={"object_unique": True, "tensor_storage_unique": True,
                             "deep_copy_per_query": True},
            attention_readout_reconstruction_max_abs={
                role: observer.readout_error_max for role, observer in observers.items()},
            no_followup_launched=True,
            scope_exclusions=["no W_O parameter intervention", "no closed loop",
                              "no parameter update", "no adaptive head selection", "no extra seeds"],
        )
        report["files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in args.output.iterdir() if p.is_file()
            and p.name not in ("progress.json", "manifest.json", "COMPLETE.json")
        }
        write(args.output / "manifest.json", report)
        write(args.output / "COMPLETE.json", report)
        write(args.output / "progress.json", report)
    except Exception as error:
        report.update(status="failed", error=repr(error),
                      elapsed_seconds=time.monotonic() - started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write(args.output / "FAILED.json", report)
        write(args.output / "progress.json", report)
        raise
    finally:
        for observer in observers.values():
            observer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-scripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("engineering", "analysis"), required=True)
    parser.add_argument("--max-minutes", type=float, default=45)
    parser.add_argument("--max-output-bytes", type=int, default=536870912)
    main(parser.parse_args())
