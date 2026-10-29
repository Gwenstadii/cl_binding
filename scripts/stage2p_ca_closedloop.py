"""Fixed-window closed-loop CA message interventions; no parameter update.

Native execution and post-window execution use the recipient throughout. Donors
read the intervention branch's CURRENT observation, action draft and flow time.
The saved four-predicate trace is the behavioral evidence, not readout distance.
"""

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from stage2m_cross_attention import make_prefix, rms, sha, tensor_sha, write
from stage2n_cross_checkpoint import matched_random_message, norm_blocks, random_seed_for
from stage2k_response_swap import verify_processors


GOALS = ("left", "front")
ALL_GOALS = ("front", "left", "right", "back")
MODES = {"FT4": ("native", "cross_same", "cross_wrong", "random_delta"),
         "R0": ("native", "cross_same", "same_model_wrong", "random_delta")}


def episode_jobs():
    """Fixed development matrix; never select episodes by an observed outcome."""
    jobs = [(role, mode, goal, index, seed) for role in ("FT4", "R0")
            for mode in MODES[role] for goal in GOALS for index in (0, 1)
            for seed in (2026102801, 2026102802)]
    jobs += [("FT4", "native", "right", index, seed) for index in (0, 1)
             for seed in (2026102801, 2026102802)]
    return jobs


def replan_seed(seed, replan_index):
    if replan_index < 0 or replan_index >= 40:
        raise ValueError("Unexpected replanning index")
    return seed + 1000003*replan_index


def donor_for(role, mode, goal):
    if mode == "native":
        return None, None
    other = "FT4" if role == "R0" else "R0"
    if mode in ("cross_same", "random_delta"):
        return other, goal
    if mode == "cross_wrong" and role == "FT4":
        return "R0", "front" if goal == "left" else "left"
    if mode == "same_model_wrong" and role == "R0":
        return "R0", "front" if goal == "left" else "left"
    raise ValueError("Unregistered donor condition")


def dump_line(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")


class LayerMessageHook:
    """Only the real o_proj output; record/patch is active during denoising."""

    def __init__(self, policy, site, torch):
        self.torch, self.site = torch, site
        expert = policy.model.vlm_with_expert
        layers = expert.get_model_layers([expert.get_vlm_model().text_model, expert.lm_expert])
        layer = layers[1][site]
        if (layer is None or "cross" not in expert.attention_mode
                or (expert.self_attn_every_n_layers > 0 and site % expert.self_attn_every_n_layers == 0)):
            raise ValueError("Registered site is not a real Cross-Attention layer")
        self.expert_layer = next(i for i, value in enumerate(expert.lm_expert.layers) if value is layer)
        self.module = layer.self_attn.o_proj
        self.active = False
        self.source = None
        self.native_message, self.message_dtype = None, None
        self.seen, self.witness_seen, self.witness_max_abs = 0, 0, 0.0
        self.handles = [self.module.register_forward_hook(self._hook),
                        self.module.register_forward_hook(self._witness)]

    def start(self, source=None):
        self.source = source
        self.native_message, self.message_dtype = None, None
        self.expected = None
        self.seen, self.witness_seen, self.witness_max_abs = 0, 0, 0.0
        self.active = True

    def _hook(self, module, inputs, output):
        if not self.active:
            return None
        self.seen += 1
        self.native_message = output.detach().clone()
        self.message_dtype = output.dtype
        if self.source is not None:
            if self.source.shape != output.shape:
                raise ValueError("Whole-message shape changed")
            patched = self.source.to(device=output.device, dtype=output.dtype).clone()
            self.expected = patched.detach().clone()
            return patched
        self.expected = output.detach().clone()
        return None

    def _witness(self, module, inputs, output):
        if self.active:
            self.witness_seen += 1
            self.witness_max_abs = max(self.witness_max_abs,
                                      float((output.float()-self.expected.float()).abs().max()))
        return None

    def finish(self):
        self.active, self.source = False, None
        if self.seen != 1 or self.witness_seen != 1 or self.native_message is None:
            raise AssertionError("Message hook did not execute exactly once in action denoising")
        if self.witness_max_abs > 1e-5:
            raise AssertionError("Actually injected M differs from planned native-cast M")
        return self.native_message, self.message_dtype, self.witness_max_abs

    def close(self):
        self.active = False
        for handle in self.handles:
            handle.remove()


class ClosedLoopRuntime:
    def __init__(self, cfg, report, torch, np, policies, evaluation, constants,
                 quat2axisangle, make_att_2d_masks, euler_integrate):
        self.cfg, self.report, self.torch, self.np = cfg, report, torch, np
        self.constants, self.quat2axisangle = constants, quat2axisangle
        self.make_masks, self.integrate = make_att_2d_masks, euler_integrate
        self.models, self.pre, self.post, self.hooks = {}, {}, {}, {}
        paths = {"R0": Path(cfg["teacher"]), "FT4": Path(cfg["student"])}
        report["processor_hashes"] = verify_processors(paths["R0"], paths["FT4"])
        report["model_sha256"] = {role: sha(path/"model.safetensors") for role, path in paths.items()}
        if report["model_sha256"] != {"R0":cfg["teacher_model_sha256"], "FT4":cfg["student_model_sha256"]}:
            raise ValueError("Checkpoint identity differs from registered source")
        for role, path in paths.items():
            policy = evaluation.load_policy(path)
            policy.eval()
            policy.config.n_action_steps = 10
            policy.config.num_steps = 10
            if (policy.config.adapt_to_pi_aloha or policy.model._rtc_enabled()
                    or policy.config.n_obs_steps != 1 or not policy.model.config.use_cache
                    or policy.config.chunk_size != 50 or policy.config.action_feature.shape[0] != 7):
                raise ValueError("Unexpected action/observation/cache settings")
            self.models[role] = policy
            self.pre[role], self.post[role] = policies.make_pre_post_processors(policy.config, pretrained_path=path)
            self.hooks[role] = LayerMessageHook(policy, cfg["site"], torch)
        report["layer_map"] = {r:{"site":cfg["site"],"expert_layer":h.expert_layer,
                                   "o_in_features":h.module.in_features,"o_out_features":h.module.out_features}
                               for r,h in self.hooks.items()}
        report["random_block_semantics"] = "15 contiguous post-W_O output coordinate blocks; NOT actual attention heads"
        if any(h.module.out_features != 720 for h in self.hooks.values()):
            raise ValueError("Registered post-output-projection coordinate width changed")
        self.counts = {"denoise_calls_by_model":{"R0":0,"FT4":0}, "patch_flows":0,
                       "donor_denoise_calls":0,"recipient_denoise_calls":0,"prefix_builds":0,
                       "integrator_velocity_calls":0,"extra_random_recipient_calibration_calls":0,
                       "official_engineering_flow_calls":0,"engineering_self_replacements":0}
        self.cache_checks, self.max_witness = 0, 0.0

    def batch(self, role, observation, instruction):
        torch, np = self.torch, self.np
        raw = {"task":instruction,"observation.state":torch.from_numpy(np.concatenate((
            observation["robot0_eef_pos"],self.quat2axisangle(observation["robot0_eef_quat"].copy()),
            observation["robot0_gripper_qpos"])).astype(np.float32))}
        for key, name in (("observation.images.image","agentview_image"),
                          ("observation.images.image2","robot0_eye_in_hand_image")):
            raw[key] = torch.from_numpy(np.ascontiguousarray(observation[name][::-1,::-1])).permute(2,0,1).float()/255
        return self.pre[role](copy.deepcopy(raw))

    def prefix(self, role, observation, instruction):
        policy = self.models[role]
        batch = self.batch(role, observation, instruction)
        images, masks = policy.prepare_images(batch)
        inputs = (images,masks,batch[self.constants.OBS_LANGUAGE_TOKENS],
                  batch[self.constants.OBS_LANGUAGE_ATTENTION_MASK],policy.prepare_state(batch))
        prefix = make_prefix(policy, inputs, self.make_masks)
        self.counts["prefix_builds"] += 1
        return prefix

    def noise(self, seed, replan_index):
        torch = self.torch
        generator = torch.Generator(device="cpu").manual_seed(replan_seed(seed,replan_index))
        value = torch.randn((1,50,self.models["FT4"].config.max_action_dim),
                            generator=generator,dtype=torch.float32,device="cpu")
        return value.cuda()

    def query(self, role, prefix, x_t, timestep, source=None, capture=False, donor=False):
        pad, original, _ = prefix
        cache = copy.deepcopy(original)
        if cache is original or any(a.keys.data_ptr()==b.keys.data_ptr()
                                   or a.values.data_ptr()==b.values.data_ptr()
                                   for a,b in zip(cache.layers,original.layers)):
            raise AssertionError("Donor/recipient KV cache alias")
        self.cache_checks += 1
        hook = self.hooks[role]
        if capture or source is not None:
            hook.start(source)
        try:
            velocity = self.models[role].model.denoise_step(pad,cache,x_t.clone(),timestep.clone())
            if not self.torch.isfinite(velocity).all():
                raise ValueError("Nonfinite generated velocity")
            captured = hook.finish() if capture or source is not None else (None,None,0.0)
        finally:
            hook.active, hook.source = False, None
        if any(layer.keys.shape[-2] != pad.shape[1] for layer in cache.layers):
            raise AssertionError("Denoise did not restore cropped prefix")
        self.counts["denoise_calls_by_model"][role] += 1
        self.counts["donor_denoise_calls" if donor else "recipient_denoise_calls"] += 1
        self.max_witness = max(self.max_witness,captured[2])
        return velocity.detach().clone(), captured

    def physical_chunk(self, role, chunk):
        physical = self.post[role](chunk.clone()).float().cpu().numpy()
        if physical.shape != (1,50,7) or not self.np.isfinite(physical).all():
            raise ValueError("Unexpected postprocessed action shape/values")
        return physical[0]

    def sample(self, role, mode, goal, observation, instructions, seed, replan_index,
               active, episode_id, norm_stream=None, self_checks=None):
        torch = self.torch
        prefix = self.prefix(role,observation,instructions[goal])
        donor_role, donor_goal = donor_for(role,mode,goal) if active and mode != "self" else (None,None)
        donor_prefix = self.prefix(donor_role,observation,instructions[donor_goal]) if donor_role else None
        initial = self.noise(seed,replan_index)
        meta = {"episode_id":episode_id,"replan_index":replan_index,"start_step":replan_index*10,
                "noise_sha256":tensor_sha(initial),"noise_generator_seed":replan_seed(seed,replan_index),
                "patch_flow_count":0,"donor_model":donor_role,"donor_goal":donor_goal,
                "recipient_denoise_calls":0,"donor_denoise_calls":0,"actual_message_witness_max_abs":0.0,
                "message_delta_rms_min":None,"message_delta_rms_max":None}
        local_deltas, flow = [], [0]
        def velocity(x_t,timestep):
            f = flow[0]; flow[0] += 1
            self.counts["integrator_velocity_calls"] += 1
            if not active:
                out,_ = self.query(role,prefix,x_t,timestep)
                meta["recipient_denoise_calls"] += 1
                return out
            if mode == "self":
                native,cap = self.query(role,prefix,x_t,timestep,capture=True)
                out,applied = self.query(role,prefix,x_t,timestep,source=cap[0],capture=True)
                error = float((native-out).abs().max())
                self_checks.append({"flow_index":f,"velocity_max_abs":error,
                                    "witness_max_abs":applied[2]})
                if error > 1e-5:
                    raise AssertionError("Same-message self replacement changes denoising")
                meta["recipient_denoise_calls"] += 2
                self.counts["engineering_self_replacements"] += 1
                return out
            _, donor = self.query(donor_role,donor_prefix,x_t,timestep,capture=True,donor=True)
            meta["donor_denoise_calls"] += 1
            source = donor[0]
            if mode == "random_delta":
                _, native = self.query(role,prefix,x_t,timestep,capture=True)
                meta["recipient_denoise_calls"] += 1
                self.counts["extra_random_recipient_calibration_calls"] += 1
                random_seed = random_seed_for(f"{episode_id}|r{replan_index}|f{f}",role,goal,
                                               self.cfg["site"],self.cfg["random_seed_salt"])
                source, evidence = matched_random_message(native[0],donor[0],15,random_seed,native[1],torch)
                if norm_stream is not None:
                    dump_line(norm_stream,{"episode_id":episode_id,"replan_index":replan_index,"flow_index":f,
                                            "correct_delta_block_l2":norm_blocks(donor[0].float()-native[0].float(),15).tolist(),
                                            **evidence})
            out, applied = self.query(role,prefix,x_t,timestep,source=source,capture=True)
            meta["recipient_denoise_calls"] += 1
            meta["patch_flow_count"] += 1
            self.counts["patch_flows"] += 1
            native_message = applied[0]
            actual_source = source.to(device=native_message.device,dtype=native_message.dtype)
            local_deltas.append(rms(actual_source.float()-native_message.float()))
            meta["actual_message_witness_max_abs"] = max(meta["actual_message_witness_max_abs"],applied[2])
            return out
        actions = self.integrate(velocity,initial,10)[...,:7]
        if flow[0] != 10:
            raise AssertionError("Expected all ten actual Euler flow steps")
        if local_deltas:
            meta["message_delta_rms_min"],meta["message_delta_rms_max"] = min(local_deltas),max(local_deltas)
        return actions,meta

    def close(self):
        for hook in self.hooks.values():
            hook.close()


def validate_config(cfg):
    if (cfg["site"] != 9 or cfg["indices"] != [0,1] or cfg["seeds"] != [2026102801,2026102802]
            or cfg["horizon"] != 400 or cfg["intervention_steps"] != 120
            or cfg["n_action_steps"] != 10 or cfg["flow_steps"] != 10
            or cfg["noise_replan_stride"] != 1000003):
        raise ValueError("Fixed closed-loop protocol changed")
    if {k:tuple(v) for k,v in cfg["roles_and_modes"].items()} != MODES:
        raise ValueError("Closed-loop condition matrix changed")
    if cfg.get("save_videos",False):
        raise ValueError("This fixed worker records full events/actions but does not encode videos; save_videos must be false")


def engineering(args,cfg,report,rt,evaluation,env,physical_states,scene_xml,instructions,base_cfg,fresh,guard):
    obs,_ = evaluation.reset_common(env,physical_states["front"][0],base_cfg["seed"],
                                    base_cfg["evaluation"]["settle_steps"],scene_xml)
    obs=fresh(env)
    checks=[]
    for role in ("R0","FT4"):
        for goal in GOALS:
            policy=rt.models[role];policy.reset()
            noise=rt.noise(cfg["seeds"][0],0)
            official=policy.predict_action_chunk(rt.batch(role,obs,instructions[goal]),noise=noise).detach()
            rt.counts["official_engineering_flow_calls"] += 10
            rt.counts["recipient_denoise_calls"] += 10
            rt.counts["denoise_calls_by_model"][role] += 10
            native,_=rt.sample(role,"native",goal,obs,instructions,cfg["seeds"][0],0,False,"engineering")
            flows=[]
            self_chunk,_=rt.sample(role,"self",goal,obs,instructions,cfg["seeds"][0],0,True,"engineering",self_checks=flows)
            native_error=float((official-native).abs().max())
            self_error=float((official-self_chunk).abs().max())
            official_physical=rt.physical_chunk(role,official)
            native_physical=rt.physical_chunk(role,native)
            self_physical=rt.physical_chunk(role,self_chunk)
            post_error=float(rt.np.max(rt.np.abs(official_physical-self_physical)))
            post_native_error=float(rt.np.max(rt.np.abs(official_physical-native_physical)))
            clip_error=float(rt.np.max(rt.np.abs(rt.np.clip(official_physical,-1,1)-rt.np.clip(self_physical,-1,1))))
            # Explicitly match normal select_action postprocessing shape, not just chunk broadcast.
            per_action_error=max(float(rt.np.max(rt.np.abs(
                rt.post[role](official[:,i,:].clone()).float().cpu().numpy()[0]-official_physical[i])))
                                 for i in range(10))
            maximum=max(native_error,self_error,post_error,post_native_error,clip_error,per_action_error,
                        *(r["velocity_max_abs"] for r in flows),*(r["witness_max_abs"] for r in flows))
            if maximum > 1e-5:
                raise AssertionError(f"Official/self/native full-chunk engineering mismatch: {maximum}")
            checks.append({"model":role,"goal":goal,"native_chunk_max_abs":native_error,
                           "self_chunk_max_abs":self_error,"post_native_max_abs":post_native_error,
                           "post_self_max_abs":post_error,"clip_max_abs":clip_error,
                           "post_per_action_max_abs":per_action_error,"flows":flows,"max_abs":maximum})
            report["engineering_checks"]=checks
            guard()
    gate={"passed":True,"checks":checks,"tolerance":1e-5,"site":cfg["site"],
          "script_sha256":report["script_sha256"],"config_sha256":report["config_sha256"],
          "model_sha256":report["model_sha256"],"sources":report["sources"],
          "cache_isolation":{"deep_copy_per_query":True,"key_value_storage_unique":True},
          "no_physical_actions_beyond_reset_settle":True}
    write(args.output/"engineering.json",gate)


def evaluation_run(args,cfg,report,rt,evaluation,env,physical_states,scene_xml,instructions,base_cfg,fresh,guard):
    from stage1_protocol import summarize_events
    gate_path=args.output.parent/"engineering/engineering.json"
    gate=json.loads(gate_path.read_text())
    if not gate["passed"] or any(gate[k] != report[k] for k in
                                  ("script_sha256","config_sha256","model_sha256","sources")):
        raise ValueError("Engineering gate/provenance differs")
    write(args.output/"engineering.json",gate)
    report["engineering_source"]={"path":str(gate_path),"sha256":sha(gate_path)}
    rows=[]
    with gzip.open(args.output/"events.jsonl.gz","wt",encoding="utf-8") as event_stream, \
            gzip.open(args.output/"replans.jsonl.gz","wt",encoding="utf-8") as replan_stream, \
            gzip.open(args.output/"norms.jsonl.gz","wt",encoding="utf-8") as norm_stream:
        for role,mode,goal,index,seed in episode_jobs():
            episode_id=f"{role}|{mode}|{goal}|i{index:02d}|s{seed}"
            report["episode_in_progress"]=episode_id
            guard()
            rt.np.random.seed(base_cfg["seed"]+index)
            obs,_=evaluation.reset_common(env,physical_states["front"][index],base_cfg["seed"]+index,
                                           base_cfg["evaluation"]["settle_steps"],scene_xml)
            obs=fresh(env)
            flat=rt.np.asarray(env.get_sim_state(),dtype=rt.np.float64)
            state_hash=hashlib.sha256(flat.tobytes()).hexdigest()
            for policy in rt.models.values():policy.reset()
            flags=evaluation.events(env,report["predicates"])
            if any(flags.values()):raise AssertionError("Already-successful reset")
            trace=[flags]
            dump_line(event_stream,{"episode_id":episode_id,"step":0,"predicates":flags})
            episode_patches=0;clipped_steps=0
            for replan in range(40):
                start_step=replan*10
                active=mode != "native" and start_step < cfg["intervention_steps"]
                normalized,meta=rt.sample(role,mode,goal,obs,instructions,seed,replan,active,episode_id,norm_stream)
                controls=rt.physical_chunk(role,normalized)
                dump_line(replan_stream,meta)
                episode_patches += meta["patch_flow_count"]
                for action_index in range(10):
                    action=controls[action_index]
                    executed=rt.np.clip(action,-1,1)
                    clipped=action != executed
                    clipped_steps += int(clipped.any())
                    obs,_,_,_=env.step(executed)
                    step=start_step+action_index+1
                    # Match source diagnostics' fresh observations at every planning boundary.
                    if step % 10 == 0:obs=fresh(env)
                    flags=evaluation.events(env,report["predicates"])
                    trace.append(flags)
                    dump_line(event_stream,{"episode_id":episode_id,"step":step,"predicates":flags,
                                            "action_unclipped":action.tolist(),"action_executed":executed.tolist(),
                                            "clipped_dimensions":clipped.tolist(),
                                            "eef_position":rt.np.asarray(obs["robot0_eef_pos"]).tolist(),
                                            "object_positions":{k:rt.np.asarray(v).tolist() for k,v in obs.items()
                                                                if k.endswith("_pos") and not k.startswith("robot")}})
                event_stream.flush();replan_stream.flush();norm_stream.flush()
                report["steps_completed"] += 10
                report["replans_completed"] += 1
                guard()
            summary=summarize_events(goal,trace)
            first=summary["first_goal_times"][goal]
            boundary=cfg["intervention_steps"]
            row={"episode_id":episode_id,"model":role,"condition":mode,"init_index":index,"noise_seed":seed,
                 "seed":base_cfg["seed"]+index,"steps":400,"initial_state_sha256":state_hash,
                 "scene_xml_sha256":report["scene_xml_sha256"],"patch_flows":episode_patches,
                 "clipped_steps":clipped_steps,"active_goals_at_window_end":[g for g in ALL_GOALS if trace[boundary][g]],
                 "first_requested_before_or_at_window_end":first is not None and first <= boundary,
                 "requested_ever_after_window":any(e[goal] for e in trace[boundary+1:]),
                 "newly_reached_requested_after_window":first is not None and first > boundary,
                 **summary}
            rows.append(row)
            with (args.output/"episodes.jsonl").open("a",encoding="utf-8") as handle:dump_line(handle,row)
            report["episodes_completed"]=len(rows)
            print(json.dumps(row),flush=True)
            guard()
    report.pop("episode_in_progress",None)
    if (len(rows) != 68 or rt.counts["patch_flows"] != 5760
            or rt.counts["integrator_velocity_calls"] != 27200
            or rt.counts["extra_random_recipient_calibration_calls"] != 1920
            or rt.counts["recipient_denoise_calls"] != 29120
            or rt.counts["donor_denoise_calls"] != 5760):
        raise AssertionError("Incomplete registered closed-loop matrix")
    report["behavior_summary"]={f"{role}/{mode}/{goal}":{
        "episodes":len(selected),"correct_ever":sum(r["correct_ever"] for r in selected),
        "terminal_correct":sum(r["terminal_category"] == "correct" for r in selected),
        "terminal_right":sum(r["terminal_goal"] == "right" for r in selected),
        "unfinished":sum(r["terminal_category"] == "unfinished" for r in selected)}
        for role,mode,goal in sorted({(r["model"],r["condition"],r["requested_goal"]) for r in rows})
        for selected in [[r for r in rows if (r["model"],r["condition"],r["requested_goal"]) == (role,mode,goal)]]}


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cfg=json.loads(args.config.read_text(encoding="utf-8"));validate_config(cfg)
    started=time.monotonic()
    report={"status":"initializing","phase":args.phase,"started_at":datetime.now(timezone.utc).isoformat(),
            "scope":"development closed-loop message intervention, not training or autonomous method",
            "configuration":cfg,"script_sha256":sha(__file__),"config_sha256":sha(args.config),
            "episodes_completed":0,"steps_completed":0,"replans_completed":0,
            "no_early_stop_on_success":True,"horizon":400,"intervention_steps":120,"site":9}
    rt=None;env=None
    try:
        sys.path.insert(0,str(args.runtime_scripts.resolve()))
        import numpy as np
        import torch
        import lerobot.policies as policies
        from lerobot.utils import constants
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        from lerobot.policies.common.flow_matching import euler_integrate
        from robosuite.utils.transform_utils import quat2axisangle
        import stage1_r0_eval as evaluation
        from stage1_smoke import DEFAULT_SCENE
        from stage2k_entry_handoff import _fresh_observation
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError("Registered runtime requires CUDA")
        for path,expected in cfg["expected_model_sources"].items():
            if sha(path) != expected:raise ValueError(f"Actual model source changed: {path}")
        rt=ClosedLoopRuntime(cfg,report,torch,np,policies,evaluation,constants,quat2axisangle,
                             make_att_2d_masks,euler_integrate)
        report["sources"]={}
        for name,module in (("evaluation",evaluation),("model",sys.modules[type(rt.models["R0"].model).__module__]),
                            ("expert",sys.modules[type(rt.models["R0"].model.vlm_with_expert).__module__])):
            report["sources"][name]={"path":module.__file__,"sha256":sha(module.__file__)}
        root_cfg=json.loads(Path(cfg["base_config"]).read_text())
        if not set(cfg["indices"]) <= set(root_cfg["evaluation"]["development_indices"]):
            raise ValueError("Closed-loop development states not in registered development partition")
        tasks,paths,predicates,physical_states=evaluation.task_resources(root_cfg.get("scene",DEFAULT_SCENE))
        instructions={g:tasks[g].language for g in ALL_GOALS}
        report["instructions"],report["predicates"]=instructions,predicates
        env=evaluation.make_env(paths["front"])
        scene_xml=evaluation.canonical_scene(paths["front"],root_cfg["seed"])
        report["scene_xml_sha256"]=hashlib.sha256(scene_xml.encode()).hexdigest()
        def guard():
            if time.monotonic()-started > cfg.get("maximum_worker_minutes",60)*60:
                raise TimeoutError("Registered worker time budget exceeded")
            size=sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if size > cfg.get("maximum_output_bytes",2*1024**3) or shutil.disk_usage(args.output).free < cfg.get("minimum_disk_free_bytes",2*1024**3):
                raise RuntimeError("Registered output/disk budget exceeded; no cleanup")
            report["counts"]={**copy.deepcopy(rt.counts),"episodes":report["episodes_completed"],
                              "steps":report["steps_completed"],"replans":report["replans_completed"]}
            write(args.output/"progress.json",{**report,"elapsed_seconds":time.monotonic()-started})
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            handler=engineering if args.phase == "engineering" else evaluation_run
            handler(args,cfg,report,rt,evaluation,env,physical_states,scene_xml,instructions,root_cfg,_fresh_observation,guard)
        guard()
        report.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-started,no_followup_launched=True,
                      actual_message_witness_max_abs=rt.max_witness,cache_isolation_checks=rt.cache_checks,
                      counts={**copy.deepcopy(rt.counts),"episodes":report["episodes_completed"],
                              "steps":report["steps_completed"],"replans":report["replans_completed"]})
        report["files"]={p.relative_to(args.output).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)}
                         for p in args.output.rglob("*") if p.is_file() and p.name not in ("manifest.json","COMPLETE.json","progress.json")}
        write(args.output/"manifest.json",report);write(args.output/"COMPLETE.json",report)
        write(args.output/"progress.json",report)
    except Exception as error:
        report.update(status="failed",error=repr(error),elapsed_seconds=time.monotonic()-started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write(args.output/"FAILED.json",report);write(args.output/"progress.json",report)
        raise
    finally:
        if rt is not None:rt.close()
        if env is not None:env.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("engineering","evaluation"),required=True)
    parser.add_argument("--runtime-scripts",type=Path,required=True)
    main(parser.parse_args())
