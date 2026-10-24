"""Fixed-state, four-predicate diagnostics; independent qualification is explicit."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import time
from contextlib import ExitStack

from stage1_protocol import GOALS, qualification, summarize_events, remap_demo_xml
from stage1_smoke import DEFAULT_SCENE, ROOT, CAMERAS, dump, load_policy, sources, task_name


def task_resources(scene=DEFAULT_SCENE):
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    # STUDY_SCENE1's back variant lives in libero_10; every other book/caddy task is in libero_90.
    pool=[t for suite in ("libero_90","libero_10") for t in benchmark.get_benchmark_dict()[suite]().tasks]
    tasks={g:next(t for t in pool if t.name==task_name(g,scene)) for g in GOALS}
    paths={g:Path(get_libero_path("bddl_files"))/t.problem_folder/t.bddl_file for g,t in tasks.items()}
    predicates={g:robosuite_parse_problem(str(p))["goal_state"] for g,p in paths.items()}
    states={g:torch.load(Path(get_libero_path("init_states"))/t.problem_folder/t.init_states_file,weights_only=False) for g,t in tasks.items()}
    return tasks,paths,predicates,states


def make_env(path):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=str(path),camera_heights=128,camera_widths=128,ignore_done=True)


def events(env, predicates):
    return {g:bool(all(env.env._eval_predicate(p) for p in parts)) for g,parts in predicates.items()}


def canonical_scene(path, seed):
    env=make_env(path)
    try:
        env.seed(seed);env.reset()
        return env.sim.model.get_xml()
    finally:env.close()


def reset_common(env, state, seed, settle, scene_xml):
    import numpy as np
    env.seed(seed);env.reset();env.reset_from_xml_string(scene_xml);env.sim.reset()
    obs=env.set_init_state(state)
    for _ in range(settle):
        obs,_,_,_=env.step([0,0,0,0,0,0,-1])
    flat=np.asarray(env.get_sim_state(),dtype=np.float64)
    return obs,flat


def verify(args, config):
    import h5py
    import numpy as np
    import robosuite
    from libero.libero import get_libero_path
    scene=config.get("scene",DEFAULT_SCENE)
    tasks,paths,predicates,states=task_resources(scene)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report={"predicates":predicates,"replays":[],"reset_partitions":config["evaluation"]}
    # Judge independently using saved simulator states and replaying real controls.
    for goal,path in sources(scene=scene).items():
        env=make_env(paths[goal])
        try:
            with h5py.File(path) as h:
                successes=0
                for name in config['evaluation']['expert_replay_demos']:
                    demo=h['data'][name]
                    xml=remap_demo_xml(demo.attrs['model_file'],get_libero_path('assets'),Path(robosuite.__file__).parent)
                    env.seed(config['seed']);env.reset();env.reset_from_xml_string(xml);env.sim.reset()
                    env.set_init_state(demo["states"][-1])
                    terminal=events(env,predicates)
                    assert terminal[goal] and sum(terminal.values())==1,(goal,name,terminal)
                    env.reset_from_xml_string(xml);env.sim.reset();env.set_init_state(demo["states"][0])
                    trace=[events(env,predicates)]
                    for action in demo["actions"][:]:
                        env.step(action);trace.append(events(env,predicates))
                    summary=summarize_events(goal,trace)
                    successes+=int(summary['correct_ever'])
                    row={"goal":goal,"demo":name,"stored_terminal":terminal,"action_replay":summary}
                    report["replays"].append(row)
                    dump(args.output/"predicate_checks.json",report)
                    print(json.dumps(row),flush=True)
                assert successes>0,f'No replay reached {goal}; investigate control interface'
        finally: env.close()
    e=config["evaluation"]
    groups=[set(e[k]) for k in ("development_indices","qualification_indices","paired_indices","reserve_indices")]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    common=states[e["common_state_source_goal"]]
    scene_xml=canonical_scene(paths[e['common_state_source_goal']],config['seed'])
    assert set.union(*groups)==set(range(len(common)))
    reset_rows=[]
    for goal in config["old_goals"]:
        env=make_env(paths[goal])
        try:
            for idx in e["screen_indices"]:
                _,flat=reset_common(env,common[idx],config["seed"]+idx,e["settle_steps"],scene_xml)
                flags=events(env,predicates)
                assert not any(flags.values()),flags
                reset_rows.append({"goal":goal,"index":idx,"state":flat.tolist()})
        finally:env.close()
    for idx in e["screen_indices"]:
        vectors=[r["state"] for r in reset_rows if r["index"]==idx]
        assert all(np.allclose(vectors[0],v,atol=1e-10,rtol=0) for v in vectors[1:]),"Cross-instruction reset differs"
    if args.audit:
        audit=json.loads(args.audit.read_text())
        train_states=np.array([row['first_sim_state'][1:] for goal in config['old_goals']
                for row in audit['files'][goal]['trajectories'] if row['demo'] in audit['splits'][goal]['train']])
        distances={i:float(np.linalg.norm(train_states-np.asarray(common[i])[1:],axis=1).min())
                for i in e['qualification_indices']}
        assert min(distances.values())>1e-8,'Qualification reset duplicates a training start'
        report['qualification_min_train_state_distances']=distances
    report.update(status="pass",common_reset_checks=reset_rows,
                  init_states_sha256=hashlib.sha256(np.asarray(common).tobytes()).hexdigest())
    report['canonical_scene_xml_sha256']=hashlib.sha256(scene_xml.encode()).hexdigest()
    report['open_loop_replay_failures']=[{'goal':r['goal'],'demo':r['demo']} for r in report['replays'] if not r['action_replay']['correct_ever']]
    report['scope']='Predicate and interface checks pass; open-loop replay failures remain reported, not discarded.'
    dump(args.output/"predicate_checks.json",report)
    print("PREDICATE_AND_COMMON_RESET_CHECKS_PASS",flush=True)


def evaluate(args, config):
    import numpy as np
    import torch
    from lerobot.policies import make_pre_post_processors
    from robosuite.utils.transform_utils import quat2axisangle
    from PIL import Image, ImageDraw

    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    e=config["evaluation"]
    indices=e[args.partition+"_indices"]
    if args.limit:indices=indices[:args.limit]
    instruction_goals=args.instruction_goals or config["old_goals"]
    if len(set(instruction_goals))!=len(instruction_goals):raise ValueError('Repeated instruction goals')
    policy=load_policy(args.checkpoint)
    policy.config.n_action_steps=config["model"]["n_action_steps"]
    policy.config.num_steps=config["model"]["flow_steps"]
    policy.eval()
    if args.base_policy:
        from lerobot.datasets import LeRobotDatasetMetadata
        metadata=LeRobotDatasetMetadata('local/cl_binding_r0_train',root=ROOT/config['data']['train_output'])
        pre,post=make_pre_post_processors(policy.config,dataset_stats=metadata.stats)
    else:
        pre,post=make_pre_post_processors(policy.config,pretrained_path=args.checkpoint)
    tasks,paths,predicates,states=task_resources(config.get("scene",DEFAULT_SCENE))
    common=states[e["common_state_source_goal"]]
    scene_xml=canonical_scene(paths[e['common_state_source_goal']],config['seed'])
    rows=[]; start=time.monotonic()
    hashes={}
    for goal in instruction_goals:
        env=make_env(paths[goal])
        try:
            for idx in indices:
                seed=config["seed"]+idx
                torch.manual_seed(seed);np.random.seed(seed)
                obs,flat=reset_common(env,common[idx],seed,e["settle_steps"],scene_xml)
                if idx in hashes:assert np.allclose(hashes[idx],flat,atol=1e-10,rtol=0)
                hashes[idx]=flat
                policy.reset();trace=[events(env,predicates)];clipped_steps=0
                assert not any(trace[0].values()),"Already-successful reset"
                with ExitStack() as stack:
                    trace_file=None;video=None
                    if args.trace:
                        trace_file=stack.enter_context((args.output/f"{goal}_{idx:02d}_trace.jsonl").open("w"))
                        if idx==indices[0]:
                            import av
                            video=stack.enter_context(av.open(str(args.output/f"{goal}_{idx:02d}.mp4"),mode="w"))
                            stream=video.add_stream("libx264",rate=5)
                            stream.width=256;stream.height=152;stream.pix_fmt="yuv420p"
                            stream.options={"crf":"23","preset":"fast"}
                    for step in range(e["horizon"]):
                        inference_state=np.concatenate((obs["robot0_eef_pos"],quat2axisangle(obs["robot0_eef_quat"].copy()),obs["robot0_gripper_qpos"])).astype(np.float32)
                        batch={"observation.state":torch.from_numpy(inference_state),"task":tasks[goal].language}
                        for key,raw in zip(CAMERAS,("agentview_image","robot0_eye_in_hand_image"),strict=True):
                            image=np.ascontiguousarray(obs[raw][::-1,::-1])
                            batch[key]=torch.from_numpy(image).permute(2,0,1).float()/255
                        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                            action=post(policy.select_action(pre(batch))).float().cpu().numpy().reshape(-1)
                        assert action.shape==(7,) and np.isfinite(action).all()
                        clipped=np.clip(action,-1,1);clipped_steps+=int(np.any(clipped!=action))
                        obs,_,_,_=env.step(clipped)
                        flags=events(env,predicates);trace.append(flags)
                        if trace_file:
                            record={"step":step+1,"inference_state":inference_state.tolist(),
                                    "action_unclipped":action.tolist(),"action_executed":clipped.tolist(),
                                    "clipped_dimensions":(action!=clipped).tolist(),"predicates_after":flags,
                                    "eef_position_after":np.asarray(obs["robot0_eef_pos"]).tolist(),
                                    "gripper_position_after":np.asarray(obs["robot0_gripper_qpos"]).tolist(),
                                    "object_positions_after":{k:np.asarray(v).tolist() for k,v in obs.items()
                                                              if k.endswith("_pos") and not k.startswith("robot")}}
                            trace_file.write(json.dumps(record)+"\n")
                        if video and step%4==0:
                            canvas=Image.new("RGB",(256,152),"white")
                            ImageDraw.Draw(canvas).text((3,3),f"{goal} init={idx} step={step+1}",fill="black")
                            for column,raw in enumerate(("agentview_image","robot0_eye_in_hand_image")):
                                canvas.paste(Image.fromarray(np.ascontiguousarray(obs[raw][::-1,::-1])),(128*column,24))
                            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format="rgb24")):
                                video.mux(packet)
                    if video:
                        for packet in stream.encode():video.mux(packet)
                row={**summarize_events(goal,trace),"init_index":idx,"seed":seed,
                     "initial_state_sha256":hashlib.sha256(flat.tobytes()).hexdigest(),
                     "scene_xml_sha256":hashlib.sha256(scene_xml.encode()).hexdigest(),
                     "steps":e["horizon"],"clipped_steps":clipped_steps}
                rows.append(row)
                with (args.output/"episodes.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
                print(json.dumps(row),flush=True)
                if idx==indices[0]:
                    for raw in ("agentview_image","robot0_eye_in_hand_image"):
                        Image.fromarray(np.ascontiguousarray(obs[raw][::-1,::-1])).save(args.output/f"{goal}_{raw}_terminal.png")
        finally:env.close()
    minimum=e["qualification_required_correct"] if args.partition=="qualification" else (len(indices) if args.partition=="screen" else e["development_required_correct"])
    q=qualification(rows,config["old_goals"],len(indices),minimum)
    counts=qualification(rows,instruction_goals,len(indices),minimum)['counts']
    report={"status":"complete","partition":args.partition,"checkpoint":str(args.checkpoint) if args.checkpoint else None,
            "policy_source":"base_with_training_stats" if args.base_policy else "checkpoint",
            "instruction_goals":instruction_goals,"gate_goals":config['old_goals'],
            "normalization_source":config['data'].get('normalization_source',config['data']['train_output']),
            "git_commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            "counts":counts,"passed_gate":q["qualified"],"seconds":time.monotonic()-start,
            "horizon":e["horizon"],"n_action_steps":policy.config.n_action_steps,
            "step_traces":args.trace,"videos_first_init_per_goal":args.trace,
            "qualified_R0":bool(q["qualified"] and args.partition=="qualification" and not args.limit)}
    dump(args.output/"evaluation.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["verify","evaluate"])
    p.add_argument("--config",type=Path,default=ROOT/"configs/stage1_r0.json")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--checkpoint",type=Path)
    p.add_argument("--partition",choices=["screen","development","qualification","paired"],default="screen")
    p.add_argument("--limit",type=int)
    p.add_argument("--audit",type=Path)
    p.add_argument("--trace",action="store_true",help="Write per-step controls/predicates and a video for the first init per goal")
    p.add_argument("--instruction-goals",nargs="+",choices=GOALS,help="Instructions to test; capability gate still applies only to config old_goals")
    p.add_argument("--base-policy",action="store_true",help="Evaluate the base with this config's training-only normalization statistics")
    a=p.parse_args();c=json.loads(a.config.read_text())
    torch_threads=__import__('torch').set_num_threads(4)
    if a.mode=="verify":verify(a,c)
    else:
        if bool(a.checkpoint)==a.base_policy:p.error("evaluate requires exactly one of --checkpoint or --base-policy")
        evaluate(a,c)
