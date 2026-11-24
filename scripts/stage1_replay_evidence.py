"""Render stored evaluation controls and verify them against the original trace.

This performs no policy inference, optimization, or new score selection. A replay
is accepted only if the initial hashes, every predicate, and recorded positions
agree with the original rollout. Original evaluation files are never modified.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--goal', choices=('front', 'left', 'right', 'back'), required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=1e-7)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root/'scripts'))
    from stage1_r0_eval import canonical_scene, events, make_env, reset_common, task_resources
    from stage1_protocol import summarize_events
    from stage1_smoke import dump
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    entries = [json.loads(line) for line in args.episodes.read_text().splitlines() if line.strip()]
    expected = [r for r in entries if r['requested_goal']==args.goal and r['init_index']==args.index]
    if len(expected)!=1:
        raise ValueError('Expected one original episode for the selected condition')
    expected = expected[0]
    controls = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    if [r['step'] for r in controls]!=list(range(1,config['evaluation']['horizon']+1)):
        raise ValueError('Trace must contain the full original rollout')
    args.output.mkdir(parents=True)
    tasks, paths, predicates, states = task_resources()
    evaluation = config['evaluation']
    scene = canonical_scene(paths[evaluation['common_state_source_goal']],config['seed'])
    if hashlib.sha256(scene.encode()).hexdigest()!=expected['scene_xml_sha256']:
        raise ValueError('Scene XML differs from the original evaluation')
    env = make_env(paths[args.goal])
    maximum_errors = {'eef':0.0,'gripper':0.0,'objects':0.0}
    replay_events = []
    book_trace = []
    try:
        obs, state = reset_common(env,states[evaluation['common_state_source_goal']][args.index],
                                 expected['seed'],evaluation['settle_steps'],scene)
        if hashlib.sha256(state.tobytes()).hexdigest()!=expected['initial_state_sha256']:
            raise ValueError('Initial physical state differs from the original evaluation')
        replay_events.append(events(env,predicates))
        with av.open(str(args.output/'replay.mp4'),mode='w') as video:
            stream = video.add_stream('libx264',rate=config['data']['fps'])
            stream.width=256; stream.height=168; stream.pix_fmt='yuv420p'
            stream.options={'crf':'20','preset':'fast'}
            for row in controls:
                obs,_,_,_=env.step(np.asarray(row['action_executed']))
                flags=events(env,predicates)
                if flags!=row['predicates_after']:
                    raise ValueError(f"Predicate mismatch at step {row['step']}")
                replay_events.append(flags)
                for label,actual,recorded in (
                    ('eef',obs['robot0_eef_pos'],row['eef_position_after']),
                    ('gripper',obs['robot0_gripper_qpos'],row['gripper_position_after'])):
                    error=float(np.max(np.abs(np.asarray(actual)-np.asarray(recorded))))
                    maximum_errors[label]=max(maximum_errors[label],error)
                for key,recorded in row['object_positions_after'].items():
                    error=float(np.max(np.abs(np.asarray(obs[key])-np.asarray(recorded))))
                    maximum_errors['objects']=max(maximum_errors['objects'],error)
                if max(maximum_errors.values())>args.atol:
                    raise ValueError(f"Position mismatch at step {row['step']}: {maximum_errors}")
                step=row['step']; active=','.join(g for g,v in flags.items() if v) or 'none'
                canvas=Image.new('RGB',(256,168),'white'); draw=ImageDraw.Draw(canvas)
                draw.text((3,3),f"request={args.goal} init={args.index} step={step}",fill='black')
                draw.text((3,18),f"goal predicate: {active}",fill='black')
                for column,key in enumerate(('agentview_image','robot0_eye_in_hand_image')):
                    canvas.paste(Image.fromarray(np.ascontiguousarray(obs[key][::-1,::-1])),(128*column,40))
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):
                    video.mux(packet)
                book_trace.append({'step':step,'predicates':flags,
                    'book_position':row['object_positions_after'].get('black_book_1_pos'),
                    'eef_position':row['eef_position_after'],'gripper_qpos':row['gripper_position_after'],
                    'gripper_action':row['action_executed'][-1]})
            for packet in stream.encode():video.mux(packet)
        summary=summarize_events(args.goal,replay_events)
        for key,value in summary.items():
            if expected[key]!=value:
                raise ValueError(f'Summary mismatch: {key}')
        result={'status':'pass','purpose':'render_original_actions_without_policy_or_training',
            'requested_goal':args.goal,'init_index':args.index,'seed':expected['seed'],
            'config_path':str(args.config),'episodes_path':str(args.episodes),'trace_path':str(args.trace),
            'trace_sha256':hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            'initial_state_sha256':expected['initial_state_sha256'],'scene_xml_sha256':expected['scene_xml_sha256'],
            'steps':len(controls),'fps':config['data']['fps'],'predicate_mismatches':0,
            'maximum_position_errors':maximum_errors,'position_atol':args.atol,'summary':summary,
            'original_episode':expected,'new_policy_rollouts':0,'training_updates':0,
            'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'video_sha256':hashlib.sha256((args.output/'replay.mp4').read_bytes()).hexdigest()}
        dump(args.output/'replay_verification.json',result)
        dump(args.output/'timeline.json',book_trace)
        print(json.dumps(result),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
