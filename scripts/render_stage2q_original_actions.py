"""Render saved controls, verify every original position/predicate, load no policy."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    sys.path.insert(0, str(args.runtime_root / 'scripts'))
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    import stage1_r0_eval as evaluation
    from stage1_smoke import DEFAULT_SCENE

    def fresh(env):
        env.sim.forward()
        env.check_success()
        env._post_process()
        env._update_observables(force=True)
        return env.env._get_observations()

    assert not args.output.exists(), 'Preserve previous render attempts'
    args.output.mkdir(parents=True)
    manifest = json.loads((args.source / 'evaluation/manifest.json').read_text())
    cfg = manifest['configuration']
    base = json.loads(Path(cfg['base_config']).read_text())
    audit = json.loads((args.source / 'audit.json').read_text())
    byid = {e['episode_id']: e for e in audit['episode_results']}
    eps = [json.loads(line) for line in (args.source / 'evaluation/episodes.jsonl').read_text().splitlines()]
    chosen = [e for e in eps if e['condition'] == 'full_old_response' and e['window'] == 120
              or args.include_right_native and e['model'] == 'FT4' and e['condition'] == 'native' and e['requested_goal'] == 'right']
    assert len(chosen) == (12 if args.include_right_native else 8)
    selected = {e['episode_id']: [] for e in chosen}
    event_path = args.source / 'evaluation/events.jsonl.gz'
    assert sha(event_path) == manifest['files']['events.jsonl.gz']['sha256']
    with gzip.open(event_path, 'rt') as f:
        for line in f:
            row = json.loads(line)
            if row['episode_id'] in selected:
                selected[row['episode_id']].append(row)
    tasks, paths, predicates, states = evaluation.task_resources(base.get('scene', DEFAULT_SCENE))
    xml = evaluation.canonical_scene(paths['front'], base['seed'])
    assert hashlib.sha256(xml.encode()).hexdigest() == manifest['scene_xml_sha256']
    env = evaluation.make_env(paths['front'])
    result = {'status': 'running', 'purpose': 'render_original_executed_actions_no_policy_inference',
              'source_root': str(args.source), 'renderer_sha256': sha(Path(__file__)),
              'source_events_sha256': sha(event_path), 'position_tolerance': 1e-7,
              'policy_inference_calls': 0, 'training_updates': 0,
              'right_is_FT4_native_not_R0_handoff': True, 'episodes': []}
    try:
        for ep in chosen:
            original = byid[ep['episode_id']]
            goal = ep['requested_goal']
            hybrid = ep['condition'] == 'full_old_response'
            route = 'R0_steps001-120_then_FT4_steps121-400' if hybrid else 'FT4_NATIVE_steps001-400'
            outcome = 'SUCCESS' if original['terminal_correct'] else 'FAIL'
            name = f"{goal.upper()}_{route}_init{ep['init_index']:02d}_seed{ep['noise_seed']}_final{outcome}.mp4"
            rows = selected[ep['episode_id']]
            assert [r['step'] for r in rows] == list(range(401))
            np.random.seed(base['seed'] + ep['init_index'])
            evaluation.reset_common(env, states['front'][ep['init_index']], base['seed'] + ep['init_index'],
                                    base['evaluation']['settle_steps'], xml)
            obs = fresh(env)
            flat = np.asarray(env.get_sim_state(), dtype=np.float64)
            assert hashlib.sha256(flat.tobytes()).hexdigest() == ep['initial_state_sha256']
            assert evaluation.events(env, predicates) == rows[0]['predicates']
            maximum = 0.
            fps = base['data']['fps']
            with av.open(str(args.output / name), 'w') as video:
                stream = video.add_stream('libx264', rate=fps)
                stream.width, stream.height, stream.pix_fmt = 768, 492, 'yuv420p'
                stream.options = {'crf': '20', 'preset': 'fast'}
                for row in rows[1:]:
                    step = row['step']
                    obs, _, _, _ = env.step(row['action_executed'])
                    if step % 10 == 0:
                        obs = fresh(env)
                    flags = evaluation.events(env, predicates)
                    assert flags == row['predicates'], (ep['episode_id'], step, flags)
                    errors = [np.max(np.abs(np.asarray(obs[k]) - np.asarray(v))) for k, v in row['object_positions'].items()]
                    errors.append(np.max(np.abs(np.asarray(obs['robot0_eef_pos']) - np.asarray(row['eef_position']))))
                    maximum = max(maximum, *map(float, errors))
                    assert maximum <= 1e-7, (ep['episode_id'], step, maximum)
                    canvas = Image.new('RGB', (768, 492), '#fafafa')
                    draw = ImageDraw.Draw(canvas)
                    controller = 'R0' if hybrid and step <= 120 else 'FT4'
                    draw.text((10, 8), f'Request: {goal.upper()} | step {step:03d}/400 | controller: {controller}', fill='black')
                    draw.text((10, 26), route, fill='black')
                    draw.text((10, 44), f'init {ep["init_index"]} | noise {ep["noise_seed"]} | ORIGINAL ACTION REPLAY (no policy inference)', fill='black')
                    draw.text((10, 62), f'Native In now: {[g for g,v in flags.items() if v]} | original final: {outcome}', fill='black')
                    draw.text((10, 80), 'Agent view                                             Wrist view', fill='black')
                    for i, key in enumerate(('agentview_image', 'robot0_eye_in_hand_image')):
                        image = Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1])).resize((384, 384))
                        canvas.paste(image, (i * 384, 108))
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
                        video.mux(packet)
                for packet in stream.encode():
                    video.mux(packet)
            result['episodes'].append({'file': name, 'episode_id': ep['episode_id'], 'requested_goal': goal,
                'route': route, 'init_index': ep['init_index'], 'noise_seed': ep['noise_seed'],
                'original_terminal_correct': original['terminal_correct'], 'original_box_correct': original['box_correct'],
                'original_box_first_targets': original['box_first_targets'], 'original_box_times': original['box_times'],
                'frames': 400, 'fps': fps, 'maximum_position_error': maximum, 'predicate_mismatches': 0,
                'sha256': sha(args.output / name), 'bytes': (args.output / name).stat().st_size})
            (args.output / 'replay_verification.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result['episodes'][-1]), flush=True)
        result['status'] = 'passed'
        (args.output / 'replay_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-right-native', action='store_true')
    main(parser.parse_args())
