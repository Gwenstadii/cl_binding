"""Post-hoc object-transport endpoints, independent of contact or final placement."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

GOALS=("left","front","right","back")

def first_run(steps, minimum=5):
    count=0;previous=None
    for step in steps:
        count=count+1 if previous is not None and step==previous+1 else 1
        previous=step
        if count>=minimum:return step-minimum+1
    return None

def inside_xy(position, site):
    # All native boxes are axis-aligned after a signed axis permutation.
    rotation=site['rotation'];sizes=site['half_size'];centre=site['centre']
    for row in rotation:
        if sum(abs(x)>1e-8 for x in row)!=1:raise ValueError('Tilted site requires another projection rule')
    extents=[sum(abs(rotation[i][j])*sizes[j] for j in range(3)) for i in range(2)]
    return all(abs(position[i]-centre[i])<=extents[i] for i in range(2))

def analyze(root):
    directory=root/'movement';geometry=json.loads((directory/'geometry.json').read_text())
    manifest=json.loads((root/'evaluation/manifest.json').read_text())
    if geometry['canonical_scene_xml_sha256']!=manifest['scene_xml_sha256']:raise ValueError('Scene mismatch')
    episodes=[json.loads(s) for s in (root/'evaluation/episodes.jsonl').read_text().splitlines()]
    traces=defaultdict(list)
    with gzip.open(root/'evaluation/events.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['step']>0:traces[row['episode_id']].append(row)
    baselines=defaultdict(list);results=[]
    for ep in episodes:
        rows=traces[ep['episode_id']]
        if [r['step'] for r in rows]!=list(range(1,401)):raise ValueError('Missing physical steps')
        initial=rows[0]['object_positions']['black_book_1_pos'];baselines[ep['init_index']].append(initial)
        conditions={'overhead_box':None,'near_6cm':.06,'near_8cm':.08,'near_10cm':.10}
        for name,radius in conditions.items():
            eligible={g:[] for g in GOALS}
            for row in rows:
                position=row['object_positions']['black_book_1_pos']
                if position[2]-initial[2]<=.03:continue
                distances={g:math.dist(position[:2],geometry['sites'][g]['centre'][:2]) for g in GOALS}
                for goal in GOALS:
                    hit=inside_xy(position,geometry['sites'][goal]) if radius is None else (
                        distances[goal]<radius and all(distances[goal]+1e-9<d for g,d in distances.items() if g!=goal))
                    if hit:eligible[goal].append(row['step'])
            for duration in (1,5):
                times={g:first_run(values,duration) for g,values in eligible.items()}
                earliest=min((t for t in times.values() if t is not None),default=None)
                first=[g for g,t in times.items() if t is not None and t==earliest]
                results.append({k:ep[k] for k in ('episode_id','model','condition','requested_goal','init_index','noise_seed')}|
                               {'metric':name,'consecutive_steps':duration,'first_entry_times':times,
                                'requested_reached':times[ep['requested_goal']] is not None,
                                'first_target':first[0] if len(first)==1 else None,'first_targets':first,
                                'right_reached':times['right'] is not None})
    for values in baselines.values():
        if max(math.dist(values[0],v) for v in values)>1e-6:raise ValueError('Step1 initial book position differs')
    groups=defaultdict(list)
    for row in results:groups[row['model'],row['condition'],row['requested_goal'],row['metric'],row['consecutive_steps']].append(row)
    summary=[]
    for key,rows in sorted(groups.items()):
        role,mode,goal,metric,duration=key
        summary.append({'model':role,'condition':mode,'requested_goal':goal,'metric':metric,
                        'consecutive_steps':duration,'episodes':len(rows),
                        'requested_reached':sum(r['requested_reached'] for r in rows),
                        'requested_first':sum(r['first_target']==goal for r in rows),
                        'right_reached':sum(r['right_reached'] for r in rows),
                        'right_first':sum(r['first_target']=='right' for r in rows),
                        'no_entry':sum(not r['first_targets'] for r in rows)})
    def info(path):return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return {'status':'posthoc_movement_analysis','source_files':{str(p.relative_to(root)):info(p) for p in
             (root/'evaluation/episodes.jsonl',root/'evaluation/events.jsonl.gz',directory/'geometry.json')},
            'script_sha256':info(Path(__file__))['sha256'],
            'protocol':{'primary':'native region XY projection, lift >3cm, 5 consecutive executed steps',
                        'secondary':'8cm XY radius and uniquely nearest, same lift and duration',
                        'sensitivity':'one-step entries and radii 6/10cm; no score-based threshold tuning',
                        'reference_height':'first recorded physical step; book position agrees across paired branches',
                        'no_contact_release_terminal_requirement':True,'no_new_model_rollouts':True,
                        'posthoc':True},'per_condition':summary,'episodes':results}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();target=args.root/'movement/results.json'
    if target.exists():raise FileExistsError(target)
    result=analyze(args.root);target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for row in result['per_condition']:
        if row['consecutive_steps']==5 and row['metric'] in ('overhead_box','near_8cm'):print(json.dumps(row))
