"""Independent raw-trajectory verification using a direct5-step region scanner.

Does not call the rollout's movement helper. Does not rerun model evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

GOALS=('left','front','right','back')
GROUPS=('N','Z','R','ZR','G','ZG','RG','ZRG','H','HZ','HR','HZR','HG','HZG','HRG','HZRG')

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main(archive):
    official=read(archive/'summary.json')
    assert official['audits']['unique_regime_cases']==640
    records={};traces={};paths=[]
    for filename in ('reference_episodes.jsonl',*[f'i{i}.jsonl' for i in range(20,40)]):
        path=archive/filename;paths.append(path)
        for row in map(json.loads,path.read_text().splitlines()):
            key=row['init'],row['condition'],row['goal'];assert key not in records;records[key]=row
    for filename in ('reference_traces.jsonl',*[f'i{i}_traces.jsonl' for i in range(20,40)]):
        path=archive/filename;paths.append(path)
        for row in map(json.loads,path.read_text().splitlines()):
            key=row['init'],row['condition'],row['goal'];assert key not in traces;traces[key]=row['rows']
    assert len(records)==len(traces)==640
    geometry=read(archive/'geometry.json')['sites'];bounds={}
    for g,site in geometry.items():
        assert all(sum(abs(x)>1e-8 for x in row)==1 for row in site['rotation'])
        ext=[sum(abs(site['rotation'][i][j])*site['half_size'][j] for j in range(3)) for i in range(3)]
        bounds[g]=[(site['centre'][i]-ext[i],site['centre'][i]+ext[i]) for i in range(3)]
    recomputed={};right_failures={c:[] for c in GROUPS}
    for key,r in records.items():
        trace=traces[key];runs=dict.fromkeys(GOALS,0);arrivals=dict.fromkeys(GOALS,None)
        for row in trace:
            p=row['obj'];lift=p[2]>r['z0']+.03
            for goal in GOALS:
                inside=lift and all(bounds[goal][i][0]<=p[i]<=bounds[goal][i][1] for i in (0,1))
                runs[goal]=runs[goal]+1 if inside else 0
                if runs[goal]==5 and arrivals[goal] is None:arrivals[goal]=row['step']-4
        earliest=min((t for t in arrivals.values() if t is not None),default=None)
        first=[g for g in GOALS if arrivals[g]==earliest] if earliest is not None else []
        assert first==r['first'] and earliest==r['first_step'],(key,first,r['first'])
        ever={g:sum(bool(row['pred'][g]) for row in trace)>0 for g in GOALS}
        terminal={g:bool(trace[-1]['pred'][g]) for g in GOALS}
        assert ever==r['ever'] and terminal==r['terminal'],key
        recomputed[key]={'first':first,'ever':ever,'terminal':terminal}
        if key[2]=='B' and first==['right'] and not ever['right']:
            xy=[v for v in trace[r['boundary']:] if all(bounds['right'][i][0]<=v['obj'][i]<=bounds['right'][i][1] for i in (0,1))]
            right_failures[key[1]].append({'init':key[0],'first_step':earliest,'minimum_center_above_right_region_top_m':min(v['obj'][2]-bounds['right'][2][1] for v in xy),'first_both_contacts':bool(trace[earliest-1]['c1'] and trace[earliest-1]['c2']),'window_end_both_contacts':bool(trace[r['boundary']+59]['c1'] and trace[r['boundary']+59]['c2'])})
    counts={}
    for c in GROUPS:
        counts[c]={}
        for goal,target in (('A','left'),('B','right')):
            rows=[recomputed[i,c,goal] for i in range(20,40)]
            counts[c][goal]={'first':sum(v['first']==[target] for v in rows),'ever':sum(v['ever'][target] for v in rows),'terminal':sum(v['terminal'][target] for v in rows),'first_distribution':dict(Counter('/'.join(v['first']) or 'none' for v in rows))}
            expected=official['summary'][c]['per_goal'][goal]
            assert counts[c][goal]['first']==expected['first_correct'] and counts[c][goal]['ever']==expected['ever_correct'] and counts[c][goal]['terminal']==expected['terminal_correct']
        for i in range(20,40):
            assert official['per_snapshot'][c]['L_first'][i-20]==int(recomputed[i,c,'A']['first']==['left'])
            assert official['per_snapshot'][c]['R_first'][i-20]==int(recomputed[i,c,'B']['first']==['right'])
    report={'pass':True,'independent_geometric_first_scanner_cases':640,'independent_raw_trace_steps':256000,'counts_match_official':True,'counts':counts,'right_first_but_never_success':right_failures,'interpretation':'descriptive height/contact only; no causal mediation inference','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,archive/'geometry.json',Path(__file__)]}}
    (archive/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS: independent direct-region scanner640 cases /256000 steps and all endpoint counts')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);main(p.parse_args().archive.resolve())
