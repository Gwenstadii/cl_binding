"""Joint Same Instruction + Different State analysis of sealed N=40/60/80 runs.

No new model calls. The common FT4 baseline must match across all three batches.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOTS={40:'stage2t_same_instruction40',60:'stage2u_same_instruction60',80:'stage2s_same_instruction80'}


def info(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def analyze(base,out):
    datasets={};baseline=None;models=None;scene=None;sources={};rows=[];tables=[]
    for n,folder in ROOTS.items():
        root=base/folder;data=root/'evaluation'
        manifest=json.loads((data/'manifest.json').read_text());audit=json.loads((root/'audit.json').read_text())
        assert manifest['status']=='complete' and audit['status']=='passed'
        assert json.loads((root/'post_run_check.json').read_text())['passed']
        for name,expected in manifest['files'].items():
            if not name.startswith('videos/'):
                assert info(data/name)==expected,name
        if models is None:models=manifest['model_sha256'];scene=manifest['scene_xml_sha256']
        assert models==manifest['model_sha256'] and scene==manifest['scene_xml_sha256']
        eps={e['episode_id']:e for e in map(json.loads,(data/'episodes.jsonl').read_text().splitlines())}
        traces={eid:[] for eid in eps}
        with gzip.open(data/'events.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line)
                if r['step']:traces[r['episode_id']].append(r)
        indexed={(e['mode'],e['requested_goal'],e['init_index'],e['noise_seed']):(e,traces[eid]) for eid,e in eps.items()}
        native={(g,i,s):[{k:r[k] for k in ('step','predicates','action_executed','eef_position','object_positions')} for r in t]
                for (m,g,i,s),(e,t) in indexed.items() if m=='native'}
        if baseline is None:baseline=native
        else:assert native==baseline,'Common baseline was altered'
        assert len(native)==12
        for g in ('left','front','right'):
            for i in (0,1):
                for s in (2026102801,2026102802):
                    en,tn=indexed['native',g,i,s];eh,th=indexed[f'handoff{n}',g,i,s]
                    assert en['initial_state_sha256']==eh['initial_state_sha256']
                    p=tn[n-1]['object_positions']['black_book_1_pos'];q=th[n-1]['object_positions']['black_book_1_pos']
                    endn=tn[n+9]['object_positions']['black_book_1_pos'];endh=th[n+9]['object_positions']['black_book_1_pos']
                    row={'boundary':n,'instruction':g,'init_index':i,'noise_seed':s,
                         'book_state_distance_cm':100*math.dist(p,q),
                         'eef_state_distance_cm':100*math.dist(tn[n-1]['eef_position'],th[n-1]['eef_position']),
                         'FT4_prefix_book_y':p[1],'R0_prefix_book_y':q[1],
                         'FT4_prefix_grasp':tn[n-1]['native_grasp'],'R0_prefix_grasp':th[n-1]['native_grasp'],
                         'FT4_prefix_post10_dy_cm':100*(endn[1]-p[1]),'R0_prefix_post10_dy_cm':100*(endh[1]-q[1]),
                         'FT4_prefix_first':','.join(en['full_movement']['first_targets']),
                         'R0_prefix_first':','.join(eh['full_movement']['first_targets']),
                         'FT4_prefix_correct':en['full_movement']['first_targets']==[g],
                         'R0_prefix_correct':eh['full_movement']['first_targets']==[g]}
                    rows.append(row)
        for row in audit['per_condition']:
            clean={k:v for k,v in row.items() if not k.startswith('pretarget_at')}
            tables.append({'boundary':n,**clean,'pretarget_at_boundary':next(v for k,v in row.items() if k.startswith('pretarget_at'))})
        datasets[n]=(manifest,indexed)
        sources[str(n)]={name:info(root/name) for name in ('audit.json','evaluation/manifest.json','evaluation/events.jsonl.gz','evaluation/episodes.jsonl')}
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('paired_states.csv',rows),('results_by_state_source.csv',tables)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    note={'status':'passed','new_policy_rollouts':0,'new_training_updates':0,'paired_comparisons':36,
          'common_native_trajectories':12,'R0_prefix_trajectories':36,'baseline_reused_at_three_boundaries':True,
          'sources':sources,'limits':['This fixes instruction throughout; it does not test language sensitivity at fixed state.',
            'Different prefix source changes position, velocity, pose, contacts and phase together.',
            'Only two original development starts and two noise schedules; repeated native data are not new samples.',
            'R0 was not trained on Right.']}
    (out/'analysis_audit.json').write_text(json.dumps(note,indent=2)+'\n',encoding='utf-8')
    sites=datasets[40][0]['configuration']['movement_geometry']['sites']
    colors={'native':'#666666','handoff':'#187d9d'}
    for entity in ('book','eef'):
        fig,axes=plt.subplots(3,3,figsize=(13,13))
        all_points=[]
        for y,n in enumerate((40,60,80)):
            indexed=datasets[n][1]
            for x,g in enumerate(('left','front','right')):
                ax=axes[y,x]
                for goal,site in sites.items():
                    c=site['centre'];ext=[sum(abs(site['rotation'][d][j])*site['half_size'][j] for j in range(3)) for d in (0,1)]
                    ax.add_patch(Rectangle((c[0]-ext[0],c[1]-ext[1]),2*ext[0],2*ext[1],fill=False,color='.8'))
                    ax.text(c[0],c[1],goal,color='.65',ha='center',fontsize=8)
                for m in ('native',f'handoff{n}'):
                    color=colors['native' if m=='native' else 'handoff']
                    for i in (0,1):
                        for s in (2026102801,2026102802):
                            e,t=indexed[m,g,i,s]
                            points=[r['object_positions']['black_book_1_pos'] if entity=='book' else r['eef_position'] for r in t]
                            all_points.extend(points[n-1:])
                            bold=i==0 and s==2026102801
                            ax.plot([p[0] for p in points[n-1:]],[p[1] for p in points[n-1:]],color=color,
                                    alpha=1 if bold else .22,lw=1.6 if bold else .7,
                                    label=('FT4 prefix' if m=='native' else 'R0 prefix') if bold else None)
                            ax.scatter(*points[n-1][:2],color=color,marker='*',s=70 if bold else 15,alpha=1 if bold else .3)
                ax.set(title=f'{g.upper()}, N={n}',xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
        # A shared world-coordinate scale makes state-source and N comparisons readable.
        for ax in axes.flat:
            ax.set_xlim(min(-.49,min(p[0] for p in all_points)-.02),max(-.17,max(p[0] for p in all_points)+.02))
            ax.set_ylim(min(-.37,min(p[1] for p in all_points)-.02),max(.09,max(p[1] for p in all_points)+.02))
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle(f'SAME instruction, DIFFERENT prefix state: FT4 continuation {entity} paths\nAll four paired cases; bold = init0/noise2026102801; native baseline is shared across N.',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/f'state_source_{entity}_grid.png',dpi=155,bbox_inches='tight');plt.close(fig)
    return note


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(analyze(a.base,a.output)))
