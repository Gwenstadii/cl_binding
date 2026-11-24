"""Post-hoc axis decomposition; distinguish inherited lateral alignment from approach."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(root):
    eps=[json.loads(x) for x in (root/'evaluation/episodes.jsonl').read_text().splitlines()]
    ids={e['episode_id']:e for e in eps};traces={eid:{} for eid in ids}
    with gzip.open(root/'evaluation/events.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['step']:traces[r['episode_id']][r['step']]=r
    a=json.loads((root/'audit.json').read_text());out=[]
    for e in eps:
        t=traces[e['episode_id']];p=t[80]['object_positions']['black_book_1_pos']
        for window in (10,20,40):
            q=t[80+window]['object_positions']['black_book_1_pos']
            d=next(d for d in a['direction_rows'] if d['episode_id']==e['episode_id'] and d['window']==window)
            out.append({k:e[k] for k in ('mode','requested_goal','init_index','noise_seed')}|{'window':window,
                'book_x80':p[0],'book_y80':p[1],'book_x_end':q[0],'book_y_end':q[1],
                'book_dx_cm':100*(q[0]-p[0]),'book_dy_cm':100*(q[1]-p[1]),
                'command_dx_sum_cm':100*d['command_sum_xy_m'][0],'command_dy_sum_cm':100*d['command_sum_xy_m'][1]})
    pairs=[]
    for g in ('left','front','right'):
        for i in (0,1):
            for seed in (2026102801,2026102802):
                ds=[next(d for d in a['direction_rows'] if d['mode']==m and d['requested_goal']==g and d['init_index']==i and d['noise_seed']==seed and d['window']==10) for m in ('native','handoff80')]
                u,v=[d['command_sum_xy_m'] for d in ds];den=math.hypot(*u)*math.hypot(*v)
                cosine=sum(x*y for x,y in zip(u,v))/den if den>1e-12 else None
                pairs.append({'requested_goal':g,'init_index':i,'noise_seed':seed,'world_command_cosine':cosine,
                    'world_command_angle_degrees':math.degrees(math.acos(max(-1,min(1,cosine)))) if cosine is not None else None})
    for name,rs in [('axis_motion.csv',out),('paired_command_vectors.csv',pairs)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    cfg=json.loads((root/'registered_config.json').read_text());sites=cfg['movement_geometry']['sites']
    fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True)
    for row,(i,seed) in enumerate([(i,s) for i in (0,1) for s in (2026102801,2026102802)]):
        for col in (0,1):
            ax=axes[row,col]
            for g,color in [('left','#6e6ab3'),('right','#c59457')]:
                s=sites[g];ext=sum(abs(s['rotation'][col][j])*s['half_size'][j] for j in range(3));c=s['centre'][col]
                ax.axhspan(c-ext,c+ext,color=color,alpha=.12,label=g+' region' if row==0 else None)
            for mode,color in [('native','#666666'),('handoff80','#177f9c')]:
                e=next(e for e in eps if e['mode']==mode and e['requested_goal']=='left' and e['init_index']==i and e['noise_seed']==seed)
                t=traces[e['episode_id']];ax.plot(range(60,141),[t[s]['object_positions']['black_book_1_pos'][col] for s in range(60,141)],color=color,label=mode)
            ax.axvline(80,color='black',linestyle=':');ax.set(title=f'Left request | init {i}, noise {seed} | {"X" if col==0 else "Y"}',ylabel='World position (m)',xlabel='Step')
    axes[0,1].legend(fontsize=8,frameon=False)
    fig.suptitle('POST-HOC axis decomposition: R0 supplies lateral Y alignment before handoff\nAfter handoff the book approaches in -X while drifting +Y; scalar target-distance reduction can hide this.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/'left_axis_decomposition.png',dpi=155,bbox_inches='tight');plt.close(fig)
    (root/'axis_analysis_note.json').write_text(json.dumps({'posthoc':True,'new_model_calls':0,
        'purpose':'Explain scalar-distance ambiguity using the same recorded command vectors and physical paths.'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
