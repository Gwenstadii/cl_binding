"""Describe sealed native prefixes; no new model calls or causal layer claims."""
import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def analyze(reference,output):
    data=reference/'evaluation'
    episodes=[json.loads(x) for x in (data/'episodes.jsonl').read_text().splitlines()]
    selected={e['episode_id']:e for e in episodes if e['condition']=='native' and e['requested_goal'] in ('left','front')}
    traces={}
    with gzip.open(data/'events.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['episode_id'] in selected and 1<=r['step']<=120:
                e=selected[r['episode_id']]
                traces.setdefault((e['model'],e['requested_goal'],e['init_index'],e['noise_seed']),[]).append(r)
    assert len(traces)==16 and all(len(t)==120 for t in traces.values())
    profiles=[]
    for h in (40,80,120):
        for role in ('R0','FT4'):
            differences=[math.dist(traces[role,'left',i,s][h-1]['object_positions']['black_book_1_pos'],
                                   traces[role,'front',i,s][h-1]['object_positions']['black_book_1_pos'])*100
                         for i in (0,1) for s in (2026102801,2026102802)]
            lifts=[(t[h-1]['object_positions']['black_book_1_pos'][2]-t[0]['object_positions']['black_book_1_pos'][2])*100
                   for (r,g,i,s),t in traces.items() if r==role]
            profiles.append({'boundary':h,'source_model':role,'LF_book_xyz_distance_cm_median':statistics.median(differences),
                'LF_book_xyz_distance_cm_min':min(differences),'LF_book_xyz_distance_cm_max':max(differences),
                'lifted_above_3cm':sum(v>3 for v in lifts),'episodes':8})
    result={'purpose':'descriptive source-trajectory context, not new intervention results',
            'source_events_sha256':hashlib.sha256((data/'events.jsonl.gz').read_bytes()).hexdigest(),
            'profiles':profiles,'limitations':['Book displacement is a physical trajectory difference, not a direct language-readout measure.',
                'Fixed time does not align grasp/movement phase across prefix goals.',
                'These are the same two development starts and two inference seeds.']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'source_progress.json').write_text(json.dumps(result,indent=2)+'\n')
    geometry=json.loads((reference/'registered_config.json').read_text())['movement_geometry']['sites']
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    colors={'left':'#287c8e','front':'#bf773c'}
    for ax,role in zip(axes[:2],('R0','FT4')):
        for goal,site in geometry.items():
            centre=site['centre'];ext=[sum(abs(site['rotation'][d][j])*site['half_size'][j] for j in range(3)) for d in (0,1)]
            ax.add_patch(Rectangle((centre[0]-ext[0],centre[1]-ext[1]),2*ext[0],2*ext[1],fill=False,edgecolor='.7'))
            ax.text(centre[0],centre[1],goal,ha='center',color='.5',fontsize=9)
        for goal in ('left','front'):
            t=traces[role,goal,0,2026102801]
            p=np.array([r['object_positions']['black_book_1_pos'] for r in t])
            ax.plot(p[:,0],p[:,1],color=colors[goal],label=goal.title())
            for h in (40,80,120):
                ax.scatter(p[h-1,0],p[h-1,1],color=colors[goal],s=25)
                ax.annotate(str(h),p[h-1,:2],xytext=(5,5 if goal=='left' else -12),textcoords='offset points',fontsize=9)
        ax.set(title=f'{role}: same init 0 / noise 2026102801',xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
        ax.legend(frameon=False,fontsize=9,loc='lower right')
    ax=axes[2]
    for role,color in [('R0','#287c8e'),('FT4','#bf773c')]:
        ds=np.array([[math.dist(a['object_positions']['black_book_1_pos'],b['object_positions']['black_book_1_pos'])*100
                      for a,b in zip(traces[role,'left',i,s],traces[role,'front',i,s])]
                     for i in (0,1) for s in (2026102801,2026102802)])
        x=np.arange(1,121)
        ax.plot(x,np.median(ds,axis=0),color=color,label=role)
        ax.fill_between(x,ds.min(axis=0),ds.max(axis=0),color=color,alpha=.15)
    for h in (40,80,120):ax.axvline(h,color='.6',linestyle=':',linewidth=1)
    ax.set(xlabel='Executed control step',ylabel='Left-vs-Front book XYZ distance (cm)',
           title='Median and range over 4 paired schedules')
    ax.legend(frameon=False)
    fig.suptitle('Original native prefixes: time changes physical state and task phase\nNo policy inference; trajectory differences do not locate a neural mechanism',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.87))
    fig.savefig(output/'source_progress.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.reference,a.output)
