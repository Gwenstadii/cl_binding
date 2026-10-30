"""Paired book/eef trajectories, controller directions, and grasp context."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot(root):
    cfg=json.loads((root/'registered_config.json').read_text());sites=cfg['movement_geometry']['sites']
    episodes=[json.loads(x) for x in (root/'evaluation/episodes.jsonl').read_text().splitlines()]
    byid={e['episode_id']:e for e in episodes};traces={e['episode_id']:[] for e in episodes}
    with gzip.open(root/'evaluation/events.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['step']:traces[r['episode_id']].append(r)
    colors={'native':'#666666','handoff60':'#177f9c'}
    lookup={(e['mode'],e['requested_goal'],e['init_index'],e['noise_seed']):traces[e['episode_id']] for e in episodes}
    def geometry(ax):
        for g,s in sites.items():
            c=s['centre'];ext=[sum(abs(s['rotation'][d][j])*s['half_size'][j] for j in range(3)) for d in (0,1)]
            ax.add_patch(Rectangle((c[0]-ext[0],c[1]-ext[1]),2*ext[0],2*ext[1],fill=False,color='.8',linewidth=.8))
            ax.text(c[0],c[1],g,color='.65',ha='center',fontsize=8)
    def path(ax,goal,index,seed,entity):
        geometry(ax)
        for mode in ('native','handoff60'):
            t=lookup[mode,goal,index,seed]
            points=np.array([r['eef_position'] if entity=='eef' else r['object_positions']['black_book_1_pos'] for r in t])
            ax.plot(points[:60,0],points[:60,1],color=colors[mode],linestyle='--',linewidth=1)
            ax.plot(points[59:,0],points[59:,1],color=colors[mode],linewidth=1.4,label='FT4 native' if mode=='native' else 'R0 40 -> FT4')
            ax.scatter(*points[59,:2],marker='*',s=40,color=colors[mode],zorder=4)
            ax.annotate('',xy=points[69,:2],xytext=points[59,:2],arrowprops={'arrowstyle':'->','color':colors[mode],'lw':2.5})
        ax.set(xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(13,12))
    for y,g in enumerate(('left','front','right')):
        for x,entity in enumerate(('book','eef')):
            path(axes[y,x],g,0,2026102801,entity);axes[y,x].set_title(f'{g.upper()}: {entity} XY')
        ax=axes[y,2]
        for mode in ('native','handoff60'):
            t=lookup[mode,g,0,2026102801];steps=np.arange(1,401)
            for entity,style in [('book','-'),('eef',':')]:
                z=[r['object_positions']['black_book_1_pos'][2] if entity=='book' else r['eef_position'][2] for r in t]
                ax.plot(steps,z,color=colors[mode],linestyle=style,label=mode+' '+entity)
        ax.axvline(40,color='#bf773c',linestyle='--');ax.set(xlim=(20,120),xlabel='Executed step',ylabel='World Z (m)',title=f'{g.upper()}: height around handoff')
    axes[0,0].legend(frameon=False,fontsize=8);axes[0,2].legend(frameon=False,fontsize=8)
    fig.suptitle('SAME instruction throughout; init 0, noise 2026102801\nDashed = steps 1-60; star = step 60; arrow = actual displacement 60-70; Right R0 prefix is untrained',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/'paired_paths_example.png',dpi=160,bbox_inches='tight');plt.close(fig)
    for entity in ('book','eef'):
        fig,axes=plt.subplots(4,3,figsize=(13,15))
        for y,(index,seed) in enumerate([(i,s) for i in (0,1) for s in (2026102801,2026102802)]):
            for x,g in enumerate(('left','front','right')):
                path(axes[y,x],g,index,seed,entity);axes[y,x].set_title(f'{g.upper()} | init {index} | noise {seed}')
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle(f'All paired {entity} XY trajectories: native FT4 vs same-instruction R0 first 40 -> FT4\nStars = step 60; arrows = actual motion 60-70. Right prefix is exploratory.',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(root/f'all_pairs_{entity}.png',dpi=150,bbox_inches='tight');plt.close(fig)
    audit=json.loads((root/'audit.json').read_text());ds=audit['direction_rows']
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for x,field in enumerate(('command_projection_requested_cm','eef_distance_reduction_requested_cm','book_distance_reduction_requested_cm')):
        ax=axes[x]
        for offset,mode in [(-.15,'native'),(.15,'handoff60')]:
            for j,g in enumerate(('left','front','right')):
                values=[r[field] for r in ds if r['mode']==mode and r['requested_goal']==g and r['window']==10]
                ax.scatter(np.full(4,j+offset),values,color=colors[mode],alpha=.8,label=mode if j==0 else None)
                ax.plot([j+offset-.08,j+offset+.08],[np.median(values)]*2,color=colors[mode],lw=2)
        ax.axhline(0,color='.7',lw=1);ax.set(xticks=range(3),xticklabels=['Left','Front','Right'],ylabel='cm (positive toward / closer)',title={'command_projection_requested_cm':'Sum of controller offsets: projection',
            'eef_distance_reduction_requested_cm':'Observed eef distance reduction','book_distance_reduction_requested_cm':'Observed book distance reduction'}[field])
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('First 10 actions after step 60; 4 paired cases per route / instruction\nController-offset sum is NOT integrated displacement; Right prefix R0 was not trained on Right.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.85));fig.savefig(root/'command_vs_motion.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);plot(p.parse_args().root)
