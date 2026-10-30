import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot(root):
    eps=[json.loads(x) for x in (root/'evaluation/episodes.jsonl').read_text().splitlines()]
    traces={e['episode_id']:[] for e in eps}
    with gzip.open(root/'evaluation/events.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['step']:traces[r['episode_id']].append(r)
    cfg=json.loads((root/'registered_config.json').read_text());sites=cfg['movement_geometry']['sites']
    colors={'left':'#267e9b','front':'#c17a32','right':'#8659a1'}
    fig,axes=plt.subplots(2,4,figsize=(17,9))
    keys=[(i,s) for i in (0,1) for s in (2026102801,2026102802)]
    for col,(index,seed) in enumerate(keys):
        for row,entity in enumerate(('book','eef')):
            ax=axes[row,col]
            for g,site in sites.items():
                c=site['centre'];ext=[sum(abs(site['rotation'][d][j])*site['half_size'][j] for j in range(3)) for d in (0,1)]
                ax.add_patch(Rectangle((c[0]-ext[0],c[1]-ext[1]),2*ext[0],2*ext[1],fill=False,color='.8'))
                ax.text(c[0],c[1],g,color='.65',ha='center',fontsize=8)
            for g,color in colors.items():
                ep=next(e for e in eps if e['requested_goal']==g and e['init_index']==index and e['noise_seed']==seed)
                t=traces[ep['episode_id']];points=np.array([r['object_positions']['black_book_1_pos'] if entity=='book' else r['eef_position'] for r in t])
                if g=='left':
                    ax.plot(points[:80,0],points[:80,1],'--',color='.6',linewidth=1)
                    ax.scatter(*points[79,:2],color='black',marker='*',s=85)
                ax.plot(points[79:,0],points[79:,1],color=color,label=g)
                ax.annotate('',xy=points[89,:2],xytext=points[79,:2],arrowprops={'arrowstyle':'->','color':color,'lw':2})
            ax.set(title=f'{entity} | init {index}, noise {seed}',xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
    axes[0,0].legend(frameon=False)
    fig.suptitle('SAME saved R0-Left step80 state -> FT4 Left / Front / Right\nBlack star = identical handoff; arrows = actual motion 80-90; no R0 hidden state transferred',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(root/'three_instruction_paths.png',dpi=165,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,(index,seed) in zip(axes,keys):
        for g,color in colors.items():
            ep=next(e for e in eps if e['requested_goal']==g and e['init_index']==index and e['noise_seed']==seed)
            t=traces[ep['episode_id']];ax.plot(range(80,161),[t[k-1]['object_positions']['black_book_1_pos'][1] for k in range(80,161)],color=color,label=g)
        ax.set(title=f'init {index}, noise {seed}',xlabel='Step',ylabel='Book world Y (m)')
    axes[0].legend(frameon=False);fig.suptitle('Same start: instruction-dependent lateral trajectories',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(root/'instruction_y_paths.png',dpi=165,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);plot(p.parse_args().root)
