"""Plot eligible same-state instruction choices, keeping achieved prefixes out."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    audit=json.loads((root/'audit.json').read_text())
    table=audit['per_condition'];pairs=audit['paired_summary']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    keys=[(goal,h) for goal in ('left','front') for h in (40,80,120)]
    columns=[(role,g) for role in ('R0','FT4') for g in ('left','front')]
    for ax,source in zip(axes,('R0','FT4')):
        values=np.full((6,4),np.nan);labels={}
        for i,(prefix,h) in enumerate(keys):
            for j,(role,goal) in enumerate(columns):
                r=next(r for r in table if (r['source_model'],r['prefix_goal'],r['boundary'],r['continuation_model'],r['requested_goal'])==(source,prefix,h,role,goal))
                n=r['pretarget_eligible']
                if n:values[i,j]=r['eligible_first_correct']/n
                labels[i,j]=f"{r['eligible_first_correct']}/{n}" if n else 'N/A'
        cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#dedede')
        ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='auto')
        for (i,j),label in labels.items():
            ax.text(j,i,label,ha='center',va='center',color='white' if np.isfinite(values[i,j]) and values[i,j]>.6 else 'black')
        ax.set(xticks=range(4),xticklabels=['R0: Left','R0: Front','FT4: Left','FT4: Front'],
               yticks=range(6),yticklabels=[f'{g.title()} prefix, {h}' for g,h in keys],
               title=f'{source} source trajectory',xlabel='Continuation model: new instruction')
    fig.suptitle('First target matches NEW instruction, among pre-target eligible states\n2 initial states x 2 noise seeds; gray = already achieved / no eligible states',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.89));fig.savefig(root/'eligible_first_choices.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,source in zip(axes,('R0','FT4')):
        for role,color,shift in [('R0','#287c8e',-.17),('FT4','#bf773c',.17)]:
            values=[];labels=[]
            for prefix,h in keys:
                r=next(r for r in pairs if (r['source_model'],r['prefix_goal'],r['boundary'],r['continuation_model'])==(source,prefix,h,role))
                n=r['pretarget_eligible_pairs'];v=r['eligible_both_instructed']
                values.append(v if n else 0);labels.append(f'{v}/{n}' if n else 'N/A')
            bars=ax.bar(np.arange(6)+shift,values,width=.32,color=color,label=role)
            ax.bar_label(bars,labels=labels,padding=3,fontsize=9)
        ax.set(xticks=range(6),xticklabels=[f'{g[0].upper()}-{h}' for g,h in keys],ylim=(0,4.8),yticks=range(5),
               title=f'{source} source trajectory',xlabel='Prefix goal - handoff step',ylabel='Pairs correct under BOTH instructions')
        ax.legend(frameon=False,loc='upper right')
    fig.suptitle('Same-state dual-instruction test: Left goes Left AND Front goes Front\nFollowing the prefix under both instructions does not pass this test',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.86));fig.savefig(root/'dual_instruction_pairs.png',dpi=170,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    plot(p.parse_args().root)
