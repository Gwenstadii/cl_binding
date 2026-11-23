"""Standalone scientific figure for the single-method K22 reproduction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('summary',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    s=json.loads(args.summary.read_text())
    profiles=list(s['profiles'])
    labels=['LR -> Front\n1k','LF -> Right\n1k','LR -> Front\n12k','LF -> Right\n12k']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True)
    fig.subplots_adjust(left=.06,right=.985,top=.84,bottom=.29,wspace=.10)
    colours=['#33749a','#b56d31','#446d51']
    x=np.arange(4)
    for ax,metric,title in zip(axes,['first','terminal'],['Correct first arrival','Final placement at step 400']):
        for j,(g,name,colour) in enumerate(zip(['A','B','joint'],['A: Left','B: Right / Front','Both on same reset'],colours)):
            values=[s['profiles'][p]['counts'][g][metric] for p in profiles]
            bars=ax.bar(x+(j-1)*.24,values,.22,color=colour,label=name)
            ax.bar_label(bars,padding=3,fontsize=9)
        ax.set_xticks(x,labels)
        ax.set_ylim(0,23)
        ax.set_yticks(range(0,21,5))
        ax.set_title(title,loc='left',fontweight='bold',pad=12)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True,color='#deded8',linewidth=.7)
    axes[0].set_ylabel('Successful resets / 20')
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.105),ncol=3,frameon=False)
    fig.suptitle('Parent mean + parallel projection of the actual continued-policy residual',fontsize=13,fontweight='bold',y=.97)
    fig.text(.5,.025,'Only controls 1-150 are intervened; continued policy controls 151-400.\nSame 20 reused resets per profile; 1k = existing branches, 12k = original endpoints. Different continuations, not a depth sweep.',fontsize=9,ha='center',va='bottom')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:
        fig.savefig(args.output.with_suffix('.'+ext),dpi=190)
    plt.close(fig)
    print('PARALLEL_FIGURE_WRITTEN',args.output)
