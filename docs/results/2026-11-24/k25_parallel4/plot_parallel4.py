"""Standalone scientific figures for duration and four-instruction evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig,path):
    for ext in ['png','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=190)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    s=json.loads(a.summary.read_text(encoding='utf-8'));a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    labels=['LR -> Front\n1k','LF -> Right\n1k','LR -> Front\n12k','LF -> Right\n12k'];profiles=list(s['timing']);x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(11,5.3),sharey=True)
    fig.subplots_adjust(left=.07,right=.98,top=.83,bottom=.28,wspace=.13)
    for ax,metric,title in zip(axes,['first','terminal'],['Both old goals: correct first arrival','Both old goals: final placement']):
        for j,(arm,name,colour) in enumerate([('parallel150','Parallel 1-150, then native F','#4384a4'),('parallel400','Parallel 1-400','#b46c39')]):
            values=[s['timing'][k][arm]['joint'][metric] for k in profiles]
            bars=ax.bar(x+(j-.5)*.34,values,.32,label=name,color=colour);ax.bar_label(bars,padding=3,fontsize=10)
        ax.set_xticks(x,labels);ax.set_ylim(0,23);ax.set_yticks(range(0,21,5));ax.set_title(title,loc='left',fontsize=11,fontweight='bold')
        ax.yaxis.grid(True,color='#deded8',linewidth=.7);ax.set_axisbelow(True)
    axes[0].set_ylabel('Paired starts / 20')
    fig.suptitle('Only the intervention cutoff changes: 150 vs 400 controls',fontsize=13,fontweight='bold',y=.96)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,frameon=False)
    fig.text(.5,.025,'Same models, gain, prompts, starts and noise. First 150 controls match bitwise.\n1k existing branches and 12k original endpoints are different continuation streams.',ha='center',va='bottom',fontsize=9)
    save(fig,a.out/'duration')
    fig,axes=plt.subplots(1,2,figsize=(10.5,5.6));fig.subplots_adjust(left=.12,right=.90,top=.82,bottom=.24,wspace=.27)
    arms=['parent','native','parallel150','parallel400'];goals=['left','front','right','back']
    for ax,metric,title in zip(axes,['first','terminal'],['Correct first arrival','Final placement at 400']):
        values=np.array([[s['four'][arm]['per_goal'][g][metric] for g in goals] for arm in arms])
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=20,aspect='auto')
        ax.set_xticks(range(4),['Left','Front','Right','Back'])
        ax.set_yticks(range(4),['Parent P','Native F','Parallel 1-150','Parallel 1-400'] if ax is axes[0] else ['']*4)
        ax.set_title(title,loc='left',fontweight='bold',pad=12)
        for i in range(4):
            for j in range(4):ax.text(j,i,str(values[i,j]),ha='center',va='center',fontsize=13,color='white' if values[i,j]>=12 else '#172d40')
    fig.colorbar(im,ax=axes,location='right',fraction=.025,pad=.035,label='Successful starts / 20',ticks=[0,5,10,15,20])
    fig.suptitle('Three old goals + Back: continuous projection on the parent contrast plane',fontsize=12,fontweight='bold',y=.95)
    fig.text(.5,.04,'P supplies only Left/Front/Right references; all four prompts use the same projector.\nBack may pass through Front before reaching Back. First arrival and final placement are separate.\nSame 20 reused resets; P has no Back training. Existing Back-3000 weights are reused.',ha='center',va='bottom',fontsize=9)
    save(fig,a.out/'four_commands')
    print('PARALLEL4_FIGURES_WRITTEN')
