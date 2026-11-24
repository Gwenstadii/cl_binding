"""Standalone figures for the matched residual-component intervention."""
import argparse
import gzip
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
    rows=[('whole150','parallel'),('whole150','whole'),('whole400','parallel'),('whole400','whole')]
    goals=['left','front','right','back'];labels=['Projected 1-150','Full residual 1-150','Projected 1-400','Full residual 1-400']
    fig,axes=plt.subplots(1,2,figsize=(11,5.6));fig.subplots_adjust(left=.17,right=.90,top=.82,bottom=.23,wspace=.22)
    for ax,metric,title in zip(axes,['first','terminal'],['Correct first arrival','Final placement at 400']):
        values=np.array([[s['comparisons'][arm][method]['per_goal'][g][metric] for g in goals] for arm,method in rows])
        im=ax.imshow(values,cmap='Blues',vmin=0,vmax=20,aspect='auto')
        ax.set_xticks(range(4),['Left','Front','Right','Back']);ax.set_yticks(range(4),labels if ax is axes[0] else ['']*4)
        ax.set_title(title,loc='left',pad=12,fontweight='bold')
        for i in range(4):
            for j in range(4):ax.text(j,i,str(values[i,j]),ha='center',va='center',fontsize=13,color='white' if values[i,j]>=12 else '#162c3e')
    fig.colorbar(im,ax=axes,fraction=.025,pad=.035,label='Successful starts / 20',ticks=[0,5,10,15,20])
    fig.suptitle('Same parent mean and gain: retain vs remove the residual component',fontsize=12,fontweight='bold',y=.95)
    fig.text(.5,.04,'Same P/F checkpoints, alpha, starts, prompts, noise, windows and full-32D precision.\n1-150 arms return to native F afterwards; all arms run through control 400.\nBack often passes through Front. First arrival and placement are separate endpoints.',ha='center',va='bottom',fontsize=9)
    save(fig,a.out/'matched_components')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),sharey=True);fig.subplots_adjust(left=.07,right=.98,top=.80,bottom=.28,wspace=.15)
    x=np.arange(4)
    for ax,arm,title in zip(axes,['whole150','whole400'],['Intervention 1-150','Intervention 1-400']):
        for j,(metric,label,colour) in enumerate([('first','First arrival','#4584a3'),('terminal','Final placement','#a36c3b')]):
            gains=[len(s['comparisons'][arm]['paired']['per_goal'][g][metric]['gained']) for g in goals]
            losses=[-len(s['comparisons'][arm]['paired']['per_goal'][g][metric]['lost']) for g in goals]
            b=ax.bar(x+(j-.5)*.32,gains,.30,color=colour,label=label);ax.bar_label(b,padding=2,fontsize=9)
            b=ax.bar(x+(j-.5)*.32,losses,.30,color=colour,alpha=.45);ax.bar_label(b,padding=2,fontsize=9)
        ax.axhline(0,color='#303030',linewidth=.8);ax.set_xticks(x,['Left','Front','Right','Back']);ax.set_ylim(-21,21);ax.set_yticks(range(-20,21,5));ax.set_title(title,loc='left',fontweight='bold')
        ax.yaxis.grid(True,color='#deded8',linewidth=.7);ax.set_axisbelow(True)
    axes[0].set_ylabel('Paired starts gained (+) or lost (-)')
    fig.suptitle('Whole residual relative to projection: gains and losses on the same starts',fontsize=12,fontweight='bold',y=.94)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.08),ncol=2,frameon=False)
    fig.text(.5,.025,'Twenty reused physical starts per instruction; bars are paired counts, not independent query samples.',ha='center',fontsize=9)
    save(fig,a.out/'paired_changes')
    # All20 Back starts, without selecting successful examples.
    back={}
    for folder,kind in [(a.summary.parent,'new')]:
        for lane in range(20):
            path=folder/f'new_{lane}.jsonl.gz'
            with gzip.open(path,'rt',encoding='utf-8') as f:
                for r in map(json.loads,f):
                    if r['goal']=='back' and r['arm'] in ['whole150','whole400','parallel150','parallel400']:
                        back[r['arm'],r['init']]=[bool(m&8) for m in r['trace']['success']]
    fig,axes=plt.subplots(2,2,figsize=(11.5,7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.98,top=.85,bottom=.16,wspace=.10,hspace=.25)
    for row,arm in enumerate(['whole150','whole400']):
        for col,method in enumerate(['parallel','whole']):
            key=('parallel150' if row==0 else 'parallel400') if method=='parallel' else arm
            values=np.array([back[key,i] for i in range(800,820)])
            c=s['comparisons'][arm][method]['per_goal']['back']
            assert int(values.any(axis=1).sum())==c['ever'] and int(values[:,-1].sum())==c['terminal']
            ax=axes[row,col];ax.imshow(values,cmap='Blues',vmin=0,vmax=1,interpolation='nearest',aspect='auto',extent=[.5,400.5,819.5,799.5])
            ax.axvline(150.5,color='#b46c39',linewidth=.9,linestyle='--')
            ax.set_title(f'{"Projected" if method=="parallel" else "Whole"} 1-{150 if row==0 else 400}: ever {c["ever"]}/20, final {c["terminal"]}/20',loc='left',fontsize=11,fontweight='bold')
            ax.set_yticks([800,805,810,815,819]);ax.set_xticks([1,100,150,200,300,400])
            if col==0:ax.set_ylabel('Initial-state ID')
            if row==1:ax.set_xlabel('Control step')
    fig.suptitle('Back placement predicate over time on every matched start',fontsize=13,fontweight='bold',y=.96)
    fig.text(.5,.04,'Blue = valid Back placement predicate; white = false. Dashed line marks control 150.\nTransient success does not count as final placement. All 20 starts are retained in each panel.',ha='center',va='bottom',fontsize=10)
    save(fig,a.out/'back_placement_time')
    print('WHOLE4_FIGURES_WRITTEN')
