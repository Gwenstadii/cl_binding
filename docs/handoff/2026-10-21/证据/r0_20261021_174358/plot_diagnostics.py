"""Standalone evidence figure; run using the existing server matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

control=Path('/home/lzx/cl_binding/.server/r0_20261021_174358')
train=Path('/home/lzx/cl_binding/outputs/r0/r0_20261021_174358/train_steps.jsonl')
rows=[json.loads(line) for line in train.read_text().splitlines()]
screens=[json.loads(p.read_text()) for p in sorted(control.glob('eval_*_screen/evaluation.json'))]
fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
window=100
x=[r['step'] for r in rows[window-1:]]
loss=[sum(r['loss'] for r in rows[i-window+1:i+1])/window for i in range(window-1,len(rows))]
axes[0].plot(x,loss,color='#2a6fbb',linewidth=1.7)
axes[0].set(xlabel='Optimizer step',ylabel='Flow-matching loss (100-step mean)',title='Training loss falls')
sx=[int(Path(s['checkpoint']).name) for s in screens]
for goal,color in zip(('front','left','right'),('#ef8a23','#2c996b','#6b64ba')):
    axes[1].plot(sx,[s['counts'][goal]['correct'] for s in screens],marker='o',label=goal,color=color)
axes[1].axhline(3,color='gray',linestyle='--',linewidth=1)
axes[1].set(xlabel='Checkpoint step',ylabel='Correct at final step / 3 per goal',
            title='Closed-loop screening remains unstable',ylim=(-.15,3.25),yticks=[0,1,2,3])
axes[1].legend(loc='upper right')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Bounded R0 development: completed, not qualified',fontsize=13)
fig.savefig(control/'r0_diagnostics.png',dpi=180)
fig.savefig(control/'r0_diagnostics.pdf')
print(control/'r0_diagnostics.png')
