"""Re-score archived K25 rollouts under requested-goal first native success.

No training, policy inference, simulator calls or alteration of archived rows.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

from native_event_protocol import rescore_native_flags

GOALS = ('left','front','right','back')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(root, output):
    folder=root/'docs/results/2026-11-24/k25_whole4_new'
    previous=root/'docs/results/2026-11-24/k25_parallel4'
    bit_candidates={g:{1,2,4,8} for g in GOALS}
    records=[];sources={}

    def collect(path, dataset, select):
        relative=str(path.relative_to(root)).replace('\\','/')
        digest=sha(path);sources[relative]=digest
        marker=path.with_name(path.name.removesuffix('.jsonl.gz')+'_COMPLETE.json')
        if marker.exists():
            m=json.loads(marker.read_text())
            assert m['rows_sha256']==digest,relative
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if not select(row):continue
                masks=row['trace']['success'];assert len(masks)==400
                first={b:next((i+1 for i,m in enumerate(masks) if m&b),None) for b in (1,2,4,8)}
                for g in GOALS:
                    bit_candidates[g].intersection_update(b for b in (1,2,4,8) if (first[b] is not None)==bool(row['ever'][g]))
                records.append(dict(dataset=dataset,source=relative,init=row['init'],arm=row['arm'],
                    goal=row['goal'],ever=row['ever'],legacy_terminal=row['terminal'],
                    first_by_bit=first,seed=row['seed']))

    for path in sorted(folder.glob('dev_*.jsonl.gz')):
        step=int(path.name.split('_')[1]);collect(path,f'dev_{step}',lambda r:True)
    for path in sorted(folder.glob('new_*.jsonl.gz')):
        collect(path,'formal800',lambda r:True)
    for path in sorted(previous.glob('four_*.jsonl.gz')):
        collect(path,'formal800_parent',lambda r:r['arm']=='parent')
    assert all(len(v)==1 for v in bit_candidates.values()),bit_candidates
    bits={g:next(iter(v)) for g,v in bit_candidates.items()};assert bits['back']==8
    compact=[]
    for row in records:
        first=row['first_by_bit'][bits[row['goal']]]
        assert (first is not None)==bool(row['ever'][row['goal']])
        # This uses the same termination helper that live per-action execution uses.
        stop=first if first is not None else 400
        flags=[{row['goal']:False}]*(stop-1)+[{row['goal']:first is not None}]
        scored=rescore_native_flags(row['goal'],flags)
        compact.append({k:row[k] for k in ('dataset','source','init','arm','goal','seed')}|
                       dict(**scored,legacy_terminal400=bool(row['legacy_terminal'][row['goal']])))

    def group(rows, ids):
        assert len(rows)==20 and {r['init'] for r in rows}==set(ids)
        assert len({r['init'] for r in rows})==20
        times=[r['first_success_step'] for r in rows if r['success']]
        return dict(n=20,success=sum(r['success'] for r in rows),
                    success_ids=[r['init'] for r in rows if r['success']],
                    median_first_success_step=median(times) if times else None,
                    legacy_terminal400=sum(r['legacy_terminal400'] for r in rows))

    development={}
    for dataset in sorted({r['dataset'] for r in compact if r['dataset'].startswith('dev_')},key=lambda x:int(x[4:])):
        development[dataset[4:]]=group([r for r in compact if r['dataset']==dataset],range(640,660))
    formal={}
    for arm in ('parent','native','parallel150','whole150','parallel400','whole400'):
        selected=[r for r in compact if r['dataset'].startswith('formal800') and r['arm']==arm]
        formal[arm]={g:group([r for r in selected if r['goal']==g],range(800,820)) for g in GOALS}
        keyed={(r['init'],r['goal']):r['success'] for r in selected}
        formal[arm]['old_joint_success']=sum(all(keyed[i,g] for g in ('left','front','right')) for i in range(800,820))
        formal[arm]['four_joint_success']=sum(all(keyed[i,g] for g in GOALS) for i in range(800,820))
    assert development['24000']['success']==20 and formal['native']['back']['success']==19
    highest=max(v['success'] for v in development.values())
    result=dict(protocol='K25-native-event-v1',horizon=400,mode='retrospective_reanalysis_no_new_rollouts',
        native_predicate_bits=bits,sources=sources,development=development,formal=formal,
        best_development_success=highest,
        tied_development_steps=[int(k) for k,v in development.items() if v['success']==highest],
        selected_step=24000,model_code='F4-B24',
        new_rollouts=0,new_training_steps=0,
        tie_break='Keep the original latest-to-earliest order on the standard-data continuation:24k is the first passing native-event checkpoint and ties the maximum20/20. Existing separate four-instruction results are reported as reused evidence, not a new untouched test. No claim of unique global optimum over other training branches.',
        rows=compact)
    registry_path=root/'configs/k25_models/F4-B24.json'
    if registry_path.exists():
        registry=json.loads(registry_path.read_text(encoding='utf-8'))
        if registry.get('status')=='ready':
            result['selected_full_checkpoint_model_sha256']=registry['full_checkpoint_model_sha256']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(bits=bits,development={k:v['success'] for k,v in development.items()},
        formal={arm:{g:d[g]['success'] for g in GOALS}|{k:d[k] for k in ('old_joint_success','four_joint_success')} for arm,d in formal.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
