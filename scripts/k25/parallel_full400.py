"""Change only the stopping boundary of the archived K22 intervention."""
import argparse
from collections import defaultdict
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys

RUN=Path(__file__).resolve().parent
PREVIOUS=RUN.parent/'k25_parallel_20261123_01'
sys.path.insert(0,str(PREVIOUS))
import parallel_reproduction as old


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def full_episode():
    original=inspect.getsource(old.episode)
    edits={'if block < 15:':'if block < 40:', 'window=[0,150]':'window=[0,400]', 'case["z0"],0,150)':'case["z0"],0,400)'}
    modified=original
    for before,after in edits.items():
        assert modified.count(before)==1,before
        modified=modified.replace(before,after)
    namespace=dict(old.__dict__)
    exec(compile(modified,'<archived_episode_full400>','exec'),namespace)
    return namespace['episode'],dict(edits=edits,original_function_sha256=sha256(original.encode()).hexdigest(),modified_function_sha256=sha256(modified.encode()).hexdigest())


def prepare():
    assert not (RUN/'full_protocol.json').exists()
    p=old.base.read(PREVIOUS/'protocol.json')
    for n,h in p['sources'].items(): assert digest(PREVIOUS/n)==h,n
    _,patch=full_episode()
    old.base.write(RUN/'full_protocol.json',dict(previous_protocol_sha256=digest(PREVIOUS/'protocol.json'),previous_sources=p['sources'],
        models=p['models'],gains=p['gains'],geometry=p['geometry'],sources={Path(__file__).name:digest(Path(__file__))},
        expected=160,profiles=p['profiles'],ids=p['ids'],window=[0,400],horizon=400,patch=patch,
        source_parent_commit='e61358e0d1c203aea7d8cb57f0dfcadb986a8711',plan_sha256=digest(RUN/'analysis_plan.json')))
    old.base.write(RUN/'cases.json',old.base.read(PREVIOUS/'cases.json'))
    print('FULL400_LOCKED',160,flush=True)


def worker(profile,lane):
    p=old.base.read(RUN/'full_protocol.json')
    assert digest(Path(__file__))==p['sources'][Path(__file__).name]
    assert digest(PREVIOUS/'protocol.json')==p['previous_protocol_sha256']
    original=old.base.read(PREVIOUS/'protocol.json')
    for n,h in p['previous_sources'].items(): assert digest(PREVIOUS/n)==h,n
    episode,patch=full_episode()
    assert patch==p['patch']
    lab=old.load(profile,original)
    lab.alpha=p['gains'][profile]['alpha']
    cases=[r for r in old.base.read(RUN/'cases.json') if r['init'] in old.spec.ids(lane)]
    gate=old.gate(lab,cases[0])
    selected=set(old.base.names())
    runtimes=(lab.rt.executer,lab.rt.parent)
    def fingerprints():
        return [old.tensors_hash({n:v.detach().cpu() for n,v in rt.models['F'].named_parameters() if n in selected}) for rt in runtimes]
    before=fingerprints()
    dest=RUN/f'full_{profile}_{lane}.jsonl.gz'
    assert not dest.exists()
    count=0
    try:
        with old.torch.inference_mode(),old.torch.autocast('cuda',dtype=old.torch.bfloat16):
            with old.gzip.open(dest,'xt',encoding='utf-8',compresslevel=5) as f:
                for case in cases:
                    for goal in 'AB':
                        row=episode(lab,case,goal)
                        row['profile']=profile
                        row['experiment']='full400'
                        f.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');f.flush()
                        count+=1
                        print('FULL400',profile,lane,case['init'],goal,row['first'],row['terminal'][row['target']],round(row['seconds'],2),flush=True)
    finally:
        lab.ctx.env.close()
    assert before==fingerprints() and count==2*len(cases)
    old.base.write(RUN/f'full_{profile}_{lane}_COMPLETE.json',dict(profile=profile,lane=lane,ids=old.spec.ids(lane),episodes=count,
        gates=gate,weights_fingerprints=before,weights_unchanged=True,normalizers=lab.normalizers,
        rows_sha256=digest(dest),protocol_sha256=digest(RUN/'full_protocol.json'),source_sha256=digest(Path(__file__))))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','worker']);a.add_argument('--profile',choices=old.spec.PROFILES);a.add_argument('--lane',type=int)
    args=a.parse_args();old.torch.set_num_threads(2)
    if args.mode=='prepare':prepare()
    else:worker(args.profile,args.lane)
