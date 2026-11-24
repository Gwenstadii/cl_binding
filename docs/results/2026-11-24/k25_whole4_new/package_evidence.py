"""Package final evidence only, with independent training-log consistency audit."""
import gzip
from hashlib import file_digest
import importlib.util
import json
import math
from pathlib import Path
import sys
import tarfile
import contextlib
with contextlib.redirect_stdout(sys.stderr):
    import whole4_new as c

run=Path(__file__).resolve().parent
assert (run/'COMPLETE.json').exists() and (run/'final_identity.json').exists() and not (run/'FAILED.json').exists()
history=run/'history';history.mkdir(exist_ok=True)
cfg=c.read(c.TRAIN/'config.json');m=c.read(c.TRAIN/'training/manifest.json')
for name,value in m['source_files'].items():assert c.digest(name)==value,name
conversion=c.read(Path(cfg['dataset'])/'r0_conversion.json');assert c.digest(Path(cfg['dataset'])/'r0_conversion.json')==m['dataset_conversion_sha256']
spec=importlib.util.spec_from_file_location('training_protocol',c.TRAIN/'back_training_protocol.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);ranges,_=t.phase_ranges(conversion['episodes'],cfg['tail_fraction'])
source=c.TRAIN/'training/train.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,24001))
for r in rows:
    assert r['indices']==t.sample_indices(ranges,cfg['seed'],r['step']-1,cfg['batch'])
    assert r['lr']==t.learning_rate(r['step'],cfg['steps'],cfg['warmup'],cfg['peak_lr'],cfg['floor_lr'])
    assert all(math.isfinite(r[k]) for k in ['loss','grad_norm','uniform_loss','terminal_phase_loss'])
with gzip.open(history/'train.jsonl.gz','wb') as f:f.write(source.read_bytes())
for dest,path in [('training_config.json',c.TRAIN/'config.json'),('training_manifest.json',c.TRAIN/'training/manifest.json'),('training_COMPLETE.json',c.TRAIN/'training/COMPLETE.json')]:
    (history/dest).write_bytes(path.read_bytes())
for step in c.read(c.TRAIN/'training/COMPLETE.json')['saved_steps']:
    (history/f'READY_{step:06d}.json').write_bytes((c.TRAIN/f'training/{step:06d}/READY.json').read_bytes())
c.write(history/'training_audit.json',dict(steps=len(rows),indices_and_schedule_exact=True,finite_losses_and_gradients=True,
    first_loss=rows[0]['loss'],last_loss=rows[-1]['loss'],source_sha256=c.digest(source)))
files=[p for p in run.iterdir() if p.is_file() and p.suffix in ['.py','.json','.log','.gz']]
files+=list(history.iterdir())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for p in sorted(files):archive.add(p,arcname=str(p.relative_to(run)),recursive=False)
