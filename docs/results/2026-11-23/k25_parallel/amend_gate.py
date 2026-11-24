"""Preserve failed tensor-audit attempt; change only audit arithmetic grouping."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import gzip
import shutil

run=Path(__file__).resolve().parent
new=run/'parallel_reproduction_fixed.py'
old=run/'parallel_reproduction.py'
old_bytes=old.read_bytes()
new_bytes=new.read_bytes()
before=b'            c=lab.alpha*(rf[0,:,:7]*dp[0,:,:7]).sum()/dp[0,:,:7].square().sum().clamp_min(1e-30)\n'
after=b'            c1=(rf[0,:,:7]*dp[0,:,:7]).sum()/dp[0,:,:7].square().sum().clamp_min(1e-30)\n            c=lab.alpha*c1\n'
assert old_bytes.count(before)==1 and old_bytes.replace(before,after)==new_bytes
assert not list(run.glob('*_COMPLETE.json'))
folder=run/'technical_attempt_01'
folder.mkdir(exist_ok=False)
for name in ('protocol.json','preparation.json','parallel_reproduction.py','run_parallel_reproduction.py','queue.log'):
    shutil.copyfile(run/name,folder/name)
interrupted=[]
for path in sorted(run.glob('*.jsonl.gz')):
    parsed=0
    try:
        with gzip.open(path,'rt') as stream:
            for line in stream:
                json.loads(line)
                parsed+=1
    except (EOFError,OSError):
        pass
    interrupted.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha256(path.read_bytes()).hexdigest(),complete_rows_readable=parsed))
    path.rename(folder/path.name)
for path in sorted(run.glob('*_[0-7].log')):
    path.rename(folder/path.name)
p=json.loads((run/'protocol.json').read_text())
assert p['sources']['parallel_reproduction.py']==sha256(old_bytes).hexdigest()
assert p['sources']['parallel_reproduction.py']==json.loads((run/'preparation.json').read_text())['sources']['parallel_reproduction.py']
record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),reason='Independent tensor audit used alpha*numerator/denominator; original uses alpha*(numerator/denominator). Float32 grouping can differ. Only expected-value audit fixed; original actual combine2 untouched.',
    old_source_sha256=sha256(old_bytes).hexdigest(),new_source_sha256=sha256(new_bytes).hexdigest(),
    old_protocol_sha256=sha256((run/'protocol.json').read_bytes()).hexdigest(),interrupted=interrupted,
    completed_shards=0,method_or_gain_or_window_changed=False,calibration_source='original preparation source; unchanged calibration logic',
    replacement=dict(before=before.decode(),after=after.decode()))
(run/'source_amendment.json').write_text(json.dumps(record,indent=2)+'\n')
old.write_bytes(new_bytes)
p['sources']['parallel_reproduction.py']=sha256(new_bytes).hexdigest()
p['technical_amendment_sha256']=sha256((run/'source_amendment.json').read_bytes()).hexdigest()
(run/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
(run/'runtime.json').rename(folder/'runtime.json')
(run/'runtime_capture.log').rename(folder/'runtime_capture.log')
print('AUDIT_GROUPING_AMENDED',len(interrupted),'interrupted shards; no complete evaluation shard')
