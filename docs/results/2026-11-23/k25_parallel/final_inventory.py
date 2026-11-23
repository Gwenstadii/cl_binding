"""Read-only end-of-run inventory; no policy or simulation execution."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

run=Path(__file__).resolve().parent
p=json.loads((run/'protocol.json').read_text())
seen={}
def check(path,expected):
    path=Path(path)
    if str(path) not in seen:
        seen[str(path)]=sha256(path.read_bytes()).hexdigest()
    assert seen[str(path)]==expected,str(path)
for profile,cfg in p['models'].items():
    check(Path(cfg['parent'])/'model.safetensors',cfg['parent_sha256'])
    path=Path(cfg['executor'])
    if profile.endswith('12000'): path=path/'model.safetensors'
    check(path,cfg['executor_sha256'])
for name,h in p['sources'].items(): check(run/name,h)
for relative,h in p['validated_dependency_sha256'].items(): check(Path('/home/lzx')/relative,h)
q=json.loads((run/'queue_COMPLETE.json').read_text())
assert q==dict(episodes=160,units=32,failures=[])
out=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),sha256=seen,queue=q,protocol_sha256=sha256((run/'protocol.json').read_bytes()).hexdigest(),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
assert not (run/'final_inventory.json').exists()
(run/'final_inventory.json').write_text(json.dumps(out,indent=2)+'\n')
print('PARALLEL_FINAL_INVENTORY_VERIFIED',len(seen))
