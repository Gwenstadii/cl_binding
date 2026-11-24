from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

run=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
p=read(run/'protocol.json');r=read(run/'runtime.json')
files=dict(p['files']);files.update(r['actual_modules'])
for n,h in p['sources'].items():files[str(run/n)]=h
for rel,h in r['dependency_sha256'].items():files[str(Path('/home/lzx')/rel)]=h
for path,h in files.items():assert sha256(Path(path).read_bytes()).hexdigest()==h,path
q=read(run/'queue_COMPLETE.json');assert q==dict(episodes=160,units=20,failures=[])
record=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),files=files,queue=q,protocol_sha256=sha256((run/'protocol.json').read_bytes()).hexdigest(),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
assert not (run/'final_inventory.json').exists()
(run/'final_inventory.json').write_text(json.dumps(record,indent=2)+'\n')
print('WHOLE4_FINAL_INVENTORY_VERIFIED',len(files))
