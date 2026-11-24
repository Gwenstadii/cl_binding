from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path

run=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
full,four,runtime=(read(run/n) for n in ['full_protocol.json','four_protocol.json','runtime.json'])
files=dict(four['files'])
files.update(runtime['actual_modules'])
for name,h in runtime['sources'].items():files[str(run/name)]=h
for relative,h in runtime['dependency_sha256'].items():files[str(Path('/home/lzx')/relative)]=h
for cp in full['models'].values():
    files[cp['parent']+'/model.safetensors']=cp['parent_sha256']
    path=Path(cp['executor'])
    if path.is_dir():path=path/'model.safetensors'
    files[str(path)]=cp['executor_sha256']
for path,h in files.items():assert sha256(Path(path).read_bytes()).hexdigest()==h,path
queue=read(run/'queue_COMPLETE.json');assert queue==dict(episodes=480,units=52,failures=[])
out=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),verified_files=files,queue=queue,
    full_protocol_sha256=sha256((run/'full_protocol.json').read_bytes()).hexdigest(),four_protocol_sha256=sha256((run/'four_protocol.json').read_bytes()).hexdigest(),
    source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
assert not (run/'final_inventory.json').exists()
(run/'final_inventory.json').write_text(json.dumps(out,indent=2)+'\n')
print('PARALLEL4_FINAL_INVENTORY_VERIFIED',len(files))
