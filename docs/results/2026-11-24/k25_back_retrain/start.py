"""Start only the hash-verified requested Back training, detached from SSH."""
from datetime import datetime,timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess

run=Path('/home/lzx/cl_binding/.server/k25_back_retrain_20261124_01')
assert Path.cwd()==run and not (run/'LAUNCHED.json').exists() and not (run/'training').exists()
expected=json.loads((run/'upload_manifest.json').read_text())
for name,digest in expected.items():
    with (run/name).open('rb') as f:assert file_digest(f,'sha256').hexdigest()==digest,name
with (run/'console.log').open('xb') as output:
    process=subprocess.Popen(['bash',str(run/'launch.sh')],stdin=subprocess.DEVNULL,
        stdout=output,stderr=subprocess.STDOUT,start_new_session=True,cwd=run)
record=dict(pid=process.pid,launched_at_utc=datetime.now(timezone.utc).isoformat(),
    command=['bash',str(run/'launch.sh')],gpu=0,session_detached=True,sources_sha256=expected,
    source_commit_before_training=json.loads((run/'config.json').read_text())['source_parent_commit'])
(run/'LAUNCHED.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
