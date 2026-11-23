from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import parallel_reproduction as core

run=Path(__file__).resolve().parent
p=json.loads((run/'protocol.json').read_text())
for name,h in p['sources'].items():
    assert core.digest(run/name)==h,name
for name,h in p['original_sources'].items():
    source=Path(core.original.k21.__file__) if name=='k21.py' else Path(core.original.__file__).with_name(name)
    assert core.digest(source)==h,name
for relative,h in p['validated_dependency_sha256'].items():
    assert core.digest(Path('/home/lzx')/relative)==h,relative
assert p['versions']=={n:importlib.metadata.version(n) for n in p['versions']}
record=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
    packages=p['versions'],torch_cuda=core.torch.version.cuda,
    gpu_names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines(),
    environment={k:os.environ.get(k) for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','MUJOCO_GL','LIBERO_CONFIG_PATH','PYTHONPATH')},
    dependency_sha256=p['validated_dependency_sha256'],original_sources=p['original_sources'],source_parent_commit=p['source_parent_commit'],
    capture_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),queue_source_sha256=core.digest(run/'run_parallel_reproduction.py'),
    analysis_plan_sha256=core.digest(run/'analysis_plan.json'),
    actual_shared_modules={str(Path(m.__file__).resolve()):core.digest(Path(m.__file__)) for m in (core.timing,core.spec.scoring)},
    source_report_location='Retained locally only; not transmitted to server.',
    commands=['K25_REAL_PROJECTION_TESTS=1 PYTHONPATH=. python -m unittest discover -s tests -p test_k25_parallel.py -v','parallel_reproduction.py prepare','CUDA_VISIBLE_DEVICES=0 parallel_reproduction.py calibrate --profile lr_front_1000','CUDA_VISIBLE_DEVICES=1 parallel_reproduction.py calibrate --profile lf_right_1000','parallel_reproduction.py lock','capture_runtime.py','run_parallel_reproduction.py (8 workers;4 per GPU)','summarize_parallel.py .'],
    roles='Matched original parent P supplies mean and full old difference axis; F actual q supplies scalar projection only. No V perturbation.')
assert not (run/'runtime.json').exists()
(run/'runtime.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print('PARALLEL_RUNTIME_VERIFIED',len(p['validated_dependency_sha256']),flush=True)
