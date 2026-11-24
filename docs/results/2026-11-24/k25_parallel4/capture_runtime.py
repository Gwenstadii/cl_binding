from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import parallel_four as core

run=Path(__file__).resolve().parent
full=core.old.base.read(run/'full_protocol.json')
four=core.old.base.read(run/'four_protocol.json')
previous=core.old.base.read(core.PREVIOUS/'protocol.json')
sources={n:core.digest(run/n) for n in ['parallel_full400.py','parallel_four.py','parallel_subspace.py','run_parallel4.py','capture_runtime.py']}
assert sources['parallel_full400.py']==full['sources']['parallel_full400.py']
assert all(sources[n]==h for n,h in four['sources'].items())
for n,h in previous['sources'].items():assert core.digest(core.PREVIOUS/n)==h,n
for relative,h in previous['validated_dependency_sha256'].items():assert core.digest(Path('/home/lzx')/relative)==h,relative
packages={n:importlib.metadata.version(n) for n in previous['versions']}
assert packages==previous['versions']
record=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,packages=packages,
    torch_cuda=core.torch.version.cuda,gpu_names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines(),
    sources=sources,previous_sources=previous['sources'],original_sources=previous['original_sources'],dependency_sha256=previous['validated_dependency_sha256'],
    full_protocol_sha256=core.digest(run/'full_protocol.json'),four_protocol_sha256=core.digest(run/'four_protocol.json'),
    plan_sha256=core.digest(run/'analysis_plan.json'),capture_source_sha256=core.digest(Path(__file__)),
    environment={k:os.environ.get(k) for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','MUJOCO_GL','LIBERO_CONFIG_PATH','PYTHONPATH']},
    actual_modules={str(Path(m.__file__).resolve()):core.digest(Path(m.__file__)) for m in [core.old,core.old.original,core.old.original.k21,core.old.timing,core.old.spec,core.old.spec.scoring,core.math]},
    source_parent_commit='e61358e0d1c203aea7d8cb57f0dfcadb986a8711',new_training=False,
    commands=['parallel_full400.py prepare','run_parallel4.py:52 jobs, initial3workers/GPU, max4/GPU after calibration',
        'PYTHONPATH=. python -m unittest discover -s tests -p test_k25_parallel4.py -v','parallel_four.py prepare',
        'CUDA_VISIBLE_DEVICES=0 parallel_four.py calibrate','parallel_four.py lock','capture_runtime.py'])
assert not (run/'runtime.json').exists()
core.old.base.write(run/'runtime.json',record)
print('PARALLEL4_RUNTIME_VERIFIED',len(previous['validated_dependency_sha256']),flush=True)
