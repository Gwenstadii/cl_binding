from datetime import datetime,timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

import whole_four as core

run=Path(__file__).resolve().parent
p=core.ref.old.base.read(run/'protocol.json')
prior=core.ref.old.base.read(core.REFERENCE/'runtime.json')
for n,h in p['sources'].items():assert core.digest(run/n)==h,n
for n,h in p['reference_sources'].items():assert core.digest(core.REFERENCE/n)==h,n
for path,h in p['files'].items():assert core.digest(Path(path))==h,path
for relative,h in prior['dependency_sha256'].items():assert core.digest(Path('/home/lzx')/relative)==h,relative
packages={n:importlib.metadata.version(n) for n in prior['packages']};assert packages==prior['packages']
mods=[core,core.math,core.math.reference,core.ref,core.ref.old,core.ref.old.original,core.ref.old.original.k21,core.ref.old.timing,core.ref.old.spec,core.ref.old.spec.scoring]
actual={str(Path(m.__file__).resolve()):core.digest(Path(m.__file__)) for m in mods}
assert set(prior['actual_modules'].values())<=set(actual.values())
record=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,packages=packages,torch_cuda=core.torch.version.cuda,
    gpu_names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines(),
    sources=p['sources'],actual_modules=actual,dependency_sha256=prior['dependency_sha256'],reference_runtime_sha256=core.digest(core.REFERENCE/'runtime.json'),
    protocol_sha256=core.digest(run/'protocol.json'),plan_sha256=core.digest(run/'analysis_plan.json'),capture_source_sha256=core.digest(Path(__file__)),
    environment={k:os.environ.get(k) for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','MUJOCO_GL','LIBERO_CONFIG_PATH','PYTHONPATH']},
    source_parent_commit=p['source_parent_commit'],new_training=False,new_calibration=False,
    commands=['PYTHONPATH=. python -m unittest discover -s tests -p test_k25_whole4.py -v','whole_four.py prepare','run_whole_four.py:20workers, max4perGPU','capture_runtime.py'])
assert not (run/'runtime.json').exists()
core.ref.old.base.write(run/'runtime.json',record)
print('WHOLE4_RUNTIME_VERIFIED',len(prior['dependency_sha256']),flush=True)
