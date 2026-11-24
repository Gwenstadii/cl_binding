from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import mu_swap as core

run=Path(__file__).resolve().parent
p=json.loads((run/"protocol.json").read_text())
for name,expected in p["sources"].items():
    assert core.digest(run/name) == expected,name
for relative,expected in p["validated_dependency_sha256"].items():
    assert core.digest(Path("/home/lzx")/relative) == expected,relative
assert p["versions"] == {name:importlib.metadata.version(name) for name in p["versions"]}
record=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
    packages=p["versions"],torch_cuda=core.torch.version.cuda,
    gpu_names=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines(),
    environment={k:os.environ.get(k) for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","MUJOCO_GL","LIBERO_CONFIG_PATH","PYTHONPATH")},
    dependency_sha256=p["validated_dependency_sha256"],
    source_parent_commit=p["parent_commit"],capture_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    queue_source_sha256=core.digest(run/"run_mu_swap.py"),analysis_plan_sha256=core.digest(run/"analysis_plan.json"),
    actual_shared_modules={str(Path(m.__file__).resolve()):core.digest(Path(m.__file__)) for m in (core.timing,core.spec.temporal)},
    commands=["python -m unittest discover -s tests -p test_k25_mu_swap.py -v","mu_swap.py prepare","capture_runtime.py","run_mu_swap.py --workers 4; bounded capacity subsequently up to8 after memory check","summarize_mu_swap.py ."],
    model_roles="Two independent runtimes: late executor F1000 and parent common-response donor P0; each runtime's internal role is F.")
path=run/"runtime.json"
assert not path.exists()
path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print("MU_RUNTIME_VERIFIED",len(p["validated_dependency_sha256"])+2,"shared project modules",flush=True)
