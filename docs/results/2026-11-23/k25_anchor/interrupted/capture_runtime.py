from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import anchor_check as core

base = Path(__file__).resolve().parent
p = json.loads((base/"protocol.json").read_text())
for relative,expected in p["validated_dependency_sha256"].items():
    assert core.digest(Path("/home/lzx")/relative) == expected,relative
assert core.digest(Path(core.timing.__file__)) == p["timing_source_sha256"]
assert core.digest(Path(core.spec.temporal.__file__)) == p["timing_spec_sha256"]
assert core.digest(Path(core.__file__)) == p["source_sha256"]
assert core.digest(Path(core.spec.__file__)) == p["spec_sha256"]
versions = {name:importlib.metadata.version(name) for name in p["versions"]}
assert versions == p["versions"]
record = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
    packages=versions,torch_cuda=core.torch.version.cuda,
    gpu_names=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).splitlines(),
    environment={k:os.environ.get(k) for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","MUJOCO_GL","LIBERO_CONFIG_PATH","PYTHONPATH")},
    dependency_sha256=p["validated_dependency_sha256"],
    extra_actual_modules={str(Path(m.__file__).resolve()):core.digest(Path(m.__file__)) for m in (core.timing,core.spec.temporal)},
    runtime_checkout_commit=subprocess.check_output(["git","-C","/home/lzx/cl_binding_stage2a","rev-parse","HEAD"],text=True).strip(),
    source_parent_commit="675be9127d3db8cfe0763488463ee0be963977dd",
    capture_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    commands=["anchor_check.py prepare","run_anchor_check.py --workers 8","summarize_anchor.py ."],
    startup_note="initial queue import failed before any episode; byte-identical shared protocol was then copied; locked physical experiment sources unchanged")
dest = base/"runtime.json"
assert not dest.exists()
dest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("RUNTIME_VALIDATED",len(p["validated_dependency_sha256"])+2,"project modules",record["gpu_names"])
