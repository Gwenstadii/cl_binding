"""Validate the unchanged server dependencies without duplicating their archive."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

run = Path(__file__).resolve().parent
previous = run.parent / "k25_order_restore_20261122_01"
manifest = json.loads((previous / "dependency_sha256.json").read_text())
sys.path[:0] = [str(run), str(previous)]
import v_timing
loaded = {str(Path(module.__file__).resolve()) for module in list(sys.modules.values())
          if getattr(module, "__file__", None)}
checked = {}
unused_changes = {}
for relative, expected in manifest.items():
    path = Path("/home/lzx") / relative
    actual = sha256(path.read_bytes()).hexdigest()
    if str(path.resolve()) in loaded:
        assert actual == expected, relative
        checked[relative] = actual
    elif actual != expected:
        unused_changes[relative] = dict(previous_snapshot_sha256=expected, current_sha256=actual,
            imported_in_this_run=False)
custom = {p for p in loaded if ("/cl_binding/.server/" in p or "/cl_binding_stage2a/scripts/" in p)
          and not p.startswith(str(run)+"/")}
assert custom == {str((Path("/home/lzx") / relative).resolve()) for relative in checked}, custom
protocol = json.loads((run / "protocol.json").read_text())
for filename, key in (("v_timing.py", "source_sha256"), ("v_timing_protocol.py", "spec_sha256")):
    assert sha256((run / filename).read_bytes()).hexdigest() == protocol[key]
old_runtime = json.loads((previous / "runtime.json").read_text())
versions = {key: importlib.metadata.version(key) for key in old_runtime["packages"]}
assert versions == old_runtime["packages"]
record = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
    executable=sys.executable, packages=versions,
    environment={key: os.environ.get(key) for key in old_runtime["environment"]},
    runtime_checkout_commit=subprocess.check_output(["git", "-C", "/home/lzx/cl_binding_stage2a", "rev-parse", "HEAD"], text=True).strip(),
    dependency_archive="../k25_order_restore/dependencies", validated_dependency_sha256=checked,
    unused_previous_helper_differences=unused_changes,
    capture_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    commands=["v_timing.py prepare", "run_v_timing.py --workers 4 --lanes 8", "summarize_v_timing.py ."],
    concurrency_note="capacity.json changes worker capacity only; changes are logged in queue.log")
dest = run / "runtime.json"
assert not dest.exists()
dest.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
print("RUNTIME_VALIDATED", len(checked), versions)
