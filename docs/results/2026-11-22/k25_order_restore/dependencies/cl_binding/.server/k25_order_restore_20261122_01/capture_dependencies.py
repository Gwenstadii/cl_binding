import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(run))
import order_restore
sys.path.insert(0, str(run.parent / "k18_training_controls_20261111_01"))
import train_controls
import k1_runtime

ctx = order_restore.sr.km.Ctx("scene2")
ctx.env.close()

roots = (Path("/home/lzx/cl_binding"), Path("/home/lzx/cl_binding_stage2a"))
sources = {Path(m.__file__).resolve() for m in list(sys.modules.values())
           if getattr(m, "__file__", None) and Path(m.__file__).suffix == ".py"}
sources.update(run / n for n in ("order_restore.py", "order_restore_standard.py", "order_restore_parent.py", "run_order_queue.py"))
index = {}
for source in sorted(sources):
    for root in roots:
        if not source.is_relative_to(root):
            continue
        rel = source.relative_to(root)
        if rel.parts[0] not in ("scripts", ".server"):
            break
        label = Path(root.name) / rel
        target = run / "dependencies" / label
        assert target.resolve().is_relative_to(run / "dependencies")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if target.exists() and source != Path(__file__).resolve():
            assert target.read_bytes() == data
        else:
            target.write_bytes(data)
        index[label.as_posix()] = hashlib.sha256(data).hexdigest()
        break
(run / "dependency_sha256.json").write_text(json.dumps(index, indent=2)+"\n", encoding="utf-8")
print("Archived", len(index), "project source dependencies")
