"""Reuse or fetch LIBERO simulation meshes/textures, never policy or demo data."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".server"
STATE.mkdir(exist_ok=True)
repo_id = "lerobot/libero-assets"
groups = ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects")
spec = importlib.util.find_spec("libero")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Install hf-libero first")
installed = Path(next(iter(spec.submodule_search_locations))) / "libero" / "assets"
cached = Path.home() / ".cache" / "libero" / "assets"


def populated(path):
    return all((path / group).is_dir() and any((path / group).rglob("*.xml")) for group in groups)


record_path = STATE / "libero_assets.json"
record = json.loads(record_path.read_text()) if record_path.exists() else {}
existing = next((path for path in (installed, cached) if populated(path) and not (path == cached and record.get("status") == "downloading")), None)
if existing:
    record.update(repo_id=repo_id, path=str(existing), status="reuse")
else:
    from huggingface_hub import HfApi, snapshot_download

    revision = record.get("revision") or HfApi(token=False).dataset_info(repo_id).sha
    # Save revision before transfer so an interrupted run resumes the same snapshot.
    record.update(repo_id=repo_id, revision=revision, path=str(cached), status="downloading")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=cached, token=False, force_download=False, max_workers=2)
    if not populated(cached):
        raise RuntimeError("LIBERO simulation assets are incomplete")
    record["status"] = "ready"
record_path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
