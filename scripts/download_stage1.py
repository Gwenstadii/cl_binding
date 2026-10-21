"""Plan or fetch only missing stage-one assets; default is a local-only plan."""

import argparse
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".server"
MODEL_ROOT = Path.home() / "hf_models"
BASE_ID = "lerobot/smolvla_base"
REFERENCE_ID = "lerobot/smolvla_libero"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DATA_ID = "yifengzhu-hf/LIBERO-datasets"
GOALS = ("front", "left", "right", "back")
FILENAMES = [
    f"STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_{goal}_compartment_of_the_caddy_demo.hdf5"
    for goal in GOALS
]


def complete_model(directory: Path) -> bool:
    """Reject LFS pointers and truncated single-file safetensors without loading."""
    weights = directory / "model.safetensors"
    if not (directory / "config.json").is_file() or not weights.is_file():
        return False
    try:
        with weights.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size < 64 * 1024 * 1024:
                return False
            header = json.loads(stream.read(header_size))
        ends = [value["data_offsets"][1] for key, value in header.items() if key != "__metadata__"]
        return bool(ends) and 8 + header_size + max(ends) == weights.stat().st_size
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False


def find_model(repo_id: str) -> Path | None:
    name = repo_id.split("/")[-1]
    candidates = [MODEL_ROOT / name, ROOT / "weights" / name]
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        cached = None
    else:
        cached = try_to_load_from_cache(repo_id, "model.safetensors")
    if isinstance(cached, str):
        candidates.append(Path(cached).parent)
    return next((path for path in candidates if complete_model(path)), None)


def find_demo(filename: str) -> Path | None:
    roots = [
        ROOT / "datasets" / "raw",
        Path.home() / "ytz" / "libero90_ablation_raw" / "libero_90",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(filename):
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                if stream.read(8) == b"\x89HDF\r\n\x1a\n":
                    return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Local inspection only (default)")
    mode.add_argument("--models", action="store_true", help="Fetch missing base policy and small VLM metadata")
    mode.add_argument("--data", action="store_true", help="Fetch only the four missing book demonstrations")
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    record_path = STATE / "resources.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    report = {"mode": "models" if args.models else "data" if args.data else "plan", "models": {}, "demonstrations": {}}
    for repo_id in (BASE_ID, REFERENCE_ID):
        existing = find_model(repo_id)
        report["models"][repo_id] = {"status": "reuse" if existing else "missing", "path": str(existing) if existing else None}
    for filename in FILENAMES:
        existing = find_demo(filename)
        report["demonstrations"][filename] = {"status": "reuse" if existing else "missing", "path": str(existing) if existing else None}

    if args.models or args.data:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        if args.models:
            if report["models"][BASE_ID]["status"] == "missing":
                revision = record.get(BASE_ID, {}).get("revision") or api.model_info(BASE_ID).sha
                destination = MODEL_ROOT / "smolvla_base"
                snapshot_download(repo_id=BASE_ID, revision=revision, local_dir=destination, force_download=False)
                if not complete_model(destination):
                    raise RuntimeError("The base policy is incomplete; do not train from it")
                record[BASE_ID] = {"revision": revision, "path": str(destination)}
                report["models"][BASE_ID] = {"status": "downloaded", "path": str(destination)}
            # A complete policy already contains VLM weights. Fetch its config and
            # tokenizer/processor only; load the full policy with load_vlm_weights=False.
            revision = record.get(VLM_ID, {}).get("revision") or api.model_info(VLM_ID).sha
            metadata_path = snapshot_download(
                repo_id=VLM_ID,
                revision=revision,
                allow_patterns=["*.json", "*.txt", "tokenizer.model", "*.jinja", "*.tiktoken"],
                force_download=False,
            )
            record[VLM_ID] = {"revision": revision, "metadata_path": metadata_path, "weights_downloaded": False}
            report["vlm_metadata"] = record[VLM_ID]
            report["reference_policy"] = "Existing smolvla_libero is reused; no automatic second download"

        if args.data:
            missing = [name for name, item in report["demonstrations"].items() if item["status"] == "missing"]
            if missing:
                revision = record.get(DATA_ID, {}).get("revision") or api.dataset_info(DATA_ID).sha
                repository_files = api.list_repo_files(DATA_ID, repo_type="dataset", revision=revision)
                selected = []
                for filename in missing:
                    matches = [path for path in repository_files if Path(path).name == filename]
                    preferred = [path for path in matches if path.startswith("libero_100/")]
                    matches = preferred or matches
                    if len(matches) != 1:
                        raise RuntimeError(f"Expected one exact remote file for {filename}; found {matches}. No bulk fallback was started.")
                    selected.append(matches[0])
                destination = ROOT / "datasets" / "raw"
                snapshot_download(repo_id=DATA_ID, repo_type="dataset", revision=revision, local_dir=destination, allow_patterns=selected, force_download=False)
                record[DATA_ID] = {"revision": revision, "directory": str(destination), "selected_files": selected}
                for filename in missing:
                    downloaded = find_demo(filename)
                    if downloaded is None:
                        raise RuntimeError(f"Missing or invalid HDF5 after download: {filename}")
                    report["demonstrations"][filename] = {"status": "downloaded", "path": str(downloaded)}

    record["last_local_inventory"] = report
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
