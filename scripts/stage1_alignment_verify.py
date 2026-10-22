"""Verify every converted state/action pair against raw old-only trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from stage1_protocol import observation_action_indices
from stage1_smoke import ROOT, dump


def main():
    import h5py
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    audit = json.loads(args.audit.read_text())
    config = audit["config"]
    assert config["data"]["action_alignment"] == "post_action_obs_to_next_action"
    result = {"status": "running", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "split_sha256": audit["split_sha256"], "splits": {}}
    for split in ("train", "validation"):
        folder = ROOT / config["data"][f"{split}_output"]
        marker_path = folder / "r0_conversion.json"
        marker = json.loads(marker_path.read_text())
        assert marker["status"] == "complete"
        assert marker["spec"]["split"] == split
        assert marker["spec"]["split_sha256"] == audit["split_sha256"]
        assert marker["spec"]["action_alignment"] == config["data"]["action_alignment"]
        assert marker["spec"]["goals"] == config["old_goals"]
        expected_order = [(g, d) for g in config["old_goals"] for d in audit["splits"][g][split]]
        assert [(e["goal"], e["demo"]) for e in marker["episodes"]] == expected_order
        files = sorted((folder / "data").rglob("*.parquet"))
        assert files
        table = pa.concat_tables([pq.read_table(p, columns=["index", "episode_index", "action", "observation.state"]) for p in files]).sort_by("index")
        assert table["index"].to_pylist() == list(range(len(table)))
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        episode_ids = np.asarray(table["episode_index"].to_pylist())
        offset = 0
        for episode_id, episode in enumerate(marker["episodes"]):
            with h5py.File(audit["files"][episode["goal"]]["path"]) as h:
                demo = h["data"][episode["demo"]]
                raw_action = demo["actions"][:].astype(np.float32)
                raw_state = np.concatenate((demo["obs/ee_states"][:], demo["obs/gripper_states"][:]), axis=1).astype(np.float32)
            obs_idx, action_idx = observation_action_indices(len(raw_action), config["data"]["action_alignment"])
            end = offset + len(obs_idx)
            assert episode["frames"] == len(obs_idx) == len(raw_action)-1
            assert np.array_equal(actions[offset:end], raw_action[list(action_idx)])
            assert np.array_equal(state[offset:end], raw_state[list(obs_idx)])
            assert np.all(episode_ids[offset:end] == episode_id)
            offset = end
        assert offset == len(table)
        stats = json.loads((folder / "meta/stats.json").read_text())
        # Training uses this directory's own stats; validation stats are never
        # passed to the trainer. Check means against the actual converted rows.
        for key, values in (("action", actions), ("observation.state", state)):
            assert np.allclose(stats[key]["mean"], values.astype(np.float64).mean(axis=0), atol=1e-5, rtol=1e-5)
        result["splits"][split] = {"trajectories": len(marker["episodes"]), "frames": len(table),
                                   "all_state_action_pairs_exact": True, "stats_means_checked": True,
                                   "conversion_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest()}
    result.update(status="pass", normalization_source=config["data"]["train_output"], n1_in_training_or_stats=False)
    dump(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
