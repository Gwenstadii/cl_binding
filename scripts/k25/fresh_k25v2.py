"""Lock a second set of physical grasp handoffs under the original K25 rules."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / "k25_paths_20261119_01"))
from path_factorization import K19, read, write

spec = importlib.util.spec_from_file_location("k19_handoff_protocol", K19 / "holdout.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

RESET_SEED = 990821
NOISE_SEED = 11940047


def generate():
    assert Path(protocol.__file__).resolve().parent == K19
    protocol.RESET_BASE, protocol.NOISE_BASE = RESET_SEED, NOISE_SEED
    protocol.generate()
    previous = read(K19 / "candidates.json")["records"]
    refs = np.asarray([r["state"][1:] for r in previous], dtype=np.float64)
    raw = read(RUN / "candidates.json")
    fresh = []
    for r in raw["records"]:
        distance = float(np.linalg.norm(refs - r["state"][1:], axis=1).min())
        if distance > 1e-6:
            fresh.append({**r, "init": r["init"] + 200, "min_distance_to_preceding_k25": distance})
    assert len(fresh) >= protocol.NEEDED
    write(RUN / "candidates.json", {**raw, "records": fresh, "reset_seed_base": RESET_SEED,
                                    "noise_seed_base": NOISE_SEED, "preceding_k25_resets_compared": len(refs)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "screen", "lock"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    a = parser.parse_args()
    assert Path(protocol.__file__).resolve().parent == K19
    protocol.RUN = RUN
    if a.mode == "generate":
        generate()
    elif a.mode == "screen":
        protocol.screen(a.lane, a.lanes)
    else:
        protocol.lock(a.lanes)
