"""Generate a third test set without looking at V interventions or route outcomes."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
V3 = RUN.parent / "k25v3_20261120_01"
sys.path.insert(0, str(V3))
from v_attribution_v3 import V2, read, write
from path_factorization import K19

RESET_SEED = 991201
NOISE_SEED = 13960041


def prepare(protocol):
    protocol.RESET_BASE, protocol.NOISE_BASE = RESET_SEED, NOISE_SEED
    protocol.generate()
    raw = read(RUN / "candidates.json")
    refs = np.asarray([r["state"][1:] for previous in (K19, V2)
                       for r in read(previous / "candidates.json")["records"]], dtype=np.float64)
    records = []
    for row in raw["records"]:
        distance = float(np.linalg.norm(refs - row["state"][1:], axis=1).min())
        if distance > 1e-6:
            records.append({**row, "init": row["init"] + 400, "min_distance_to_previous_k25": distance})
    assert len(records) >= protocol.NEEDED
    write(RUN / "candidates.json", {**raw, "records": records, "reset_seed_base": RESET_SEED,
                                    "noise_seed_base": NOISE_SEED, "k25_resets_compared": len(refs)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "screen", "lock"))
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("k19_handoff", K19 / "holdout.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    assert Path(protocol.__file__).resolve().parent == K19
    protocol.RUN = RUN
    if args.mode == "generate":
        prepare(protocol)
    elif args.mode == "screen":
        protocol.screen(args.lane, args.lanes)
    else:
        protocol.lock(args.lanes)
