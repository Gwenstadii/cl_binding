"""Map Front demonstration action positions to phases, retaining unknowns."""
import json
from pathlib import Path

from stage1_smoke import ROOT, sources

import h5py
import numpy as np
import pyarrow.parquet as pq
import robosuite
from libero.libero import get_libero_path
from stage1_protocol import remap_demo_xml
from stage1_r0_eval import events, make_env, task_resources

RUN = Path(__file__).resolve().parent
PHASES = ("grasp_lift", "transport", "approach_place", "unknown")


def first_sustained(mask, count=5):
    indices = np.flatnonzero(np.convolve(mask, np.ones(count, dtype=int), mode="valid") == count)
    return int(indices[0]) if len(indices) else None


def main():
    config = json.loads((RUN / "config.json").read_text())
    dataset = ROOT / config["data"]["validation_output"]
    conversion = json.loads((dataset / "r0_conversion.json").read_text())
    offset = conversion["spec"]["action_offset"]
    assert offset == 1 and conversion["spec"]["goals"] == ["front"]
    converted = sorted((r for p in (dataset / "data").rglob("*.parquet")
                        for r in pq.read_table(p, columns=["index", "episode_index", "frame_index", "action"]).to_pylist()),
                       key=lambda r: r["index"])
    _, paths, predicates, _ = task_resources()
    env = make_env(paths["front"])
    labels = []
    demos = []
    try:
        with h5py.File(sources(("front",))["front"], "r") as raw:
            for episode, rec in enumerate(conversion["episodes"]):
                demo = raw["data"][rec["demo"]]
                actions = demo["actions"][:]
                begin = len(labels)
                rows = converted[begin:begin + rec["frames"]]
                assert all(r["index"] == begin + j and r["episode_index"] == episode and r["frame_index"] == j for j, r in enumerate(rows))
                np.testing.assert_array_equal(np.asarray([r["action"] for r in rows], dtype=np.float32), actions[offset:].astype(np.float32))
                xml = remap_demo_xml(demo.attrs["model_file"], get_libero_path("assets"), Path(robosuite.__file__).parent)
                env.seed(config["seed"])
                env.reset()
                env.reset_from_xml_string(xml)
                env.sim.reset()
                book_id = env.env.obj_body_id["black_book_1"]
                book, success = [], []
                for state in demo["states"][:]:
                    env.set_state(state)
                    env.sim.forward()
                    book.append(env.sim.data.body_xpos[book_id].copy())
                    success.append(events(env, {"front": predicates["front"]})["front"])
                book = np.asarray(book)
                eef = demo["obs/ee_states"][:, :3]
                held = ((book[:, 2] - book[0, 2]) > 0.03) & (np.linalg.norm(book - eef, axis=1) < 0.15) & (actions[:, 6] > 0)
                lift = first_sustained(held)
                good = np.flatnonzero(success)
                transport_start = approach_start = None
                if lift is not None and len(good) and lift < good[0]:
                    direction = book[good[0], :2] - book[lift, :2]
                    distance = np.linalg.norm(direction)
                    if distance > 0.03:
                        progress = (book[:, :2] - book[lift, :2]) @ (direction / distance)
                        first = np.flatnonzero((progress[lift:good[0]] >= 0.015) & held[lift:good[0]])
                        last = np.flatnonzero(progress[lift:good[0] + 1] >= 0.70 * distance)
                        if len(first) and len(last) and last[0] - first[0] >= 5:
                            transport_start, approach_start = lift + int(first[0]), lift + int(last[0])
                for frame in range(offset, len(actions)):
                    labels.append(3 if transport_start is None else 0 if frame < transport_start else 1 if frame < approach_start else 2)
                demos.append({"demo": rec["demo"], "lift": lift, "transport_start": transport_start, "approach_start": approach_start,
                              "first_success": int(good[0]) if len(good) else None})
    finally:
        env.close()
    assert len(labels) == len(converted)
    report = {"labels": labels, "names": PHASES, "demos": demos, "split_sha256": conversion["spec"]["split_sha256"],
              "action_offset": offset, "counts": dict(zip(PHASES, np.bincount(labels, minlength=4).tolist())),
              "definition": "lift>3cm, EEF gap<15cm, close command and 5 consecutive frames; transport starts at 1.5cm projected advance, approach starts at 70% advance toward first successful Front position; missing boundaries unknown"}
    (RUN / "front_validation_phases.json").write_text(json.dumps(report, indent=2))
    print(report["counts"], flush=True)


if __name__ == "__main__":
    main()

