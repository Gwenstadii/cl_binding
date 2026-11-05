"""Cross fixed hidden states with both action readouts on 40-step snapshots."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from k8_transplant_attribution import SERVER


ROLES = ("R", "F")
FLOW_INDICES = (0, 4, 8)


def digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def probe(rt, role, prefix, x, t):
    model = rt.models[role].model
    captured = []

    def save_hidden(_module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = model.action_out_proj.register_forward_pre_hook(save_hidden)
    try:
        native = rt.query(role, prefix, x, t)
    finally:
        handle.remove()
    assert len(captured) == 1
    h = captured[0]
    projected = {m: rt.models[m].model.action_out_proj(h) for m in ROLES}
    assert rt.torch.equal(native, projected[role])
    return h, projected


def main(output):
    sys.path[:0] = [str(SERVER / directory / "code") for directory in (
        "k1_screen_20261101_01", "k4_text_attention_20261102_01")]
    import k1_main as km
    from k4_text_attention import INSTR, run_state

    pair = json.loads((SERVER / "k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_bytes())["families"]["scene2"]
    ctx = km.Ctx("scene2")
    rt = km.make_runtime(ctx, pair, list(ROLES))
    records, hidden, drafts = [], {}, {}
    with ctx.torch.no_grad(), ctx.torch.autocast("cuda", dtype=ctx.torch.bfloat16):
        for init in range(10):
            # R0 produces the same Left/40 state for all four cells.
            obs, _, _, seed = run_state(ctx, rt, init, 40)
            prefixes = {(role, goal): rt.prefix(role, obs, INSTR[goal])
                        for role in ROLES for goal in ("A", "B", "N")}
            anchor_goal = "AB"[init % 2]
            for anchor_role in ROLES:
                anchors, calls = {}, []

                def velocity(x, t):
                    i = len(calls)
                    calls.append(i)
                    if i in FLOW_INDICES:
                        anchors[i] = x.clone(), t.clone()
                    return rt.query(anchor_role, prefixes[anchor_role, anchor_goal], x, t)

                rt.integrate(velocity, rt.noise(seed, 4), 10)
                assert tuple(anchors) == FLOW_INDICES and len(calls) == 10
                for flow, (x, t) in anchors.items():
                    anchor = f"{init}-{anchor_role}-{flow}"
                    drafts[anchor] = x.float().cpu().numpy()
                    row = {
                        "init": init, "seed": seed, "anchor_role": anchor_role, "anchor_goal": anchor_goal,
                        "flow": flow, "t": t.item(), "draft_sha256": digest(x), "values": {},
                    }
                    for role in ROLES:
                        for goal in ("A", "B", "N"):
                            h, projected = probe(rt, role, prefixes[role, goal], x, t)
                            key = f"{role}{goal}"
                            hidden[anchor + "-" + key] = h[:, :10].float().cpu().numpy()
                            row["values"][key] = {
                                "hidden_sha256": digest(h),
                                "velocity": {m: v[0, :10, :7].float().cpu().tolist()
                                             for m, v in projected.items()},
                            }
                    records.append(row)
            print(f"Completed snapshot {init}: {len(records)} anchors", flush=True)
    assert len(records) == 60
    rt.np.savez_compressed(output / "hidden_and_drafts.npz", **hidden, **drafts)
    (output / "fields.json").write_text(json.dumps({
        "model_sha256": rt.sha, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_readout_identity_checks": len(records) * 6,
        "snapshot_recipe": "R0 Left first 40 K4-compatible control steps",
        "anchors": records,
    }, allow_nan=False) + "\n", encoding="utf-8")
    print("Complete: 60 matched anchors", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
