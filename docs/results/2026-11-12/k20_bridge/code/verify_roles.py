"""Verify that the second loaded role is exactly the original K20 policy."""
import torch

from eval_bridge import INSTR, K19, Lab5, RUN, handoff, load_arm, load_patch, read, snapshots, write


def check():
    lab = Lab5()
    result = {}
    try:
        for suite, seed in (("dev", 20261021), ("confirm", 20261121)):
            record = snapshots()[20] if suite == "dev" else read(K19 / "holdout.json")["selected"][0]
            load_patch(lab, seed)
            load_arm(lab, seed, "route", 1000)
            a, b = (lab.rt.models[role].state_dict() for role in "FR")
            assert a.keys() == b.keys()
            for name in a:
                assert torch.equal(a[name], b[name]), name
            obs, rows = handoff(lab, record, suite)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for q in "AB":
                    inputs = (INSTR[q], obs, record["seed"], len(rows) // 10)
                    assert torch.equal(lab.native("F", *inputs), lab.native("R", *inputs))
            result[suite] = {"equal_tensors": len(a), "equal_native_chunks": 2}
        write(RUN / "role_verification.json", result)
        print(result, flush=True)
    finally:
        lab.ctx.env.close()


if __name__ == "__main__":
    torch.set_num_threads(2)
    check()

