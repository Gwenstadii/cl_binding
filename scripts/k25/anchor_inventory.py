"""Read-only provenance and norm audit of the eight exact historical V tensors."""

from datetime import datetime, timezone
import math
from pathlib import Path

import torch

import anchor_check as core


def inventory():
    run = Path(__file__).resolve().parent
    protocol = core.base.read(run/"protocol.json")
    result = dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=core.digest(Path(__file__)),lines={},
                  interpretation="Parameter norms are descriptive; they are not causal responsibility fractions or recovery qualifications")
    for line in core.spec.LINES:
        weights = {s:core.base.selected_weights(Path(protocol["checkpoints"][f"{line}:{s}"]["path"])) for s in (0,30,1000)}
        _,selected = core.base.sr.parameters_for("cross_v",weights[1000])
        assert set(selected) == {f"model.vlm_with_expert.lm_expert.layers.{i}.self_attn.v_proj.weight" for i in range(1,16,2)}
        d0 = {n:weights[0][n].double()-weights[1000][n].double() for n in sorted(selected)}
        d30 = {n:weights[30][n].double()-weights[1000][n].double() for n in sorted(selected)}
        early = {n:weights[0][n].double()-weights[30][n].double() for n in sorted(selected)}
        def norm(v):
            return math.sqrt(sum(float(x.square().sum()) for x in v.values()))
        n0,n30,ne = norm(d0),norm(d30),norm(early)
        cosine = sum(float((d0[n]*d30[n]).sum()) for n in selected)/(n0*n30)
        result["lines"][line] = dict(delta_sha256={"0":core.tensors_hash(d0),"30":core.tensors_hash(d30)},
            norm_0_to_1000=n0,norm_30_to_1000=n30,norm_0_to_30=ne,
            cosine_delta0_delta30=cosine,tensors={n:dict(shape=list(weights[0][n].shape),dtype=str(weights[0][n].dtype),
                norm_0_to_30=float(torch.linalg.vector_norm(early[n])),norm_30_to_1000=float(torch.linalg.vector_norm(d30[n])),
                norm_0_to_1000=float(torch.linalg.vector_norm(d0[n]))) for n in sorted(selected)})
    target = run/"weights_inventory.json"
    assert not target.exists()
    core.base.write(target,result)
    print({line:{k:v for k,v in r.items() if k != "tensors"} for line,r in result["lines"].items()},flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    inventory()
