"""Build a hybrid checkpoint: one functional group of trainable parameters taken from a donor, the rest from a base.

Restoration (base = adapted endpoint, donor = parent) asks which group brings old goal selection back; damage
(base = parent, donor = adapted endpoint) asks which group alone induces substitution. Frozen VLM tensors must be
identical in both checkpoints. Processors and config come from the base; nothing is trained.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

EXPERT = r"model\.vlm_with_expert\.lm_expert\."
LAYER = EXPERT + r"layers\.(\d+)\."
GROUPS = {
    # expert projections that read the frozen VLM keys/values (cross-attention layers are the odd ones)
    "read_kv": lambda k, i: re.match(LAYER + r"self_attn\.[kv]_proj\.", k) and i % 2 == 1,
    "queries": lambda k, i: re.match(LAYER + r"self_attn\.q_proj\.", k),
    "self_kv": lambda k, i: re.match(LAYER + r"self_attn\.[kv]_proj\.", k) and i % 2 == 0,
    "attn_out": lambda k, i: re.match(LAYER + r"self_attn\.o_proj\.", k),
    "mlp": lambda k, i: re.match(LAYER + r"mlp\.", k),
    "norms": lambda k, i: re.match(LAYER + r"(input|post_attention)_layernorm\.", k) or re.match(EXPERT + r"norm\.", k),
    "state_proj": lambda k, i: k.startswith("model.state_proj."),
    "action_io": lambda k, i: re.match(r"model\.(action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)\.", k),
    "early_expert": lambda k, i: i is not None and i < 8,
    "late_expert": lambda k, i: i is not None and i >= 8,
}


def layer_index(key):
    match = re.match(LAYER, key)
    return int(match.group(1)) if match else None


def group_keys(keys, group):
    test = GROUPS[group]
    return sorted(k for k in keys if test(k, layer_index(k)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(base, donor, group, output, trainable):
    from safetensors import safe_open
    from safetensors.torch import save_file
    if output.exists():
        raise FileExistsError(output)
    with safe_open(base / "model.safetensors", "pt") as b, safe_open(donor / "model.safetensors", "pt") as d:
        if set(b.keys()) != set(d.keys()):
            raise ValueError("Base and donor tensors differ in names")
        metadata = b.metadata()
        swapped = group_keys(b.keys(), group)
        if not swapped or not set(swapped) <= trainable:
            raise ValueError(f"Group {group} is empty or touches frozen tensors")
        tensors, changed = {}, 0
        for key in b.keys():
            tensor = (d if key in swapped else b).get_tensor(key)
            if key not in trainable and not b.get_tensor(key).equal(d.get_tensor(key)):
                raise ValueError(f"Frozen tensor differs between base and donor: {key}")
            if key in swapped:
                changed += int(not tensor.equal(b.get_tensor(key)))
            tensors[key] = tensor
    output.mkdir(parents=True)
    for item in base.iterdir():
        if item.name in ("model.safetensors", "training_state.pt", "CHECKPOINT_COMPLETE.json"):
            continue
        (shutil.copytree if item.is_dir() else shutil.copy2)(item, output / item.name)
    save_file(tensors, output / "model.safetensors", metadata=metadata)
    manifest = {"base": str(base), "base_model_sha256": sha256(base / "model.safetensors"),
                "donor": str(donor), "donor_model_sha256": sha256(donor / "model.safetensors"),
                "group": group, "swapped_tensors": swapped, "swapped_tensor_count": len(swapped),
                "swapped_tensors_that_differ": changed,
                "model_sha256": sha256(output / "model.safetensors"),
                "scope": "evaluation-only hybrid; processors/config from base"}
    manifest["swapped_parameters"] = sum(tensors[k].numel() for k in swapped)
    (output / "SWAP_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("group", "swapped_tensor_count", "swapped_tensors_that_differ", "swapped_parameters")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--donor", type=Path, required=True)
    p.add_argument("--group", choices=sorted(GROUPS), required=True)
    p.add_argument("--trainable", type=Path, required=True, help="trainable_parameters.json of the adaptation run")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    names = set(json.loads(a.trainable.read_text())["names"])
    build(a.base, a.donor, a.group, a.output, names)
