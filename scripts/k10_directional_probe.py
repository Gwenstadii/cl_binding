"""Select CA/SA/MLP sites on even snapshots and test directional edits on odd snapshots."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import sys

from k10_fixed_input_projection import ROLES, digest
from k8_transplant_attribution import SERVER


def modules(policy):
    layers = policy.model.vlm_with_expert.lm_expert.layers
    assert len(layers) == 16
    result = {}
    for i, layer in enumerate(layers):
        result[f"{'CA' if i % 2 else 'SA'}-{i}"] = layer.self_attn.o_proj
        result[f"MLP-{i}"] = layer.mlp
    return result


def trace(rt, role, pre, x, t, sites):
    values = {}

    def record(name):
        def hook(_module, _inputs, out):
            assert name not in values
            values[name] = out.detach().clone()
        return hook

    handles = [module.register_forward_hook(record(name)) for name, module in sites.items()]
    try:
        velocity = rt.query(role, pre, x, t)
    finally:
        for handle in handles:
            handle.remove()
    assert values.keys() == sites.keys()
    return velocity, values


def patch(rt, pre, x, t, site, delta):
    calls = []

    def hook(_module, _inputs, out):
        assert out.shape == delta.shape
        calls.append(1)
        return (out.float() + delta).to(dtype=out.dtype)

    handle = site.register_forward_hook(hook)
    try:
        value = rt.query("F", pre, x, t)
    finally:
        handle.remove()
    assert len(calls) == 1
    return value


def main(run):
    sys.path[:0] = [str(SERVER / name / "code") for name in (
        "k1_screen_20261101_01", "k4_text_attention_20261102_01")]
    import k1_main as km
    from k4_text_attention import INSTR, run_state

    fields = json.loads((run / "fields.json").read_bytes())
    pair = json.loads((SERVER / "k3_e2_20261102_01/prereg/k3_e2_prereg.json").read_bytes())["families"]["scene2"]
    ctx = km.Ctx("scene2")
    rt = km.make_runtime(ctx, pair, list(ROLES))
    torch = ctx.torch
    assert fields["model_sha256"] == rt.sha
    sites = {role: modules(rt.models[role]) for role in ROLES}
    anchors = defaultdict(list)
    for row in fields["anchors"]:
        anchors[row["init"]].append(row)
    scores, tests, checks = defaultdict(list), [], 0
    selected = {}
    with ctx.np.load(run / "hidden_and_drafts.npz") as arrays, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for init in (*range(0, 10, 2), *range(1, 10, 2)):
            if init == 1:
                medians = {site: median(items) for site, items in scores.items()}
                selected = {kind: max((k for k in medians if k.startswith(kind + "-")),
                                      key=medians.get) for kind in ("CA", "SA", "MLP")}
                print("Selected", selected, flush=True)
            obs, _, _, seed = run_state(ctx, rt, init, 40)
            prefixes = {(m, g): rt.prefix(m, obs, INSTR[g]) for m in ROLES for g in ("A", "B")}
            for row in anchors[init]:
                assert seed == row["seed"]
                anchor = f"{init}-{row['anchor_role']}-{row['flow']}"
                x = torch.as_tensor(arrays[anchor], device="cuda")
                t = torch.tensor([row["t"]], device="cuda")
                assert digest(x) == row["draft_sha256"]
                values, messages = {}, {}
                for key, pre in prefixes.items():
                    m, g = key
                    values[key], messages[key] = trace(rt, m, pre, x, t, sites[m])
                    original = torch.tensor(row["values"][m + g]["velocity"][m], device="cuda")
                    assert torch.equal(values[key][0, :10, :7].float(), original)
                    checks += 1
                d = {
                    m: {site: (messages[m, "A"][site].float() - messages[m, "B"][site].float()) / 2
                        for site in sites[m]} for m in ROLES
                }
                if init % 2 == 0:
                    for site in sites["R"]:
                        old = d["R"][site][:, :10].reshape(-1)
                        new = d["F"][site][:, :10].reshape(-1)
                        n = old.norm()
                        # Rank sites by loss along their own R0 contrast, within module type.
                        scores[site].append(max(0.0, (n - (new @ old) / n.clamp_min(1e-9)).item()))
                    continue
                old = (values["R", "A"].float() - values["R", "B"].float())[0, :10, :3].reshape(-1) / 2
                denominator = old @ old
                assert denominator.item() > 1e-9
                for site in selected.values():
                    layer = sites["F"][site]
                    correction = d["R"][site] - d["F"][site]
                    reverse = -d["R"][site] - d["F"][site]
                    rng = torch.Generator(device="cpu").manual_seed(seed + row["flow"])
                    random = torch.randn(correction.shape, generator=rng).cuda()
                    random *= correction.norm() / random.norm()
                    for condition, delta in (
                        ("native", None), ("correct", correction), ("reversed", reverse), ("random", random),
                    ):
                        pair = [values["F", g] if delta is None else
                                patch(rt, prefixes["F", g], x, t, layer, sign * delta)
                                for g, sign in (("A", 1), ("B", -1))]
                        a, b = (v[0, :10, :3].float().reshape(-1) for v in pair)
                        diff = (a - b) / 2
                        mean = (a + b) / 2
                        base = (values["F", "A"].float() + values["F", "B"].float())[0, :10, :3].reshape(-1) / 2
                        tests.append({
                            "init": init, "anchor": anchor, "site": site, "condition": condition,
                            "projection": ((diff @ old) / denominator).item(),
                            "mean_delta_over_r0_d": ((mean - base).norm() / old.norm()).item(),
                            "injection_norm": delta.norm().item() if delta is not None else 0,
                        })
            print(f"Finished snapshot {init}", flush=True)
    assert checks == 240 and len(tests) == 360
    summary = {}
    for site in selected.values():
        grouped = {(r["anchor"], r["condition"]): r for r in tests if r["site"] == site}
        summary[site] = {
            cond: {
                "median_projection": median(r["projection"] for (a, c), r in grouped.items() if c == cond),
                "median_gain": median(r["projection"] - grouped[a, "native"]["projection"]
                                      for (a, c), r in grouped.items() if c == cond),
                "median_mean_delta_over_r0_d": median(r["mean_delta_over_r0_d"]
                                                       for (a, c), r in grouped.items() if c == cond),
            } for cond in ("native", "correct", "reversed", "random")
        }
    (run / "directional_probe.json").write_text(json.dumps({
        "selected": selected, "selection_medians": medians, "native_exact_matches": checks,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary, "tests": tests,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.run)
