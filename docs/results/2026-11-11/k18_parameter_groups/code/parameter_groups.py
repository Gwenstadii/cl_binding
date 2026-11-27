"""K18-C. Intervene on checkpoint parameter changes before native inference."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import time

import torch
from safetensors import safe_open

RUN = Path(__file__).resolve().parent
K18 = RUN.parent / "k18_training_cause_20261111_01"
MEASUREMENT_SHA = "4bc20e48c64e3856924b03d20aff2425deff41f2808dbd90f24ff3ce466bdb2a"
assert hashlib.sha256((K18 / "measurement.py").read_bytes()).hexdigest() == MEASUREMENT_SHA
sys.path.insert(0, str(K18))
from measurement import BLOCKS, INSTR, Lab5, ORIGINAL, checkpoint, load, make_bank, rebuild, snapshots

GROUPS = ("state", "action_time", "attention", "mlp_norm", "output")
INITS = tuple(range(20, 40))
SIGN_SEED = 20261111


def group_of(name):
    if name.startswith("model.state_proj."):
        return "state"
    if name.startswith(("model.action_in_proj.", "model.action_time_mlp_in.", "model.action_time_mlp_out.")):
        return "action_time"
    if name.startswith("model.action_out_proj."):
        return "output"
    if name.startswith("model.vlm_with_expert.lm_expert."):
        if ".self_attn." in name:
            return "attention"
        if ".mlp." in name or "norm." in name:
            return "mlp_norm"
    raise ValueError(name)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Window:
    def __init__(self, lab, start, end, audit=False):
        self.label = f"{start:06d}_{end:06d}"
        self.path_a = checkpoint(lab, start)
        self.path_b = checkpoint(lab, end)
        self.model = lab.rt.models["F"]
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        official = read(ORIGINAL / "trainable_parameters.json")
        assert list(self.params) == official["names"]
        assert sum(p.numel() for p in self.params.values()) == official["count"] == 99880992
        self.groups = {n: group_of(n) for n in self.params}
        assert set(self.groups.values()) == set(GROUPS) and len(self.groups) == 155
        self.a, self.b = {}, {}
        frozen = []
        with (
            safe_open(self.path_a / "model.safetensors", framework="pt") as a,
            safe_open(self.path_b / "model.safetensors", framework="pt") as b,
        ):
            assert a.keys() == b.keys()
            for n in a.keys():
                if n in self.params:
                    self.a[n], self.b[n] = a.get_tensor(n), b.get_tensor(n)
                    assert self.a[n].dtype == self.b[n].dtype == self.params[n].dtype, (
                        n, self.a[n].dtype, self.b[n].dtype, self.params[n].dtype
                    )
                elif audit:
                    assert torch.equal(a.get_tensor(n), b.get_tensor(n)), n
                    frozen.append(n)
        assert set(self.a) == set(self.b) == set(self.params)
        load(lab, self.path_a)
        if audit:
            self.audit = {"groups": {k: [] for k in GROUPS}, "unchanged_other_tensors": len(frozen)}
            total = 0.
            for n in self.params:
                delta = self.b[n].double() - self.a[n].double()
                square = float(delta.square().sum())
                total += square
                self.audit["groups"][self.groups[n]].append(
                    {"name": n, "size": delta.numel(), "dtype": str(self.a[n].dtype), "delta_l2": square**.5}
                )
            self.audit["total_delta_l2"] = total**.5
            self.audit["endpoints"] = {str(s): hashlib.file_digest((path/"model.safetensors").open("rb"), "sha256").hexdigest()
                                       for s, path in ((start, self.path_a), (end, self.path_b))}
            for s in (start, end):
                self.apply("early" if s == start else "late")
                with safe_open(checkpoint(lab, s) / "model.safetensors", framework="pt") as ref:
                    assert set(self.model.state_dict()) == set(ref.keys())
                    assert all(torch.equal(p.detach().cpu(), ref.get_tensor(n)) for n, p in self.model.state_dict().items())
            self.audit["both_endpoints_all_tensors_exact"] = True

    @torch.no_grad()
    def apply(self, condition, verify_random=False):
        mode, _, group = condition.partition(":")
        assert (mode in ("early", "late") and not group) or (
            mode in ("add", "undo", "random_add", "random_replace") and group in GROUPS
        ), condition
        rng = torch.Generator(device="cpu").manual_seed(SIGN_SEED)
        rel_errors = []
        inner_product = total_real_sq = total_random_sq = locked_sq = 0.
        for n, p in self.params.items():
            selected = self.groups[n] == group
            a, b = self.a[n], self.b[n]
            if mode.startswith("random") and selected:
                origin = a.double()
                delta = b.double() - origin
                opposite = (origin-delta).to(a.dtype)
                can_flip = (opposite.double()-origin) == -delta
                flip = torch.randint(0, 2, a.shape, generator=rng, dtype=torch.bool) & can_flip
                value = torch.where(flip, opposite, b)
                if verify_random:
                    actual = value.double() - origin
                    real_sq, random_sq = float(delta.square().sum()), float(actual.square().sum())
                    rel_errors.append(abs(random_sq**.5 - real_sq**.5) / max(real_sq**.5, 1e-12))
                    inner_product += float((delta * actual).sum())
                    locked_sq += float((delta.square() * ~can_flip).sum())
                    total_real_sq += real_sq
                    total_random_sq += random_sq
            else:
                use_late = mode == "late" or (mode == "add" and selected) or (
                    mode in ("undo", "random_replace") and not selected
                )
                value = b if use_late else a
            p.copy_(value)
        if rel_errors:
            error = max(rel_errors)
            assert error < 1e-4, (condition, error)
            return {"max_tensor_norm_relative_error": error,
                    "cosine_with_true_group_update": inner_product / (total_real_sq * total_random_sq)**.5,
                    "rounding_locked_update_energy_fraction": locked_sq / total_real_sq}
        return None


def fields(lab, bank):
    result = []
    current = None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for init, route, denoise, obs, x, t, _, _ in bank:
            if init != current:
                prefix = {g: lab.rt.prefix("F", obs, INSTR[g]) for g in "AB"}
                current = init
            result.append(torch.stack([lab.rt.query("F", prefix[g], x, t)[0].float() for g in "AB"]))
    return torch.stack(result)


def dot(a, b):
    return (a * b).flatten(1).sum(1)


def summarize_fields(bank, values, early, late):
    old = torch.stack([torch.stack([a[0], b[0]]) for *_, a, b in bank])
    d = (values[:, 0] - values[:, 1]) * .5
    old_d = (old[:, 0] - old[:, 1]) * .5
    mu = values.mean(1)
    base = early.mean(1)
    drift = late.mean(1) - base
    rows = [{"init": x[0], "route": x[1], "denoise_index": x[2]} for x in bank]
    for n in (10, 50):
        ref = old_d[:, :n, :7]
        cur = d[:, :n, :7]
        denominator = dot(ref, ref).clamp_min(1e-12)
        metrics = {"gain": dot(ref, cur) / denominator,
                   "norm_ratio": (dot(cur, cur) / denominator).sqrt(),
                   "cosine": dot(ref, cur) / (dot(cur, cur)*denominator).sqrt().clamp_min(1e-12)}
        for k, block in BLOCKS.items():
            change = (mu-base)[:, :n, block]
            endpoint = drift[:, :n, block]
            metrics[f"mu_{k}_rms"] = change.square().flatten(1).mean(1).sqrt()
            metrics[f"mu_{k}_projection"] = dot(change, endpoint) / dot(endpoint, endpoint).clamp_min(1e-12)
            if k == "P":
                continue
            channel_ref, channel = old_d[:, :n, block], d[:, :n, block]
            metrics[f"gain_{k}"] = dot(channel_ref, channel) / dot(channel_ref, channel_ref).clamp_min(1e-12)
        columns = {k: v.tolist() for k, v in metrics.items()}
        for i, row in enumerate(rows):
            row[str(n)] = {k: v[i] for k, v in columns.items()}
    return rows


def scenario_metrics(rows, key, n="10"):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["init"]].append(row[n][key])
    assert set(by_scene) == set(INITS) and {len(v) for v in by_scene.values()} == {6}
    return [median(by_scene[i]) for i in INITS]


def summarize_probe(data):
    a, b = data["early"], data["late"]
    base = scenario_metrics(a, "gain_H")
    late = scenario_metrics(b, "gain_H")
    result = {}
    for k in GROUPS:
        add = scenario_metrics(data[f"add:{k}"], "gain_H")
        undo = scenario_metrics(data[f"undo:{k}"], "gain_H")
        random_add = scenario_metrics(data[f"random_add:{k}"], "gain_H")
        random_replace = scenario_metrics(data[f"random_replace:{k}"], "gain_H")
        contrasts = {
            "drop_on_add": [x-y for x, y in zip(base, add)],
            "rescue_on_undo": [x-y for x, y in zip(undo, late)],
            "additional_drop_vs_random_add": [x-y for x, y in zip(random_add, add)],
            "additional_drop_vs_random_replace": [x-y for x, y in zip(random_replace, late)],
        }
        result[k] = {label: {"median": median(vals), "positive": sum(v > 0 for v in vals)}
                     for label, vals in contrasts.items()}
    rank = sorted(GROUPS, key=lambda k: min(result[k][c]["median"] for c in ("drop_on_add", "rescue_on_undo")), reverse=True)
    return {"group_effects": result, "rank": rank, "full_drop": median(x-y for x, y in zip(base, late))}


def probe(lab, window, start, end):
    bank = make_bank(lab, INITS)
    data, controls = {}, {}
    window.apply("early")
    early = fields(lab, bank)
    window.apply("late")
    late = fields(lab, bank)
    for condition, value, step in (("early", early, start), ("late", late, end)):
        rows = summarize_fields(bank, value, early, late)
        previous = read(K18 / f"probe_channels_{step:06d}.json")
        for cur, old in zip(rows, previous, strict=True):
            assert (cur["init"], cur["route"], cur["denoise_index"]) == (old["init"], old["route"], old["denoise_index"])
            for horizon in ("10", "50"):
                assert abs(cur[horizon]["gain_H"]-old[horizon]["contrast_by_channel"]["H"]["gain"]) < 2e-6
                assert abs(cur[horizon]["gain"]-old[horizon]["gain"]) < 2e-6
        data[condition] = rows
    for k in GROUPS:
        for mode in ("add", "undo", "random_add", "random_replace"):
            condition = f"{mode}:{k}"
            control = window.apply(condition, verify_random=True)
            if control is not None:
                controls[condition] = control
            data[condition] = summarize_fields(bank, fields(lab, bank), early, late)
            print("PROBE", window.label, condition, flush=True)
    summary = summarize_probe(data)
    group = summary["rank"][0]
    selected = ["early", "late", *(f"{mode}:{group}" for mode in ("add", "undo", "random_add", "random_replace"))]
    write(RUN / f"probe_{window.label}.json", {"summary": summary, "conditions": data, "random_checks": controls})
    write(RUN / f"audit_{window.label}.json", window.audit)
    write(RUN / f"selection_{window.label}.json", {"conditions": selected, "top_group": group,
          "rule": "largest minimum of median paired H-gain loss on add and H-gain recovery on undo, before native scores"})
    print("PROBE_COMPLETE", window.label, group, flush=True)


def native(lab, window, lane, lanes, conditions):
    saved = snapshots()
    completed = []
    with (RUN / f"native_{window.label}_w{lane}.jsonl").open("x") as handle:
        for condition in conditions:
            window.apply(condition)
            for init in INITS[lane::lanes]:
                snap = saved[init]
                for goal in "ABN":
                    if goal == "N":
                        obs, _ = lab.ctx.reset(init)
                        z0, rows = float(obs[lab.ctx.obj][2]), []
                    else:
                        obs, rows = rebuild(lab, snap)
                        z0 = snap["z0"]
                        assert not any(r["hits"] or any(r["pred"].values()) for r in rows)
                    boundary = len(rows)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for r in range(len(rows)//10, 40):
                            chunk = lab.native("F", INSTR[goal], obs, snap["seed"], r)
                            obs, part = lab.execute(obs, lab.rt.physical(chunk), r*10, z0)
                            rows.extend(part)
                    assert len(rows) == 400
                    move = lab.ctx.geo.movement("scene2", lab.ctx.geometry, rows, z0, None)
                    result = {"condition": condition, "init": init, "goal": goal, "first": move["first_targets"],
                              "first_step": move["first_step"], "times": move["times"], "boundary": boundary,
                              "trace": [{k: row[k] for k in ("step", "obj", "eef", "relquat", "c1", "c2")}
                                        for row in rows if row["step"] >= boundary and row["step"] % 10 == 0],
                              "ever": {g: any(x["pred"][g] for x in rows)
                              for g in lab.ctx.geo.GOALS}, "terminal": rows[-1]["pred"]}
                    handle.write(json.dumps(result, allow_nan=False) + "\n")
                    handle.flush()
                    print("NATIVE", window.label, condition, init, goal, result["first"], flush=True)
            completed.append(condition)
    write(RUN / f"native_{window.label}_w{lane}_complete.json", {"conditions": completed, "lane": lane})


def selected_after_baselines(window):
    yield "early"
    yield "late"
    selection = RUN / f"selection_{window.label}.json"
    while not selection.exists():
        time.sleep(5)
    yield from read(selection)["conditions"][2:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "native"))
    parser.add_argument("--window", nargs=2, type=int, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--select-after-baselines", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    lab = Lab5()
    try:
        window = Window(lab, *args.window, audit=args.mode == "probe")
        if args.mode == "probe":
            probe(lab, window, *args.window)
        else:
            if args.select_after_baselines:
                assert args.conditions is None
                selected = selected_after_baselines(window)
            else:
                selected = args.conditions or read(RUN / f"selection_{window.label}.json")["conditions"]
            native(lab, window, args.lane, args.lanes, selected)
    finally:
        lab.ctx.env.close()
