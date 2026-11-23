"""Replace late LF common response with parent common response at identical inputs.

The donor is queried only on the two fixed old canonical prompts. It never
receives the actual query. This is a two-policy diagnostic, not a single-policy
K25 repair. The executor, V anchor, sampling and scoring stay fixed.
"""

import argparse
from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
TIMING = RUN.parent/"k25_v_timing_20261122_01"
sys.path.insert(0,str(TIMING))
import v_timing as timing
import mu_swap_protocol as spec
from mu_swap_math import compose
from anchor_tensor_hash import tensors_hash

base = timing.base


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare():
    assert not (RUN/"protocol.json").exists()
    assert shutil.disk_usage(RUN).free > 256*1024**2
    old = base.read(TIMING/"protocol.json")
    previous = base.read(TIMING/"runtime.json")
    assert digest(Path(timing.__file__)) == old["source_sha256"]
    assert digest(Path(base.__file__)) == old["base_source_sha256"]
    assert digest(Path(base.sr.__file__)) == old["baseline_source_sha256"]
    for relative,expected in previous["validated_dependency_sha256"].items():
        assert digest(Path("/home/lzx")/relative) == expected,relative
    cases = base.read(TIMING/"cases.json")
    assert [r["init"] for r in cases] == list(spec.IDS)
    ctx = base.sr.km.Ctx("scene2")
    try:
        assert ctx.geometry == old["geometry"]
        for case in cases:
            assert ctx.signature(base.sr.reset(ctx,case)) == case["signature"]
    finally:
        ctx.env.close()
    cp = old["checkpoints"][spec.LINE]
    for label in ("parent","anchor","late"):
        path = Path(cp[label])/("model.safetensors" if label == "parent" else "")
        assert digest(path) == cp[label+"_sha256"]
    versions = {name:importlib.metadata.version(name) for name in previous["packages"]}
    assert versions == previous["packages"]
    base.write(RUN/"cases.json",cases)
    base.write(RUN/"protocol.json",dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        line=spec.LINE,ids=spec.IDS,goals=spec.GOALS,jobs=spec.jobs(),expected=spec.EXPECTED,
        actual_replayed_references=200,new_conditions=400,units=40,horizon=400,checkpoints=cp,
        geometry=old["geometry"],versions=versions,cases_sha256=digest(RUN/"cases.json"),
        timing_protocol_sha256=digest(TIMING/"protocol.json"),
        sources={name:digest(RUN/name) for name in ("mu_swap.py","mu_swap_protocol.py","mu_swap_math.py","anchor_tensor_hash.py","v_timing.py","v_timing_protocol.py")},
        base_source_sha256=old["base_source_sha256"],baseline_source_sha256=old["baseline_source_sha256"],
        processors={name:digest(Path(cp["parent"])/name) for name in ("config.json","policy_preprocessor_step_5_normalizer_processor.safetensors","policy_postprocessor_step_0_unnormalizer_processor.safetensors")},
        validated_dependency_sha256=previous["validated_dependency_sha256"],
        rule="V anchor30 and gain5.294 fixed; matched background contrast changes only mu. Own-residual gain2.959 is a separate K14-style control, not multiplied into the V condition.",
        donor_rule="Parent0 at current observation, x and t; only fixed old A/B prompts; arithmetic mean only; never actual q or donor contrast.",
        scope="Two-policy diagnostic; same20 reused resets; no independent holdout, wording generalization or new-task preservation claim.",
        parent_commit="362eb5fc3264c214c894e0ad3cf835eb9d86cf41"))
    print("MU_PREPARED",spec.EXPECTED,"actual episodes, including 200 full reference replays",flush=True)


class MixedMean(base.sr.Restore):
    def __init__(self,lab,parent_rt,full,deltas,job):
        super().__init__(lab,full,deltas,spec.GAIN,0.)
        self.parent_rt,self.job = parent_rt,job

    def patch(self,text,obs,seed,block,mode):
        job = self.job
        if job["family"] == "v" and job["background"] == "late":
            return super().patch(text,obs,seed,block,mode)
        prefixes = self.prefixes(obs,(text,*(base.sr.INSTR[g] for g in "ABN")))
        pair = {g:prefixes[base.sr.INSTR[g]] for g in "AB"}
        parent_prefix = {g:self.parent_rt.prefix("F",obs,base.sr.INSTR[g]) for g in "AB"} if job["background"] == "parent" else {}
        x0 = self.rt.noise(seed,block)
        t0 = torch.ones(x0.shape[0],device=x0.device)
        initial = {q:self.value(p,x0,t0) for q,p in prefixes.items()}
        route = base.sr.rank(initial[text],*(initial[base.sr.INSTR[g]] for g in "ABN"))
        sign = 1. if job["arm"] == "forward" else -1.
        coefficient = job["gain"]*sign*base.sr.SIGN[route["decoded"]]
        relay = self.relays[1.]
        projected = relay.cache(pair) if coefficient else None
        count = parent_calls = 0
        gaps,residuals,deltas = [],[],[]

        def velocity(x,t):
            nonlocal count,parent_calls
            cache = initial if count == 0 else {}
            def evaluate(q):
                if q not in cache:
                    cache[q] = self.value(prefixes[q],x,t)
                return cache[q]
            count += 1
            actual = evaluate(text)
            a,b = (evaluate(base.sr.INSTR[g]) for g in "AB")
            late_mean = (a.float()+b.float())*.5
            if parent_prefix:
                pa,pb = (self.parent_rt.query("F",parent_prefix[g],x,t) for g in "AB")
                parent_calls += 2
                parent_mean = (pa.float()+pb.float())*.5
                gaps.append(base.sr.rms((parent_mean-late_mean)[...,:7]))
            else:
                parent_mean = late_mean
            delta = self.local_contrast(relay,projected,pair,x,t)-base.sr.contrast(a,b) if coefficient else 0.
            if coefficient:
                deltas.append(base.sr.rms(delta[...,:7]))
            residuals.append(base.sr.rms((actual.float()-late_mean)[...,:7]))
            out = compose(actual.float(),late_mean,parent_mean,delta,job["background"],job["beta"],coefficient)
            return out.to(actual.dtype)

        controls = self.rt.integrate(velocity,x0,10)[...,:7]
        assert count == 10 and parent_calls == (20 if parent_prefix else 0)
        route.update(block=block,coefficient=coefficient,contrast_rms=sum(deltas)/len(deltas) if deltas else 0.,
            mean_gap_rms=sum(gaps)/len(gaps) if gaps else 0.,late_residual_rms=sum(residuals)/len(residuals),
            parent_velocity_queries=parent_calls,parent_prefix_prompts=list("AB") if parent_prefix else [])
        return controls,route


def gates(lab,parent_rt,full,delta,case):
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
        vgate = base.gate(lab,base.sr.Restore(lab,full,delta,spec.GAIN,0.),full,delta,case)
        obs = base.sr.reset(lab.ctx,case)
        x = lab.rt.noise(case["seed"],0)
        t = torch.ones(x.shape[0],device=x.device)
        f = {g:lab.rt.query("F",lab.rt.prefix("F",obs,base.sr.INSTR[g]),x,t) for g in "AB"}
        p = {g:parent_rt.query("F",parent_rt.prefix("F",obs,base.sr.INSTR[g]),x,t) for g in "AB"}
        mf = (f["A"].float()+f["B"].float())*.5
        mp = (p["A"].float()+p["B"].float())*.5
        prefix = {g:lab.rt.prefix("F",obs,base.sr.INSTR[g]) for g in "AB"}
        relay = base.sr.Restore(lab,full,delta,spec.GAIN,0.).relays[1.]
        cache = relay.cache(prefix)
        with relay.whole(cache,"A",False):
            va = lab.rt.query("F",prefix["A"],x,t)
        with relay.whole(cache,"B",False):
            vb = lab.rt.query("F",prefix["B"],x,t)
        dd = base.sr.contrast(va,vb)-base.sr.contrast(f["A"],f["B"])
        mf64 = (f["A"].double()+f["B"].double())*.5
        mp64 = (p["A"].double()+p["B"].double())*.5
        checks=[]
        for beta,gain in ((1.,0.),(1.,spec.GAIN),(spec.BETA,0.)):
            for polarity in ((1.,-1.) if gain else (1.,)):
                output = {}
                for bg in ("late","parent"):
                    output[bg] = {g:compose(f[g].double(),mf64,mp64,dd.double(),bg,beta,
                                           gain*polarity*base.sr.SIGN[g]) for g in "AB"}
                for g in "AB":
                    torch.testing.assert_close(output["parent"][g]-output["late"][g],mp64-mf64,atol=1e-12,rtol=1e-12)
                torch.testing.assert_close(output["parent"]["A"]-output["parent"]["B"],
                    output["late"]["A"]-output["late"]["B"],atol=1e-12,rtol=1e-12)
                cast_errors=[]
                for bg,expected in (("late",mf64),("parent",mp64)):
                    torch.testing.assert_close((output[bg]["A"]+output[bg]["B"])*.5,expected,atol=1e-12,rtol=1e-12)
                    cast={g:compose(f[g].float(),mf,mp,dd,bg,beta,gain*polarity*base.sr.SIGN[g]).to(f[g].dtype) for g in "AB"}
                    cast_errors.append(base.sr.rms(((cast["A"].float()+cast["B"].float())*.5-expected.float())[...,:7]))
                checks.append(dict(beta=beta,gain=gain,polarity=polarity,shared_shift_equal=True,contrast_preserved=True,pair_mean_correct=True,cast_mean_rms_max=max(cast_errors)))
        for g in "AB":
            assert torch.equal(compose(f[g].float(),mf,mp,0.,"late",1.,0.).to(f[g].dtype),f[g])
        return dict(executor_runtime_roles=list(lab.rt.models),donor_runtime_roles=list(parent_rt.models),
            actual_policies=2,donor_sha256=parent_rt.sha,donor_canonical_queries_only=True,
            parent_mean_gap_rms=base.sr.rms((mp-mf)[...,:7]),native_identity_bitwise=True,
            v_gate=vgate,response_gates=checks)


def references(init,goal):
    lane = list(spec.IDS).index(init)%8
    path = TIMING/f"{spec.LINE}_{lane}.jsonl.gz"
    marker = base.read(TIMING/f"{spec.LINE}_{lane}_COMPLETE.json")
    assert digest(path) == marker["rows_sha256"]
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        return {(r["window"],r["arm"]):r for r in map(json.loads,stream) if r["init"] == init and r["goal"] == goal}


def worker(lane):
    protocol = base.read(RUN/"protocol.json")
    for name,expected in protocol["sources"].items():
        assert digest(RUN/name) == expected,name
    base.configure(spec.LINE)
    init,goal = spec.unit(lane)
    case = next(r for r in base.read(RUN/"cases.json") if r["init"] == init)
    cp = protocol["checkpoints"]
    parent = Path(cp["parent"])
    full = base.selected_weights(Path(cp["late"]))
    early = base.selected_weights(Path(cp["anchor"]))
    _,selected = base.sr.parameters_for("cross_v",full)
    delta = {n:early[n].double()-full[n].double() for n in sorted(selected)}
    dh = tensors_hash(delta)
    old = references(init,goal)
    assert old["full_400","forward"]["delta_sha256"] == dh
    lab = base.Lab(parent)
    base.sr.assign(lab,"F",full)
    parent_rt = base.sr.km.make_runtime(lab.ctx,{"FT4":str(parent)},["F"])
    assert parent_rt.sha == {"F":cp["parent_sha256"]}
    parent_weights = base.selected_weights(parent/"model.safetensors")
    for rt,expected in ((lab.rt,full),(parent_rt,parent_weights)):
        params = dict(rt.models["F"].named_parameters())
        assert all(torch.equal(params[n].detach().cpu(),v) for n,v in expected.items())
    gate = gates(lab,parent_rt,full,delta,case)
    print("MU_READY",lane,init,goal,"CUDA_PEAK_RESERVED",torch.cuda.max_memory_reserved(),flush=True)
    dest = RUN/f"unit_{lane:02d}.jsonl.gz"
    assert not dest.exists()
    replayed=[]
    count=0
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            with gzip.open(dest,"xt",encoding="utf-8",compresslevel=5) as out:
                for job in spec.jobs():
                    model = MixedMean(lab,parent_rt,full,delta,job) if job["family"] != "native" else None
                    row = timing.episode(lab,model,case,base.sr.INSTR[goal],job["window"],job["arm"])
                    if spec.reference_job(job):
                        r = old[job["window"],job["arm"]]
                        assert all(row[k] == r[k] for k in ("trace","signatures","action_hashes","decisions","first","first_step","times","terminal","first_grasp","ever","pre","post")),(lane,job,"reference mismatch")
                        replayed.append(list(spec.job_key(job)))
                    row.update(**job,line=spec.LINE,goal=goal,target=base.sr.LABEL[goal],seed=case["seed"],
                        anchor=30,delta_sha256=dh,reference_replay=spec.reference_job(job))
                    out.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
                    out.flush()
                    count+=1
                    print("MU",lane,job["family"],job["background"],job["window"],job["arm"],init,goal,row["first"],row["terminal"][row["target"]],round(row["seconds"],2),flush=True)
    finally:
        lab.ctx.env.close()
    assert count == 15 and len(replayed) == 5
    for rt,expected in ((lab.rt,full),(parent_rt,parent_weights)):
        params = dict(rt.models["F"].named_parameters())
        assert all(torch.equal(params[n].detach().cpu(),v) for n,v in expected.items())
    base.write(RUN/f"unit_{lane:02d}_COMPLETE.json",dict(lane=lane,init=init,goal=goal,episodes=count,
        rows_sha256=digest(dest),protocol_sha256=digest(RUN/"protocol.json"),sources=protocol["sources"],
        delta_sha256=dh,reference_replayed=replayed,gates=gate,
        executor_weights_sha256=tensors_hash(full),parent_weights_sha256=tensors_hash(parent_weights),
        weights_unchanged_at_end=True,executor_processor=lab.rt.processor_check,donor_processor=parent_rt.processor_check,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","worker"))
    parser.add_argument("--lane",type=int)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        prepare()
    else:
        assert args.lane is not None and 0 <= args.lane < 40
        worker(args.lane)
