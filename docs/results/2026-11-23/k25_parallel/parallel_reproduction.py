"""Reproduce K22's original parallel component, including its full-32D output.

The unmodified original combine2('par') is the actual velocity implementation.
An adapter supplies the matched parent and continued policy and actual q.
"""

import argparse
from collections import defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import importlib.metadata
import gzip
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file

RUN=Path(__file__).resolve().parent
SERVER=Path("/home/lzx/cl_binding/.server")
sys.path[:0]=[str(SERVER/"k22_deepen_20261113_01/code"),str(SERVER/"k25_v_timing_20261122_01")]
import k22 as original
import v_timing as timing
import parallel_protocol as spec
from anchor_tensor_hash import tensors_hash

base=timing.base


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


class DualRuntime:
    def __init__(self,executor,parent):
        self.executer,self.parent=executor,parent
        self.torch=torch
        self.integrate=executor.integrate
        self.values={}

    def runtime(self,role):
        return self.parent if role == "R" else self.executer

    def prefix(self,role,obs,text):
        return role,text,self.runtime(role).prefix("F",obs,text)

    def query(self,role,prefix,x,t):
        assert prefix[0] == role
        value=self.runtime(role).query("F",prefix[2],x,t)
        self.vdtype=value.dtype
        self.values[role,prefix[1]]=value
        return value

    def noise(self,seed,block):
        return self.executer.noise(seed,block)

    def physical(self,chunk):
        return self.executer.physical(chunk)

    def sample(self,condition,key,obs,texts,seed,block):
        role="R" if condition == "NAT-R0" else "F"
        return self.runtime(role).sample("NAT-FT4",key,obs,texts,seed,block)


def load(profile,config):
    cp=config["models"][profile]
    lab=original.k21.Lab(family="scene2",models=False)
    lab.line,lab.family=profile,"scene2"
    lab.goal=spec.GOALS[profile]
    lab.text={g:lab.ctx.all_texts[v] for g,v in lab.goal.items()}
    assert all(q == f"pick up the book and place it in the {lab.goal[g]} compartment of the caddy" for g,q in lab.text.items())
    fpath=cp["parent"] if profile.endswith("1000") else cp["executor"]
    frt=original.k21.km.make_runtime(lab.ctx,{"FT4":fpath},["F"])
    if profile.endswith("1000"):
        base.sr.assign(type("Holder",(),{"rt":frt})(),"F",base.selected_weights(Path(cp["executor"])))
    prt=original.k21.km.make_runtime(lab.ctx,{"FT4":cp["parent"]},["F"])
    normalizers={}
    for name in ("policy_preprocessor_step_5_normalizer_processor.safetensors","policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        paths=[Path(fpath)/name,Path(cp["parent"])/name]
        tensors=[load_file(str(p)) for p in paths]
        keys=[k for k in tensors[0] if k.startswith(("action.","observation."))]
        assert all(torch.equal(tensors[0][k].double().flatten(),tensors[1][k].double().flatten()) for k in keys)
        normalizers[name]=dict(paths=[str(p) for p in paths],sha256=[digest(p) for p in paths],used_keys=keys,values_equal=True)
    lab.rt=DualRuntime(frt,prt)
    lab.normalizers=normalizers
    return lab


def prepare():
    assert not (RUN/"preparation.json").exists()
    old=base.read(SERVER/"k25_v_timing_20261122_01/protocol.json")
    models={}
    for profile in spec.PROFILES:
        line=profile.replace("12000","1000")
        cp=old["checkpoints"][line]
        parent=cp["parent"]
        if profile.endswith("1000"):
            executor=cp["late"]
        else:
            executor=original.k21.LINES["r1_front" if profile.startswith("lr_") else "r0_s2"]["F"]
        efile=Path(executor) if profile.endswith("1000") else Path(executor)/"model.safetensors"
        models[profile]=dict(parent=parent,executor=executor,parent_sha256=digest(Path(parent)/"model.safetensors"),executor_sha256=digest(efile),
            continuation_note="K25 existing1000 stream" if profile.endswith("1000") else "original K21/K22 endpoint; not assumed same continuation trajectory as1000")
        assert models[profile]["parent_sha256"] == cp["parent_sha256"]
    cases=base.read(SERVER/"k25_v_timing_20261122_01/cases.json")
    ctx=original.k21.km.Ctx("scene2")
    try:
        assert ctx.geometry == old["geometry"]
        for r in cases:
            obs=base.sr.reset(ctx,r)
            assert ctx.signature(obs) == r["signature"]
    finally:
        ctx.env.close()
    base.write(RUN/"cases.json",cases)
    sources={n:digest(RUN/n) for n in ("parallel_reproduction.py","parallel_protocol.py","v_timing_protocol.py","anchor_tensor_hash.py")}
    originals={"k22.py":digest(Path(original.__file__)),"k21.py":digest(Path(original.k21.__file__)),"k22p3.py":digest(SERVER/"k22_deepen_20261113_01/code/k22p3.py")}
    runtime=base.read(SERVER/"k25_v_timing_20261122_01/runtime.json")
    for rel,h in runtime["validated_dependency_sha256"].items():
        assert digest(Path("/home/lzx")/rel) == h
    base.write(RUN/"preparation.json",dict(created_at_utc=datetime.now(timezone.utc).isoformat(),models=models,cases_sha256=digest(RUN/"cases.json"),geometry=old["geometry"],
        profiles=spec.PROFILES,ids=spec.IDS,window=spec.WINDOW,horizon=400,expected=160,sources=sources,original_sources=originals,
        validated_dependency_sha256=runtime["validated_dependency_sha256"],versions=runtime["packages"],
        source_parent_commit="19b99a47728cb8cbe1b8fcaef4763f5868179863",only_method="original old mu + parallel component; no new ablations or sweeps",
        calibration_rule="1000 profiles: exact K21 dev inits0-9, adapted N-prefix160steps, first Euler inputs at boundaries30..150 every10;130 norm ratios; upper median; capped10. 12000: original alpha reused."))
    print("PARALLEL_PREPARED",160,flush=True)


def calibrate(profile):
    assert profile.endswith("1000") and not (RUN/f"calibration_{profile}.json").exists()
    cfg=base.read(RUN/"preparation.json")
    lab=load(profile,cfg)
    stats=[]
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for init in range(10):
                seed=original.k21.km.noise_seed("scene2",init)
                obs,_=lab.ctx.reset(init)
                z0,plate=float(obs[lab.ctx.obj][2]),lab.ctx.plate_xy(obs)
                for block in range(16):
                    if block >= 3:
                        stats.append(dict(init=init,step=block*10,**lab.fields(obs,seed,block)))
                    chunk=lab.native("F",lab.text["N"],obs,seed,block)
                    obs,_=lab.execute(obs,lab.rt.physical(chunk),block*10,z0,plate)
                print("CALIBRATION",profile,init,flush=True)
    finally:
        lab.ctx.env.close()
    assert len(stats) == 130
    med={k:spec.upper_median([s[k] for s in stats]) for k in ("ratio","cos","mu_shift_over_dold","muF_to_N_over_muR_to_N")}
    base.write(RUN/f"calibration_{profile}.json",dict(profile=profile,alpha=min(10.,med["ratio"]),n=130,median=med,stats=stats,
        normalizers=lab.normalizers,preparation_sha256=digest(RUN/"preparation.json"),source_sha256=digest(Path(__file__)),behaviour_used_for_selection=False))
    print("CALIBRATED",profile,min(10.,med["ratio"]),med,flush=True)


def lock():
    cfg=base.read(RUN/"preparation.json")
    assert not (RUN/"protocol.json").exists()
    gains={}
    for p in spec.PROFILES:
        if p.endswith("1000"):
            path=RUN/f"calibration_{p}.json"
            cal=base.read(path)
            assert cal["preparation_sha256"] == digest(RUN/"preparation.json")
            gains[p]=dict(alpha=cal["alpha"],source=str(path),sha256=digest(path),rule="same K21 norm-ratio rule on independent development states; no evaluation outcomes")
        elif p.startswith("lr_"):
            gains[p]=dict(alpha=float(original.k21.LINES["r1_front"]["alpha"]),source="original k21.py LINES[r1_front].alpha",sha256=cfg["original_sources"]["k21.py"],rule="reuse original fixed2.959")
        else:
            path=SERVER/"k21_transfer_20261113_01/lines/r0_s2/alpha.json"
            gains[p]=dict(alpha=base.read(path)["alpha"],source=str(path),sha256=digest(path),rule="reuse original K21 norm-ratio calibration")
    base.write(RUN/"protocol.json",dict(**cfg,gains=gains,locked_at_utc=datetime.now(timezone.utc).isoformat(),preparation_sha256=digest(RUN/"preparation.json"),
        method="c=alpha*<rF(q)[:7],dP[:7]>/max(||dP[:7]||²,1e-30); output=(muP+c*dP) full32 cast to native dtype; only first150 controls, then F actual q"))
    print("PROJECTION_LOCKED",{p:g["alpha"] for p,g in gains.items()},flush=True)


def projection(lab,pre,text,x,t,diag):
    lab.rt.values.clear()
    out=original.Lab2.combine2(lab,"par",None,pre,"Q",x,t,diag)
    v=lab.rt.values
    pa,pb=(v["R",lab.text[g]].float() for g in "AB")
    fa,fb=(v["F",lab.text[g]].float() for g in "AB")
    actual=v["F",text].float()
    dp=(pa-pb)*.5
    df=(fa-fb)*.5
    resid=actual-(fa+fb)*.5
    ref,real=dp[0,:,:7],resid[0,:,:7]
    energy=float(ref.square().sum())
    diag["zero_axis"].append(int(energy <= 1e-30))
    diag["parent_norm"].append(float(ref.norm()))
    diag["late_norm"].append(float(df[0,:,:7].norm()))
    diag["residual_norm"].append(float(real.norm()))
    diag["mean_gap_norm"].append(float((((pa+pb)-(fa+fb))*.5)[0,:,:7].norm()))
    diag["coefficient"].append(lab.alpha*diag["c1"][-1])
    return out


def gate(lab,case):
    obs=base.sr.reset(lab.ctx,case)
    x=lab.rt.noise(case["seed"],0)
    t=torch.ones(1,device=x.device)
    tests=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
        for g in "AB":
            pre={(r,k):lab.rt.prefix(r,obs,lab.text[k]) for r in "RF" for k in "AB"}
            pre["F","Q"]=lab.rt.prefix("F",obs,lab.text[g])
            diag=defaultdict(list)
            out=projection(lab,pre,lab.text[g],x,t,diag)
            values=lab.rt.values
            pa,pb=(values["R",lab.text[k]].float() for k in "AB")
            fa,fb=(values["F",lab.text[k]].float() for k in "AB")
            actual=values["F",lab.text[g]].float()
            dp=(pa-pb)/2
            rf=actual-(fa+fb)/2
            c1=(rf[0,:,:7]*dp[0,:,:7]).sum()/dp[0,:,:7].square().sum().clamp_min(1e-30)
            c=lab.alpha*c1
            expected=((pa+pb)/2+c*dp).to(out.dtype)
            assert torch.equal(out,expected)
            without_q={k:v for k,v in pre.items() if k[1] != "Q"}
            original_out=original.Lab2.combine2(lab,"par",None,without_q,g,x,t,defaultdict(list))
            assert torch.equal(out,original_out)
            tests.append(dict(goal=g,original_output_bitwise=True,formula_full32_bitwise=True,coefficient_from_50x7=True,padding_from_parent_axis=True,coefficient=float(c)))
    return tests


def episode(lab,case,goal):
    started=monotonic()
    text=lab.text[goal]
    obs=base.sr.reset(lab.ctx,case)
    trace=dict(start=dict(object=obs[lab.ctx.obj].tolist(),eef=obs["robot0_eef_pos"].tolist()),object=[],eef=[],contacts=[],success=[],regions=[],actions=[])
    signatures=[timing.signature(lab,obs)]
    hashes=[]
    decisions=[]
    for block in range(40):
        if block < 15:
            pre={(r,g):lab.rt.prefix(r,obs,lab.text[g]) for r in "RF" for g in "AB"}
            pre["F","Q"]=lab.rt.prefix("F",obs,text)
            diag=defaultdict(list)
            chunk=lab.rt.integrate(lambda x,t:projection(lab,pre,text,x,t,diag),lab.rt.noise(case["seed"],block),10)[...,:7]
            assert all(len(v) == 10 for v in diag.values())
            decisions.append(dict(block=block,denoising=dict(diag)))
        else:
            chunk=lab.native("F",text,obs,case["seed"],block)
        physical=lab.rt.physical(chunk)
        actual=np.clip(physical[:10],-1,1).astype(np.float32)
        hashes.append(sha256(actual.tobytes()).hexdigest())
        obs,events=lab.execute(obs,physical,block*10,case["z0"],None)
        for row,action in zip(events,actual,strict=True):
            for k,field in (("object","obj"),("eef","eef")):
                trace[k].append(row[field])
            trace["contacts"].append(int(row["c1"])+2*int(row["c2"]))
            trace["success"].append(sum(1<<j for j,g in enumerate(spec.scoring.SITES) if row["pred"][g]))
            trace["regions"].append(sum(1<<j for j,g in enumerate(spec.scoring.SITES) if g in row["hits"]))
            trace["actions"].append(action.tolist())
        signatures.append(timing.signature(lab,obs))
    result=spec.scoring.score(trace,lab.ctx.geometry,case["z0"])
    return dict(init=case["init"],goal=goal,target=lab.goal[goal],text=text,seed=case["seed"],alpha=lab.alpha,window=[0,150],
        **result,**spec.scoring.interval_metrics(trace,lab.ctx.geometry,case["z0"],0,150),trace=trace,signatures=signatures,action_hashes=hashes,
        decisions=decisions,seconds=monotonic()-started)


def worker(profile,lane):
    p=base.read(RUN/"protocol.json")
    for n,h in p["sources"].items():
        assert digest(RUN/n) == h,n
    assert digest(Path(original.__file__)) == p["original_sources"]["k22.py"]
    assert digest(Path(original.k21.__file__)) == p["original_sources"]["k21.py"]
    lab=load(profile,p)
    lab.alpha=p["gains"][profile]["alpha"]
    cases=[r for r in base.read(RUN/"cases.json") if r["init"] in spec.ids(lane)]
    checks=gate(lab,cases[0])
    paths=(lab.rt.executer,lab.rt.parent)
    selected=set(base.names())
    before=[tensors_hash({n:v.detach().cpu() for n,v in rt.models["F"].named_parameters() if n in selected}) for rt in paths]
    dest=RUN/f"{profile}_{lane}.jsonl.gz"
    assert not dest.exists()
    count=0
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            with gzip.open(dest,"xt",encoding="utf-8",compresslevel=5) as out:
                for case in cases:
                    for goal in "AB":
                        r=episode(lab,case,goal)
                        r["profile"]=profile
                        out.write(json.dumps(r,allow_nan=False,separators=(",",":"))+"\n")
                        out.flush()
                        count+=1
                        print("PARALLEL",profile,lane,case["init"],goal,r["first"],r["terminal"][r["target"]],round(r["seconds"],2),flush=True)
    finally:
        lab.ctx.env.close()
    after=[tensors_hash({n:v.detach().cpu() for n,v in rt.models["F"].named_parameters() if n in selected}) for rt in paths]
    assert before == after and count == 2*len(cases)
    base.write(RUN/f"{profile}_{lane}_COMPLETE.json",dict(profile=profile,lane=lane,ids=spec.ids(lane),episodes=count,
        gates=checks,normalizers=lab.normalizers,weights_fingerprints=before,weights_unchanged=True,rows_sha256=digest(dest),
        protocol_sha256=digest(RUN/"protocol.json"),sources=p["sources"],original_sources=p["original_sources"]))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","calibrate","lock","worker"))
    parser.add_argument("--profile",choices=spec.PROFILES)
    parser.add_argument("--lane",type=int,default=0)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare": prepare()
    elif args.mode == "calibrate": calibrate(args.profile)
    elif args.mode == "lock": lock()
    else: worker(args.profile,args.lane)
