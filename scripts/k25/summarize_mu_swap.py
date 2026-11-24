"""Rescore the complete common-response factorial, retaining every failed episode."""

import argparse
from collections import Counter
from datetime import datetime
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import struct

import mu_swap_protocol as p
import summarize_v_timing as previous_analysis


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def sig(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def rows(path,partial=False):
    try:
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            while True:
                try:
                    text=stream.readline()
                except EOFError:
                    if partial:
                        return
                    raise
                if not text:
                    return
                yield json.loads(text)
    except EOFError:
        if not partial:
            raise


def archive(base,short,remote,date):
    candidates=(base.parent/short,base.parent.parent/date/short,base.parent/remote)
    found=[path for path in candidates if (path/"protocol.json").exists()]
    assert found,("Missing reference archive",candidates)
    return found[0]


def references(base,protocol):
    old=archive(base,"k25_v_timing","k25_v_timing_20261122_01","2026-11-22")
    order=archive(base,"k25_order_restore","k25_order_restore_20261122_01","2026-11-22")
    anchor=archive(base,"k25_anchor","k25_anchor_20261123_02","2026-11-23")
    assert digest(old/"protocol.json") == protocol["timing_protocol_sha256"]
    op=read(old/"protocol.json")
    assert op["checkpoints"][p.LINE] == protocol["checkpoints"]
    assert op["source_sha256"] == protocol["sources"]["v_timing.py"]
    assert op["spec_sha256"] == protocol["sources"]["v_timing_protocol.py"]
    assert digest(Path(previous_analysis.__file__)) == read(old/"summary.json")["analysis_source_sha256"]
    assert digest(order/"protocol.json") == op["previous_protocol_sha256"]
    for relative,expected in protocol["validated_dependency_sha256"].items():
        assert digest(order/"dependencies"/relative) == expected,relative
    archived=order/"dependencies/cl_binding/.server"
    assert digest(archived/"k25_order_restore_20261122_01/order_restore.py") == protocol["base_source_sha256"]
    assert digest(archived/"k25_final_20261121_01/self_restore.py") == protocol["baseline_source_sha256"]
    assert read(old/"cases.json") == read(base/"cases.json") == read(anchor/"cases.json")
    result={}
    for lane in range(8):
        path=old/f"{p.LINE}_{lane}.jsonl.gz"
        marker=read(old/f"{p.LINE}_{lane}_COMPLETE.json")
        assert digest(path) == marker["rows_sha256"]
        for row in rows(path):
            if row["window"] == "native" or row["window"] in p.WINDOWS:
                result[row["window"],row["arm"],row["goal"],row["init"]]=row
    assert len(result) == 200
    donor={}
    ap=read(anchor/"protocol.json")
    assert ap["checkpoints"][p.LINE+":0"]["sha256"] == protocol["checkpoints"]["parent_sha256"]
    summary=read(anchor/"summary.json")
    manifest=read(anchor/"artifact_manifest.json")
    summary_name="docs/results/2026-11-23/k25_anchor/summary.json"
    assert digest(anchor/"summary.json") == manifest["files"][summary_name]["sha256"]
    for lane in range(8):
        path=anchor/f"{p.LINE}_{lane}.jsonl.gz"
        marker=read(anchor/f"{p.LINE}_{lane}_COMPLETE.json")
        assert digest(path) == marker["rows_sha256"]
        for row in rows(path):
            if row["family"] == "policy" and row["anchor"] == 0 and row["goal"] in "AB":
                donor[row["goal"],row["init"]]=row
    assert len(donor) == 40
    return result,donor,summary["weights_fingerprints"][p.LINE],dict(
        timing_protocol_sha256=digest(old/"protocol.json"),
        donor_archive_manifest_sha256=digest(anchor/"artifact_manifest.json"),donor_summary_sha256=digest(anchor/"summary.json"))


def dense(row,protocol,case):
    target=p.GOALS[row["goal"]]
    assert row["line"] == p.LINE and row["seed"] == case["seed"] and row["target"] == target
    assert row["text"] == f"pick up the book and place it in the {target} compartment of the caddy"
    result=p.temporal.score(row["trace"],protocol["geometry"],case["z0"])
    assert all(row[k] == v for k,v in result.items())
    offsets=(0,400) if row["window"] == "native" else p.temporal.WINDOWS[row["window"]]
    assert row["offsets"] == list(offsets)
    interval=p.temporal.interval_metrics(row["trace"],protocol["geometry"],case["z0"],*offsets)
    for k,v in interval.items():
        if k == "first_10_approach_mm":
            assert all(math.isclose(row[k][g],x,rel_tol=1e-10,abs_tol=1e-7) for g,x in v.items())
        else:
            assert row[k] == v,k
    assert len(row["signatures"]) == 41 and row["signatures"][0] == sig(case["signature"])
    assert len(row["action_hashes"]) == 40
    for block,expected in enumerate(row["action_hashes"]):
        values=[v for a in row["trace"]["actions"][block*10:(block+1)*10] for v in a]
        assert sha256(struct.pack("<70f",*values)).hexdigest() == expected


def validate(row,protocol,case):
    dense(row,protocol,case)
    assert row["anchor"] == 30 and row["reference_replay"] == p.reference_job(row)
    decisions=row["decisions"]
    active=[] if row["family"] == "native" else p.temporal.active_blocks(row["window"])
    assert [d["block"] for d in decisions] == active
    assert row["routed"] == dict(Counter(d["decoded"] for d in decisions))
    for d in decisions:
        assert d["decoded"] == min("ABN",key=d["distance"].__getitem__)
        assert d["distance"][row["goal"]] == 0.
        assert all(math.isfinite(v) and v >= 0 for v in d["distance"].values())
        sign=1. if row["arm"] == "forward" else -1.
        assert d["coefficient"] == row["gain"]*sign*{"A":1.,"B":-1.,"N":0.}[d["decoded"]]
        assert math.isfinite(d["contrast_rms"]) and d["contrast_rms"] >= 0
        if row["family"] != "v":
            assert d["contrast_rms"] == d["coefficient"] == 0
        if row["background"] == "parent":
            assert d["parent_velocity_queries"] == 20 and d["parent_prefix_prompts"] == ["A","B"]
        else:
            assert d.get("parent_velocity_queries",0) == 0
        for field in ("mean_gap_rms","late_residual_rms"):
            if field in d:
                assert math.isfinite(d[field]) and d[field] >= 0


def collect(base,partial=False):
    base=base.resolve()
    protocol=read(base/"protocol.json")
    assert protocol["jobs"] == p.jobs() and protocol["expected"] == 600 and protocol["units"] == 40
    assert protocol["ids"] == list(p.IDS) and protocol["goals"] == p.GOALS and protocol["line"] == p.LINE
    assert protocol["cases_sha256"] == digest(base/"cases.json")
    for name,expected in protocol["sources"].items():
        assert digest(Path(__file__).with_name(name)) == expected,name
    runtime=read(base/"runtime.json")
    assert runtime["capture_source_sha256"] == digest(base/"capture_runtime.py")
    assert runtime["queue_source_sha256"] == digest(Path(__file__).with_name("run_mu_swap.py"))
    assert runtime["packages"] == protocol["versions"] and runtime["dependency_sha256"] == protocol["validated_dependency_sha256"]
    assert sorted(runtime["actual_shared_modules"].values()) == sorted(protocol["sources"][n] for n in ("v_timing.py","v_timing_protocol.py"))
    assert runtime["analysis_plan_sha256"] == digest(base/"analysis_plan.json")
    plan=read(base/"analysis_plan.json")
    assert plan["locked_before_queue"]
    assert datetime.fromisoformat(protocol["created_at_utc"]) < datetime.fromisoformat(plan["created_at_utc"]) < datetime.fromisoformat(runtime["captured_at_utc"])
    for name,record in read(base/"history/sources.json").items():
        assert Path(name).name == name and digest(base/"history"/name) == record["sha256"]
    cases={r["init"]:r for r in read(base/"cases.json")}
    reference,donor,fingerprints,provenance=references(base,protocol)
    lookup={}
    completed=replayed=0
    cast=[]
    delta_sha=None
    for lane in range(40):
        init,goal=p.unit(lane)
        file=base/f"unit_{lane:02d}.jsonl.gz"
        mark=base/f"unit_{lane:02d}_COMPLETE.json"
        if partial and not file.exists():
            continue
        expected=p.expected_keys([init],goal)
        found=set()
        for r in rows(file,partial):
            k=p.key(r)
            assert k in expected and k not in lookup
            validate(r,protocol,cases[init])
            lookup[k]=r
            found.add(k)
            if r["reference_replay"]:
                old=reference[r["window"],r["arm"],goal,init]
                assert all(r[f] == old[f] for f in ("trace","signatures","action_hashes","decisions","first","first_step","times","terminal","first_grasp","ever","pre","post"))
                replayed+=1
        if partial and not mark.exists():
            continue
        m=read(mark)
        assert found == expected and m["episodes"] == 15
        assert (m["lane"],m["init"],m["goal"]) == (lane,init,goal)
        assert m["rows_sha256"] == digest(file) and m["protocol_sha256"] == digest(base/"protocol.json")
        assert m["sources"] == protocol["sources"] and m["weights_unchanged_at_end"]
        assert m["executor_weights_sha256"] == fingerprints["1000"] and m["parent_weights_sha256"] == fingerprints["0"]
        assert m["executor_processor"] == m["donor_processor"]
        assert len(m["reference_replayed"]) == 5
        assert {tuple(x) for x in m["reference_replayed"]} == {p.job_key(j) for j in p.jobs() if p.reference_job(j)}
        if delta_sha is not None:
            assert delta_sha == m["delta_sha256"]
        delta_sha=m["delta_sha256"]
        assert all(lookup[k]["delta_sha256"] == delta_sha for k in found)
        assert all(r["delta_sha256"] == delta_sha for r in reference.values())
        g=m["gates"]
        assert g["actual_policies"] == 2 and g["executor_runtime_roles"] == g["donor_runtime_roles"] == ["F"]
        assert g["donor_sha256"] == {"F":protocol["checkpoints"]["parent_sha256"]}
        assert g["donor_canonical_queries_only"] and g["native_identity_bitwise"]
        vg=g["v_gate"]
        assert vg["single_model_roles"] == ["F"]
        assert all(vg[f] for f in ("v_only_overlay_equal","weights_restored","mean_preserved_fp64","new_goal_bypass_bitwise"))
        assert len(g["response_gates"]) == 4
        assert {(c["beta"],c["gain"],c["polarity"]) for c in g["response_gates"]} == {(1.,0.,1.),(1.,p.GAIN,1.),(1.,p.GAIN,-1.),(p.BETA,0.,1.)}
        for c in g["response_gates"]:
            assert all(c[f] for f in ("shared_shift_equal","contrast_preserved","pair_mean_correct"))
            assert math.isfinite(c["cast_mean_rms_max"]) and c["cast_mean_rms_max"] >= 0
            cast.append(c["cast_mean_rms_max"])
        completed+=1
    native_job=p.jobs()[0]
    prefix_checks=background_reads=reverse_reads=0
    for r in lookup.values():
        if r["family"] == "native":
            continue
        before=lookup.get((*p.job_key(native_job),r["goal"],r["init"]))
        if before is None:
            assert partial
            continue
        previous_analysis.same_prefix(before,r,p.temporal.WINDOWS[r["window"]][0]//10)
        prefix_checks+=1
        if r["background"] == "parent" and r["family"] in ("v","own"):
            job={**{k:r[k] for k in p.FIELDS[:6]},"background":"late"}
            before=lookup.get((*p.job_key(job),r["goal"],r["init"]))
            if before is not None:
                previous_analysis.same_prefix(before,r,p.temporal.WINDOWS[r["window"]][0]//10)
                assert before["decisions"][0]["distance"] == r["decisions"][0]["distance"]
                assert before["decisions"][0]["coefficient"] == r["decisions"][0]["coefficient"]
                background_reads+=1
        if r["family"] == "v" and r["arm"] == "reverse":
            job={**{k:r[k] for k in p.FIELDS[:6]},"arm":"forward"}
            before=lookup.get((*p.job_key(job),r["goal"],r["init"]))
            if before is not None:
                previous_analysis.same_prefix(before,r,p.temporal.WINDOWS[r["window"]][0]//10)
                assert before["decisions"][0]["distance"] == r["decisions"][0]["distance"]
                assert before["decisions"][0]["coefficient"] == -r["decisions"][0]["coefficient"]
                reverse_reads+=1
    for r in donor.values():
        dense(r,protocol,cases[r["init"]])
    if not partial:
        assert len(lookup) == 600 and completed == 40 and replayed == 200
        assert prefix_checks == 560 and background_reads == 240 and reverse_reads == 160
        assert read(base/"queue_COMPLETE.json")["failures"] == []
    meta=dict(actual_episodes=len(lookup),reference_replays=replayed,new_conditions=len(lookup)-replayed,
        independent_starts=20,donor_policy_context_records=40,completed_units=completed,
        verified_native_prefixes=prefix_checks,matched_background_first_reads=background_reads,matched_reverse_first_reads=reverse_reads,
        source_commit=protocol["parent_commit"],protocol_sha256=digest(base/"protocol.json"),
        runtime_sha256=digest(base/"runtime.json"),analysis_plan_sha256=digest(base/"analysis_plan.json"),
        analysis_source_sha256=digest(Path(__file__)),shared_analysis_sha256=digest(Path(previous_analysis.__file__)),
        delta_sha256=delta_sha,weights_fingerprints=fingerprints,max_pair_mean_cast_rms=max(cast,default=None),**provenance)
    return lookup,donor,meta


def group(records):
    pg={}
    for goal in "AB":
        rr=sorted((r for r in records if r["goal"] == goal),key=lambda r:r["init"])
        assert [r["init"] for r in rr] == list(p.IDS)
        target=p.GOALS[goal]
        decisions=[d for r in rr for d in r["decisions"]]
        pg[goal]=dict(n=20,first_correct=sum(r["first"] == [target] for r in rr),
            terminal_correct=sum(r["terminal"][target] for r in rr),ever_correct=sum(r["ever"][target] for r in rr),
            grasped=sum(r["first_grasp"] is not None for r in rr),first=dict(Counter(p.temporal.primary(r) for r in rr)),
            first_read=dict(Counter(r["decisions"][0]["decoded"] for r in rr if r["decisions"])),
            all_blocks_read_correct=sum(bool(r["decisions"]) and all(d["decoded"] == goal for d in r["decisions"]) for r in rr) if decisions else None,
            tied_blocks=sum(d["margin"] == 0 for d in decisions),parent_velocity_queries=sum(d.get("parent_velocity_queries",0) for d in decisions),
            first_block_delta_rms_median=median(r["decisions"][0]["contrast_rms"] for r in rr) if decisions else None,
            first_block_mean_gap_rms_median=median(r["decisions"][0]["mean_gap_rms"] for r in rr) if decisions and all("mean_gap_rms" in r["decisions"][0] for r in rr) else None,
            pre_first=dict(Counter("none" if r["pre"]["first"] == [] else "+".join(r["pre"]["first"]) for r in rr)),
            pre_grasp=sum(r["pre_grasp"] is not None for r in rr))
    lookup={(r["goal"],r["init"]):r for r in records}
    return dict(per_goal=pg,joint_first=sum(all(lookup[g,i]["first"] == [p.GOALS[g]] for g in "AB") for i in p.IDS),
        joint_terminal=sum(all(lookup[g,i]["terminal"][p.GOALS[g]] for g in "AB") for i in p.IDS))


def paired(before,after):
    result={}
    for g in "AB":
        a,b=({r["init"]:r for r in rr if r["goal"] == g} for rr in (before,after))
        assert set(a) == set(b) == set(p.IDS)
        target=p.GOALS[g]
        result[g]=dict(first_gained=[i for i in p.IDS if a[i]["first"] != [target] and b[i]["first"] == [target]],
            first_lost=[i for i in p.IDS if a[i]["first"] == [target] and b[i]["first"] != [target]],
            terminal_gained=[i for i in p.IDS if not a[i]["terminal"][target] and b[i]["terminal"][target]],
            terminal_lost=[i for i in p.IDS if a[i]["terminal"][target] and not b[i]["terminal"][target]],
            first_transitions=dict(Counter(p.temporal.primary(a[i])+"->"+p.temporal.primary(b[i]) for i in p.IDS)))
    a,b=({(r["goal"],r["init"]):r for r in rr} for rr in (before,after))
    first=lambda table,i: all(table[g,i]["first"] == [p.GOALS[g]] for g in "AB")
    terminal=lambda table,i: all(table[g,i]["terminal"][p.GOALS[g]] for g in "AB")
    result["joint"]=dict(first_gained=[i for i in p.IDS if not first(a,i) and first(b,i)],
        first_lost=[i for i in p.IDS if first(a,i) and not first(b,i)],
        terminal_gained=[i for i in p.IDS if not terminal(a,i) and terminal(b,i)],
        terminal_lost=[i for i in p.IDS if terminal(a,i) and not terminal(b,i)])
    return result


def summarize(base):
    lookup,donor,meta=collect(base)
    def select(job):
        return [lookup[*p.job_key(job),g,i] for i in p.IDS for g in "AB"]
    native=select(p.jobs()[0])
    groups=[]
    byjob={}
    for job in p.jobs():
        rr=select(job)
        byjob[p.job_key(job)]=rr
        groups.append(dict(job=job,**group(rr),vs_native=paired(native,rr)))
    background=[]
    v_added=[]
    reversed_v=[]
    beta_added=[]
    for job in p.jobs():
        if job["background"] == "parent" and job["family"] in ("v","own"):
            before={**job,"background":"late"}
            background.append(dict(job=job,paired=paired(select(before),select(job))))
        if job["background"] == "parent" and job["family"] == "v":
            before=next(j for j in p.jobs() if j["family"] == "mu_only" and j["window"] == job["window"])
            v_added.append(dict(job=job,paired=paired(select(before),select(job))))
        if job["family"] == "v" and job["arm"] == "reverse":
            before={**job,"arm":"forward"}
            reversed_v.append(dict(job=job,paired=paired(select(before),select(job))))
        if job["family"] == "own":
            before=native if job["background"] == "late" else select(next(j for j in p.jobs() if j["family"] == "mu_only" and j["window"] == job["window"]))
            beta_added.append(dict(job=job,paired=paired(before,select(job))))
    return dict(**meta,groups=groups,parent_policy=group(list(donor.values())),
        background_pairs=background,v_added_to_parent_mean=v_added,reverse_pairs=reversed_v,beta_added=beta_added)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args()
    if args.partial:
        _,_,meta=collect(args.folder,True)
        print(json.dumps(meta,indent=2))
    else:
        summary=summarize(args.folder)
        path=args.folder/"summary.json"
        if args.check:
            assert read(path) == summary
        else:
            assert not path.exists()
            path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
        print(json.dumps({k:v for k,v in summary.items() if k not in ("groups","parent_policy","background_pairs","v_added_to_parent_mean","reverse_pairs","beta_added")},indent=2))
