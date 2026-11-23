"""Audit all anchor qualifications without equating a competent donor with V repair."""

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

import anchor_protocol as p
import anchor_recovery as recovery
import summarize_v_timing as timing_audit

same_prefix = timing_audit.same_prefix


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def sig(obj):
    return sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def sibling(base, short, remote):
    return next(x for x in (base.parent.parent/"2026-11-22"/short,base.parent/remote) if (x/"protocol.json").exists())


def rows(path, partial=False):
    try:
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            while True:
                try:
                    line = stream.readline()
                except EOFError:
                    if partial:
                        return
                    raise
                if not line:
                    return
                yield json.loads(line)
    except EOFError:
        if not partial:
            raise


def validate(r,protocol,case):
    family,a,gain,window,arm,goal,init = p.key(r)
    target = p.GOALS[r["line"]][goal]
    assert r["target"] == target and r["seed"] == case["seed"]
    assert r["text"] == f"pick up the book and place it in the {target} compartment of the caddy"
    assert r["offsets"] == list(p.offsets(r))
    result = p.temporal.score(r["trace"],protocol["geometry"],case["z0"])
    assert all(r[k] == v for k,v in result.items())
    interval = p.temporal.interval_metrics(r["trace"],protocol["geometry"],case["z0"],*p.offsets(r))
    for k,v in interval.items():
        if k == "first_10_approach_mm":
            assert all(math.isclose(r[k][g],x,rel_tol=1e-10,abs_tol=1e-7) for g,x in v.items())
        else:
            assert r[k] == v
    assert len(r["signatures"]) == 41 and r["signatures"][0] == sig(case["signature"])
    assert len(r["action_hashes"]) == 40
    for k,expected in enumerate(r["action_hashes"]):
        flat = [x for action in r["trace"]["actions"][k*10:k*10+10] for x in action]
        assert sha256(struct.pack("<70f",*flat)).hexdigest() == expected
    decisions = r["decisions"]
    active = p.temporal.active_blocks(window) if family == "delta" else []
    assert [d["block"] for d in decisions] == active
    assert dict(Counter(d["decoded"] for d in decisions)) == r["routed"]
    for d in decisions:
        assert d["decoded"] == min("ABN",key=d["distance"].__getitem__)
        assert d["distance"][goal] == 0.
        assert all(math.isfinite(x) and x >= 0 for x in d["distance"].values())
        assert math.isfinite(d["contrast_rms"]) and d["contrast_rms"] >= 0
        assert d["coefficient"] == gain*({"A":1.,"B":-1.,"N":0.}[d["decoded"]])*(1 if arm == "forward" else -1)
    assert family in ("policy","v_weights","delta") and a in p.ANCHORS


def load_references(base,protocol,cases):
    old = sibling(base,"k25_v_timing","k25_v_timing_20261122_01")
    original = sibling(base,"k25_order_restore","k25_order_restore_20261122_01")
    assert digest(old/"protocol.json") == protocol["timing_protocol_sha256"]
    old_protocol = read(old/"protocol.json")
    assert digest(Path(timing_audit.__file__)) == read(old/"summary.json")["analysis_source_sha256"]
    assert cases == read(old/"cases.json") == read(original/"cases.json")["test"]
    assert digest(original/"protocol.json") == old_protocol["previous_protocol_sha256"]
    archived = original/"dependencies/cl_binding/.server"
    assert digest(archived/"k25_order_restore_20261122_01/order_restore.py") == protocol["base_source_sha256"]
    assert digest(archived/"k25_final_20261121_01/self_restore.py") == protocol["baseline_source_sha256"]
    for name,expected in protocol["validated_dependency_sha256"].items():
        assert digest(original/"dependencies"/name) == expected
    reused,late,parent = {},{},{}
    old_delta = {}
    for line in p.LINES:
        for lane in range(8):
            marker = read(old/f"{line}_{lane}_COMPLETE.json")
            file = old/f"{line}_{lane}.jsonl.gz"
            assert digest(file) == marker["rows_sha256"]
            if line in old_delta:
                assert old_delta[line] == marker["delta_sha256"]
            old_delta[line] = marker["delta_sha256"]
            for row in rows(file):
                if row["window"] == "native":
                    late[line,row["goal"],row["init"]] = row
                elif row["window"] in p.WINDOWS:
                    r = dict(row,family="delta",anchor=30,origin="imported_k25_v_timing")
                    reused[line,*p.key(r)] = r
            source = original/f"main_{line}_{lane}.jsonl"
            reference_marker = read(original/f"main_{line}_{lane}_COMPLETE.json")
            assert digest(source) == reference_marker["rows_sha256"]
            assert reference_marker["protocol_sha256"] == old_protocol["previous_protocol_sha256"]
            for text in source.read_text(encoding="utf-8").splitlines():
                r = json.loads(text)
                if r["goal"] == "N" and r["variant"] == "canon" and r["arm"] == "native":
                    late[line,"N",r["init"]] = r
        if line == "lf_right_1000":
            for lane in range(8):
                file = original/f"parent_{lane}.jsonl"
                reference_marker = read(original/f"parent_{lane}_COMPLETE.json")
                assert digest(file) == reference_marker["rows_sha256"]
                assert reference_marker["base_protocol_sha256"] == old_protocol["previous_protocol_sha256"]
                for text in file.read_text(encoding="utf-8").splitlines():
                    r = json.loads(text)
                    parent[line,r["goal"],r["init"]] = r
    assert len(reused) == 320 and len(late) == 120 and len(parent) == 40
    return reused,late,parent,old_delta


def load_recovery(base,protocol):
    """Bind every retained row to the interrupted file, without selecting outcomes."""
    manifest = read(base/"recovery_manifest.json")
    assert manifest["recovery_source_sha256"] == protocol["recovery_source_sha256"]
    original = next(x for x in (base/"interrupted",Path(manifest["original_run"]))
                    if (x/"protocol.json").exists())
    assert digest(original/"protocol.json") == manifest["original_protocol_sha256"]
    old = read(original/"protocol.json")
    plan = read(base/"analysis_plan.json")
    assert plan["locked_before_queue_launch"] and digest(original/"analysis_plan.json") == digest(base/"analysis_plan.json")
    assert datetime.fromisoformat(old["created_at_utc"]) < datetime.fromisoformat(plan["created_at_utc"]) < datetime.fromisoformat(read(original/"runtime.json")["captured_at_utc"])
    assert digest(original/"anchor_check.py") == old["source_sha256"] == manifest["original_source_sha256"]
    assert digest(original/"anchor_protocol.py") == old["spec_sha256"] == protocol["spec_sha256"]
    for k in ("lines","ids","goals","anchors","gains","windows","arms","jobs","all_jobs",
              "expected_new","expected_total","checkpoints","cases_sha256","timing_protocol_sha256",
              "timing_source_sha256","timing_spec_sha256","base_source_sha256","baseline_source_sha256",
              "geometry","versions","frozen_reconstruction","validated_dependency_sha256"):
        assert old[k] == protocol[k],k
    cached = {}
    for name,expected in manifest["files"].items():
        assert Path(name).name == name and digest(original/name) == expected["sha256"]
        actual = {}
        for row,h in recovery.decoded_rows(original/name):
            k = recovery.key_text(p.key(row))
            assert k not in actual
            actual[k] = h
        assert actual == expected["row_sha256"] and len(actual) == expected["rows"]
        cached[name] = expected
    assert sum(x["rows"] for x in cached.values()) == manifest["verified_complete_original_rows"] == 804
    assert manifest["missing_grid_episodes"] == p.EXPECTED_NEW-804 == 556
    return manifest,cached


def collect(base,partial=False):
    base = base.resolve()
    protocol = read(base/"protocol.json")
    assert protocol["source_sha256"] == digest(Path(__file__).with_name("anchor_check.py"))
    assert protocol["spec_sha256"] == digest(Path(p.__file__))
    assert protocol["timing_source_sha256"] == digest(Path(__file__).with_name("v_timing.py"))
    assert protocol["timing_spec_sha256"] == digest(Path(p.temporal.__file__))
    assert protocol["tensor_hash_source_sha256"] == digest(Path(__file__).with_name("anchor_tensor_hash.py"))
    assert protocol["recovery_source_sha256"] == digest(Path(recovery.__file__))
    assert protocol["jobs"] == p.jobs() and protocol["all_jobs"] == p.jobs(True)
    assert protocol["expected_new"] == p.EXPECTED_NEW and protocol["expected_total"] == p.EXPECTED_TOTAL
    assert protocol["cases_sha256"] == digest(base/"cases.json")
    runtime = read(base/"runtime.json")
    assert runtime["capture_source_sha256"] == digest(base/"capture_runtime.py")
    assert runtime["packages"] == protocol["versions"]
    assert runtime["dependency_sha256"] == protocol["validated_dependency_sha256"]
    assert runtime["recovery_manifest_sha256"] == digest(base/"recovery_manifest.json")
    assert sorted(runtime["extra_actual_modules"].values()) == sorted([protocol["timing_source_sha256"],protocol["timing_spec_sha256"]])
    init = read(base/"initialization_audit.json")
    assert init["both_parent0_are_recorded_continuation_initializations"]
    for name,expected in init["manifests"].items():
        assert digest(base/"history"/name) == expected
    for name,record in read(base/"history/sources.json").items():
        assert Path(name).name == name and digest(base/"history"/name) == record["sha256"]
        assert (base/"history"/name).stat().st_size == record["bytes"]
    lr,lf = (read(base/"history"/name) for name in ("LR_initialization_manifest.json","LF_initialization_manifest.json"))
    assert lr["initialization_model_sha256"] == protocol["checkpoints"]["lr_front_1000:0"]["sha256"]
    assert lf["parent_sha256"] == protocol["checkpoints"]["lf_right_1000:0"]["sha256"]
    assert lr["frozen_hash"] == lf["frozen_hash"] == init["frozen_hash"]
    lrc,lfr = (read(base/"history"/name) for name in ("LR_step30_COMPLETE.json","LF_step30_READY.json"))
    assert lrc["step"] == lfr["step"] == 30 and lrc["status"] == "pass" and lrc["reload_max_difference"] == 0
    assert lrc["frozen_hash_unchanged"] and lfr["sha256"] == protocol["checkpoints"]["lf_right_1000:30"]["sha256"]
    bank = read(base/"cases.json")
    assert [r["init"] for r in bank] == list(p.IDS)
    assert len({r["signature"]["sim"] for r in bank}) == 20
    cases = {r["init"]:r for r in bank}
    inventory = read(base/"weights_inventory.json")
    assert inventory["source_sha256"] == digest(Path(__file__).with_name("anchor_inventory.py"))
    for line in p.LINES:
        inv = inventory["lines"][line]
        assert len(inv["tensors"]) == 8 and -1 <= inv["cosine_delta0_delta30"] <= 1
    imported,late,parent,old_delta = load_references(base,protocol,bank)
    recovery_manifest,cache_files = load_recovery(base,protocol)
    lookup = dict(imported)
    fullhashes,deltas,gates = {},{},[]
    new = parent_replays = probes = completed = 0
    cached_rows = executed_rows = technical_replays = 0
    for line in p.LINES:
        for lane in range(8):
            file = base/f"{line}_{lane}.jsonl.gz"
            markfile = base/f"{line}_{lane}_COMPLETE.json"
            if partial and not file.exists():
                continue
            ids = list(p.IDS)[lane::8]
            expected = p.expected_keys(ids)
            found = set()
            retained = cache_files.get(file.name,dict(rows=0,row_sha256={},sha256=None))
            for r in rows(file,partial):
                k = p.key(r)
                assert r["line"] == line and k in expected and k not in found
                assert r["origin"] == "new_episode"
                rk = recovery.key_text(k)
                if rk in retained["row_sha256"]:
                    assert sha256(recovery.row_text(r).encode()).hexdigest() == retained["row_sha256"][rk]
                validate(r,protocol,cases[r["init"]])
                found.add(k)
                lookup[line,*k] = r
                new += 1
                if r["family"] == "policy" and r["anchor"] == 0 and (line,r["goal"],r["init"]) in parent:
                    old = parent[line,r["goal"],r["init"]]
                    assert all(r[x] == old[x] for x in ("text","first","first_step","times","terminal","first_grasp"))
                    assert r["signatures"][8] == sig(old["boundary_signature"])
                    parent_replays += 1
            if partial and not markfile.exists():
                continue
            m = read(markfile)
            assert found == expected and m["episodes"] == len(expected)
            assert m["ids"] == ids and m["line"] == line and m["lane"] == lane and m["lanes"] == 8
            assert m["rows_sha256"] == digest(file) and m["protocol_sha256"] == digest(base/"protocol.json")
            assert m["source_sha256"] == protocol["source_sha256"] and m["spec_sha256"] == protocol["spec_sha256"]
            assert m["cached_original_episodes"] == retained["rows"]
            assert m["cached_row_sha256"] == retained["row_sha256"]
            assert m["original_partial_file_sha256"] == retained["sha256"]
            assert m["executed_episodes"] == len(expected)-retained["rows"]
            should_replay = any(json.loads(k)[0] == "delta" for k in retained["row_sha256"])
            assert m["engineering_replay_episodes"] == int(should_replay)
            assert m["replay_bitwise_equal"] == should_replay
            if should_replay:
                assert m["replay_key"][0] == "delta" and recovery.key_text(m["replay_key"]) in retained["row_sha256"]
            else:
                assert m["replay_key"] is None
            cached_rows += m["cached_original_episodes"]
            executed_rows += m["executed_episodes"]
            technical_replays += m["engineering_replay_episodes"]
            assert m["delta_sha256"]["30"] == old_delta[line]
            assert m["delta_sha256"] == inventory["lines"][line]["delta_sha256"]
            for k in found:
                r = lookup[line,*k]
                assert r["delta_sha256"] == (None if r["family"] == "policy" else m["delta_sha256"][str(r["anchor"])])
            if line in fullhashes:
                assert fullhashes[line] == m["weights_fingerprints"] and deltas[line] == m["delta_sha256"]
            fullhashes[line],deltas[line] = m["weights_fingerprints"],m["delta_sha256"]
            assert len(m["gates"]) == 4 and {(g["anchor"],g["gain"]) for g in m["gates"]} == {(a,g) for a in p.ANCHORS for g in p.GAINS}
            for g in m["gates"]:
                assert g["single_model_roles"] == ["F"]
                assert all(g[k] for k in ("v_only_overlay_equal","weights_restored","mean_preserved_fp64","new_goal_bypass_bitwise"))
                assert math.isfinite(g["mean_cast_rms"]) and g["mean_cast_rms"] >= 0
            gates.extend(m["gates"])
            assert m["previous_parent_replays"] == (2*len(ids) if line == "lf_right_1000" else 0)
            assert m["imported_condition_first_block_probes"] == 4*len(ids)
            probes += m["imported_condition_first_block_probes"]
            completed += 1
    if not partial:
        assert new == 1360 and len(lookup) == 1680 and parent_replays == 40 and probes == 160 and completed == 16
        assert cached_rows == 804 and executed_rows == 556 and technical_replays == 8
        assert read(base/"queue_COMPLETE.json")["failures"] == []
    # Imported controls remain unchanged, but their dense traces are also rescored.
    for r in imported.values():
        validate(r,protocol,cases[r["init"]])
        assert r["delta_sha256"] == old_delta[r["line"]]
    prefix_checks = polarity = 0
    for (line,*_),r in lookup.items():
        if r["family"] != "delta":
            continue
        before = late[line,r["goal"],r["init"]]
        same_prefix(before,r,p.offsets(r)[0]//10)
        prefix_checks += 1
        if r["arm"] == "reverse":
            fkey = (line,r["family"],r["anchor"],r["gain"],r["window"],"forward",r["goal"],r["init"])
            if fkey not in lookup:
                assert partial
                continue
            pos = lookup[fkey]
            same_prefix(pos,r,p.offsets(r)[0]//10)
            assert pos["decisions"][0]["distance"] == r["decisions"][0]["distance"]
            assert pos["decisions"][0]["coefficient"] == -r["decisions"][0]["coefficient"]
            polarity += 1
    meta = dict(new_episodes=new,imported_episodes=320,late_native_references=120,
        primary_grid_episodes=new,cached_original_episodes=cached_rows,executed_recovery_grid_episodes=executed_rows,
        technical_replay_episodes=technical_replays,physical_complete_executions=cached_rows+executed_rows+technical_replays,
        recovery_manifest_sha256=digest(base/"recovery_manifest.json"),
        original_interrupted_protocol_sha256=recovery_manifest["original_protocol_sha256"],
        completed_lanes=completed,previous_parent_replays=parent_replays,imported_first_block_probes=probes,
        verified_native_prefixes=prefix_checks,verified_polarity_pairs=polarity,
        protocol_sha256=digest(base/"protocol.json"),analysis_plan_sha256=digest(base/"analysis_plan.json"),
        analysis_source_sha256=digest(Path(__file__)),delta_sha256=deltas,weights_fingerprints=fullhashes,
        inventory_sha256=digest(base/"weights_inventory.json"),
        initialization_audit_sha256=digest(base/"initialization_audit.json"),runtime_sha256=digest(base/"runtime.json"),
        max_mean_cast_rms=max((g["mean_cast_rms"] for g in gates),default=None),independent_starts=20)
    return lookup,late,meta


def group(records,line,goals="AB"):
    pg = {}
    for goal in goals:
        rr = sorted((r for r in records if r["goal"] == goal),key=lambda r:r["init"])
        assert [r["init"] for r in rr] == list(p.IDS)
        target = p.GOALS[line][goal]
        decisions = [d for r in rr for d in r["decisions"]]
        pg[goal] = dict(n=20,first_correct=sum(r["first"] == [target] for r in rr),
            terminal_correct=sum(r["terminal"][target] for r in rr),ever_correct=sum(r["ever"][target] for r in rr),
            grasped=sum(r["first_grasp"] is not None for r in rr),first=dict(Counter(p.temporal.primary(r) for r in rr)),
            first_read=dict(Counter(r["decisions"][0]["decoded"] for r in rr if r["decisions"])),
            all_blocks_read_correct=sum(bool(r["decisions"]) and all(d["decoded"] == goal for d in r["decisions"]) for r in rr) if decisions else None,
            tied_blocks=sum(d["margin"] == 0 for d in decisions),
            first_block_delta_rms_median=median(r["decisions"][0]["contrast_rms"] for r in rr) if decisions else None)
    table = {(r["goal"],r["init"]):r for r in records}
    return dict(per_goal=pg,
        joint_first=sum(all(table[g,i]["first"] == [p.GOALS[line][g]] for g in "AB") for i in p.IDS),
        joint_terminal=sum(all(table[g,i]["terminal"][p.GOALS[line][g]] for g in "AB") for i in p.IDS))


def paired(before,after,line,goal):
    a,b = ({r["init"]:r for r in rr if r["goal"] == goal} for rr in (before,after))
    assert set(a) == set(b) == set(p.IDS)
    target = p.GOALS[line][goal]
    def ids(fn):
        return [i for i in p.IDS if fn(a[i],b[i])]
    return dict(first_gained=ids(lambda x,y:x["first"] != [target] and y["first"] == [target]),
        first_lost=ids(lambda x,y:x["first"] == [target] and y["first"] != [target]),
        terminal_gained=ids(lambda x,y:not x["terminal"][target] and y["terminal"][target]),
        terminal_lost=ids(lambda x,y:x["terminal"][target] and not y["terminal"][target]),
        first_transitions=dict(Counter(f"{p.temporal.primary(a[i])}->{p.temporal.primary(b[i])}" for i in p.IDS)))


def summarize(base):
    lookup,late,meta = collect(base)
    result = dict(**meta,lines={})
    for line in p.LINES:
        def select(family,anchor,gain,window,arm):
            return [lookup[line,family,anchor,gain,window,arm,g,i] for i in p.IDS for g in ("ABN" if family == "policy" else "AB")]
        native = [late[line,g,i] for i in p.IDS for g in "ABN"]
        item = dict(late_native=group(native,line,"ABN"),policies={},v_weights={},delta={},anchor0_vs_30={},
            b_improvement_without_aggregate_harm=[],improved_all_degraded_endpoints=[],reached_parent_old_counts=[])
        policies = {a:select("policy",a,0.,"native","native") for a in p.ANCHORS}
        for a in p.ANCHORS:
            item["policies"][str(a)] = group(policies[a],line,"ABN")
            vr = select("v_weights",a,1.,"full_400","native")
            item["v_weights"][str(a)] = dict(**group(vr,line),vs_late={g:paired(native,vr,line,g) for g in "AB"})
            item["delta"][str(a)] = {}
            for gain in p.GAINS:
                by_window = {}
                for window in p.WINDOWS:
                    arms = {}
                    positive = select("delta",a,gain,window,"forward")
                    for arm in p.ARMS:
                        rr = select("delta",a,gain,window,arm)
                        arms[arm] = dict(**group(rr,line),vs_late={g:paired(native,rr,line,g) for g in "AB"})
                        if arm == "reverse":
                            arms[arm]["vs_forward"] = {g:paired(positive,rr,line,g) for g in "AB"}
                    by_window[window] = arms
                    pos = arms["forward"]
                    old = item["late_native"]["per_goal"]
                    score = pos["per_goal"]
                    preserves = all(score[g][k] >= old[g][k] for g in "AB" for k in ("first_correct","terminal_correct"))
                    improves_both = all(score["B"][k] > old["B"][k] for k in ("first_correct","terminal_correct"))
                    candidate = dict(anchor=a,gain=gain,window=window,
                        paired_A_first_losses=pos["vs_late"]["A"]["first_lost"],paired_A_terminal_losses=pos["vs_late"]["A"]["terminal_lost"])
                    if preserves and improves_both:
                        item["b_improvement_without_aggregate_harm"].append(candidate)
                    parent_score = item["policies"]["0"]["per_goal"]
                    if all(score[g][k] > old[g][k] if parent_score[g][k] > old[g][k] else score[g][k] >= old[g][k]
                           for g in "AB" for k in ("first_correct","terminal_correct")):
                        item["improved_all_degraded_endpoints"].append(candidate)
                    if all(score[g][k] >= parent_score[g][k] for g in "AB" for k in ("first_correct","terminal_correct")):
                        item["reached_parent_old_counts"].append(candidate)
                item["delta"][str(a)][str(gain)] = by_window
            d1 = select("delta",a,1.,"full_400","forward")
            item["v_weights"][str(a)]["vs_delta1"] = {g:paired(d1,vr,line,g) for g in "AB"}
        item["policy30_vs_parent0"] = {g:paired(policies[0],policies[30],line,g) for g in "ABN"}
        for gain in p.GAINS:
            out = {}
            for window in p.WINDOWS:
                out[window] = {arm:{g:paired(select("delta",30,gain,window,arm),select("delta",0,gain,window,arm),line,g) for g in "AB"} for arm in p.ARMS}
            item["anchor0_vs_30"][str(gain)] = out
        result["lines"][line] = item
    for category in ("b_improvement_without_aggregate_harm","improved_all_degraded_endpoints","reached_parent_old_counts"):
        sets = [{(r["anchor"],r["gain"],r["window"]) for r in result["lines"][line][category]} for line in p.LINES]
        result["shared_"+category] = [dict(anchor=a,gain=g,window=w) for a,g,w in sorted(set.intersection(*sets))]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder",type=Path)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--partial",action="store_true")
    args = parser.parse_args()
    if args.partial:
        lookup,late,meta = collect(args.folder,True)
        print(json.dumps(meta,indent=2))
        counts = Counter()
        for (line,*_),r in lookup.items():
            if r["family"] == "policy":
                k = (line,r["anchor"],r["goal"])
                counts[k,"n"] += 1
                counts[k,"first"] += r["first"] == [r["target"]]
                counts[k,"terminal"] += r["terminal"][r["target"]]
        for key in sorted({k for k,metric in counts}):
            print("POLICY",key,{m:counts[key,m] for m in ("n","first","terminal")})
    else:
        result = summarize(args.folder)
        dest = args.folder/"summary.json"
        if args.check:
            assert read(dest) == result
        else:
            assert not dest.exists()
            dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
        print(json.dumps({k:v for k,v in result.items() if k != "lines"},indent=2))
