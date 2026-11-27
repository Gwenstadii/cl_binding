"""Read-only reanalysis of K14 published per-snapshot/per-state judgement records."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def transitions(before, after, keys):
    gained = [k for k in keys if not before[k] and after[k]]
    lost = [k for k in keys if before[k] and not after[k]]
    return {"n":len(keys), "before_success":sum(before[k] for k in keys),
            "after_success":sum(after[k] for k in keys), "gained":gained, "lost":lost,
            "changed":len(gained)+len(lost), "net":len(gained)-len(lost)}


def main(source, output):
    b1=read(source/"judgement/k14_judgement.json")
    b2=read(source/"judgement/k14_b2_judgement.json")
    b3=read(source/"judgement/k14_b3_judgement.json")
    selected=read(source/"b3_state_selection.json")
    inits=b1["inits"]
    assert inits==list(range(20,40))
    all_keys=[f"{i}{g}" for g in "AB" for i in inits]
    def group(name):
        p=b2["per_snapshot"][name]
        return {f"{i}{g}":p[f"{side}_term"][j]
                for g,side in (("A","L"),("B","R")) for j,i in enumerate(inits)}
    I,J,L=map(group,("I_cand60","J_cand400","L_Rfull"))
    t0,t5=(b3["per_state"][g] for g in ("T0_cand","T5_full"))
    eligible=sorted(t0)
    excluded=sorted(set(all_keys)-set(eligible))
    assert len(eligible)==26 and len(excluded)==14 and set(t0)==set(t5)
    assert all(J[k]==t0[k] for k in eligible)
    assert {f"{x['init']}{x['goal']}" for x in selected["eligible"]}==set(eligible)
    assert {f"{x['init']}{x['goal']}" for x in selected["excluded"]}==set(excluded)
    late=transitions(t0,t5,eligible)
    target=b3["comparisons"]["T5_full>T0_cand:term"]
    assert len(late["gained"])==target["gained"]==3 and len(late["lost"])==target["lost"]==2
    late["clustered_ci"]=target["ci"]
    duration=transitions(I,J,all_keys)
    assert len(duration["gained"])==2 and len(duration["lost"])==1
    earlier=transitions(J,L,all_keys)
    assert len(earlier["gained"])==9 and len(earlier["lost"])==0
    decomposition={"all_40":earlier, "eligible_26":transitions(J,L,eligible),
                   "excluded_14":transitions(J,L,excluded),
                   "early_advantage_rescued_by_late_switch":sorted(set(earlier["gained"]) & set(late["gained"]))}
    cosine_example={}
    for c in (.75,.77):
        cosine_example[str(c)]={"angle_degrees":math.degrees(math.acos(c)),
            "perpendicular_norm_over_old_norm":math.sqrt(1-c*c),
            "perpendicular_squared_norm_fraction":1-c*c,
            "equal_norm_old_minus_candidate_norm_over_old":math.sqrt(2-2*c)}
    sources=[source/"source_report.md",source/"b3_state_selection.json",*sorted((source/"judgement").glob("*.json")),
             *sorted((source/"code").glob("*.py")),*sorted((source/"prereg").glob("*.json"))]
    result={"scope":"review of report, code and judgement JSON; no new rollouts or independent full-trace audit",
            "late_T0_to_T5":late, "duration_I_to_J":duration,
            "L_vs_J_gap_by_third_batch_selection":decomposition,
            "excluded_reasons":selected["excluded"],
            "single_vector_geometry_examples_not_empirical_information_fractions":cosine_example,
            "padding_ratio_diagnostic":{g:b1["diagnostics"][g]["pad_over_real_diff"] for g in ("C_amp","F_old")},
            "source_sha256":{str(p.relative_to(source)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    output.mkdir(parents=True,exist_ok=True)
    (output/"review_analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("late_T0_to_T5","duration_I_to_J","L_vs_J_gap_by_third_batch_selection")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.source.resolve(),a.output.resolve())
