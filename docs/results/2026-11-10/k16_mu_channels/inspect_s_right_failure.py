"""Exploratory S-right first-arrival states and HS/HSP padding differences."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def stats(values):
    values = list(values)
    return {"n": len(values), "minimum": min(values), "median": median(values), "maximum": max(values)}


def main():
    records, traces, source_paths = {}, {}, []
    for init in range(20, 40):
        assert json.loads((HERE / f"i{init}_COMPLETE.json").read_text())["n_episodes"] == 16
        for suffix, output in (("", records), ("_traces", traces)):
            path = HERE / f"i{init}{suffix}.jsonl"
            source_paths.append(path)
            for x in map(json.loads, path.read_text().splitlines()):
                key = x["init"], x["condition"], x["goal"]
                assert key not in output
                output[key] = x["rows"] if suffix else x
    assert len(records) == len(traces) == 320
    geometry = json.loads((HERE / "geometry.json").read_text())["sites"]
    site = geometry["right"]
    half = [sum(abs(site["rotation"][i][j]) * site["half_size"][j] for j in range(3)) for i in range(3)]
    top = site["centre"][2] + half[2]
    inside_xy = lambda row: all(abs(row["obj"][i] - site["centre"][i]) <= half[i] for i in (0, 1))

    def state(row):
        return {"step": row["step"], "book_xyz_m": row["obj"], "eef_xyz_m": row["eef"],
                "both_contacts": bool(row["c1"] and row["c2"]),
                "book_center_above_right_region_top_m": row["obj"][2] - top,
                "gripper_sign": 1 if row["a"][6] > 0 else -1 if row["a"][6] < 0 else 0,
                "native_right_success": bool(row["pred"]["right"])}

    cases = []
    for init in range(20, 40):
        row = records[init, "S", "B"]
        if row["first"] != ["right"]:
            continue
        trace = traces[init, "S", "B"]
        boundary, first = row["boundary"], row["first_step"]
        in_xy = [x for x in trace[boundary:] if inside_xy(x)]
        assert in_xy and not row["ever"]["right"]
        cases.append({"init": init, "boundary": boundary, "first_right_step": first,
                      "first_right_start": state(trace[first - 1]),
                      "first_right_run_confirmed": state(trace[first + 3]),
                      "window_end": state(trace[boundary + 59]), "final": state(trace[-1]),
                      "right_xy_steps_after_boundary": len(in_xy),
                      "minimum_book_center_z_while_in_right_xy_m": min(x["obj"][2] for x in in_xy),
                      "minimum_above_region_top_while_in_right_xy_m": min(x["obj"][2] - top for x in in_xy),
                      "all_right_xy_states_center_above_region_top": all(x["obj"][2] > top for x in in_xy),
                      "after_first_steps_with_no_finger_contact": sum(not (x["c1"] or x["c2"]) for x in trace[first - 1:]),
                      "after_first_negative_gripper_steps": sum(x["a"][6] < 0 for x in trace[first - 1:])})
    assert len(cases) == 10
    padding = []
    for init in range(20, 40):
        for goal in "AB":
            a, b = records[init, "HS", goal], records[init, "HSP", goal]
            ta, tb = traces[init, "HS", goal], traces[init, "HSP", goal]
            end = a["boundary"] + 59
            padding.append({"init": init, "goal": goal,
                            "full_trace_equal": ta == tb,
                            "first_targets_equal": a["first"] == b["first"],
                            "ever_flags_equal": a["ever"] == b["ever"],
                            "terminal_flags_equal": a["terminal"] == b["terminal"],
                            "first_step_difference_HSP_minus_HS": None if a["first_step"] is None or b["first_step"] is None else b["first_step"] - a["first_step"],
                            "window_end_book_position_difference_m": math.dist(ta[end]["obj"], tb[end]["obj"]),
                            "final_book_position_difference_m": math.dist(ta[-1]["obj"], tb[-1]["obj"]),
                            "maximum_post_boundary_book_position_difference_m": max(math.dist(x["obj"], y["obj"]) for x, y in zip(ta[a["boundary"]:], tb[b["boundary"]:]))})
    report = {"status": "exploratory descriptive audit, not causal mediation or training-root-cause identification",
              "right_region_center_z_m": site["centre"][2], "right_region_upper_z_m": top,
              "S_right_first_correct_cases": cases,
              "S_right_first_correct_summary": {
                  "n": 10,
                  "at_first_right_both_contacts": sum(x["first_right_start"]["both_contacts"] for x in cases),
                  "at_window_end_both_contacts": sum(x["window_end"]["both_contacts"] for x in cases),
                  "at_final_both_contacts": sum(x["final"]["both_contacts"] for x in cases),
                  "all_right_xy_states_center_above_region_top_cases": sum(x["all_right_xy_states_center_above_region_top"] for x in cases),
                  "first_entry_above_region_top_m": stats(x["first_right_start"]["book_center_above_right_region_top_m"] for x in cases),
                  "per_episode_minimum_above_region_top_while_in_right_xy_m": stats(x["minimum_above_region_top_while_in_right_xy_m"] for x in cases),
              },
              "HS_HSP_padding_comparison": {
                  "n_episodes": 40, "full_trace_equal": sum(x["full_trace_equal"] for x in padding),
                  "first_target_labels_equal": sum(x["first_targets_equal"] for x in padding),
                  "all_goal_ever_flags_equal": sum(x["ever_flags_equal"] for x in padding),
                  "all_goal_terminal_flags_equal": sum(x["terminal_flags_equal"] for x in padding),
                  "window_end_book_position_difference_m": stats(x["window_end_book_position_difference_m"] for x in padding),
                  "final_book_position_difference_m": stats(x["final_book_position_difference_m"] for x in padding),
                  "per_episode_maximum_post_boundary_book_position_difference_m": stats(x["maximum_post_boundary_book_position_difference_m"] for x in padding),
                  "per_episode": padding,
              },
              "limits": ["First region entry requires XY containment and lift, not insertion. A center above the 3D region upper bound cannot satisfy that 3D containment condition.",
                         "Geometric height and finger contacts explain what was observed; they do not isolate which of z/rotation/gripper in S produced the effect.",
                         "Relative pose and contact are post-treatment variables. No mediation or training-update cause has been established.",
                         "Equal HS/HSP endpoint labels do not imply equal trajectories or universal padding irrelevance."],
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [*source_paths, HERE / "geometry.json", Path(__file__)]}}
    out = HERE / "s_right_failure_descriptors.json"
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "S_right_first_correct_summary": report["S_right_first_correct_summary"],
                      "HS_HSP_summary": {k: v for k, v in report["HS_HSP_padding_comparison"].items() if k != "per_episode"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
