# K25 whole4: matched full residual versus parallel4

This archive contains 160 NEW complete 400-control evaluations: 20 reused physical starts (800-819), four canonical instructions, and two intervention windows. It reuses 160 frozen parallel150/parallel400 rollouts plus 160 parent/native context rollouts from the sibling `k25_parallel4` archive. These 320 references are not new experiments. No training, checkpoint selection, gain sweep or other repair method was added.

## Matched treatment

P is the K19 three-old replay-1000 parent, reconstructed from R1-24000. F is the original selected K26 Back-3000 continuation. Fixed old references are Left, Front and Right. The actual complete instruction is queried only in F; P only receives the fixed three old references. Back receives the same intervention as the old goals. Both controllers construct the response in float64 and cast the complete 50x32 result back to the native output dtype.

- Full: `muP + alpha * (vF(q) - muF)` on ALL 32 dimensions, including padding.
- Parallel: `muP + alpha * B32 @ pinv(B7) @ (vF(q) - muF)7`, exactly as frozen parallel4. The fit is over 50x7 and lifts the same coefficients to 32 dimensions. Orthogonality is an active-7D statement, not a full-32D Euclidean projection claim.
- `alpha=1.6349165456934696` is reused exactly from parallel4: 130 independent development input norm ratios, upper median capped at 10. It does not depend on the projection operator. No test behaviour or post-projection rescaling selects the gain.
- Window 1-150 returns to native F at control151; window 1-400 uses the intervention throughout. Every rollout runs all 400 controls. There is no true-label bypass or early termination.

`analysis_plan.json` predates the new evaluations. `protocol.json` binds cases, model and normalizer identities, the existing calibration, and executed source. `whole_four.py` compiles the ORIGINAL parallel4 episode source unchanged, replacing only its global `field` callable; metadata is renamed after execution. `whole_residual.py` changes only the response combination. This is K15's full-residual strategy extended to three old references and matched parallel4 precision, not a bitwise rerun of historical binary K15 or its fixed gain2.959.

## Trace and diagnostic schema

Each `whole_<lane>.jsonl.gz` contains eight complete trajectories (four goals x two windows) from one physical start. Its `_COMPLETE.json` binds row SHA, source and protocol SHA, four real-tensor formula gates, and all155 originally trained parameter fingerprints before/after. All20 shards completed without failures; `queue_COMPLETE.json` records episodes160, units20, failures[]. Every failure is retained.

Each trajectory contains all400 object/EEF positions, clipped actual7D actions, contacts, region and simulator-placement masks; 40 action-block hashes; 41 state signatures. Sites bit order is Left, Front, Right, Back. First arrival requires lift above3cm and five consecutive region controls, with an earliest UNIQUE correct target. Placement is the separate simulator predicate; final means the predicate at400, ever means it was true at least once. Therefore transient Back placement need not satisfy the five-step lifted Back passage criterion.

Projection is still computed as a COUNTERFACTUAL diagnostic on the full controller's actual input. Existing `rank`, `projected_norm`, `coefficients`, `scaled_projection_over_parent`, `off_axis_fraction` and `old_axis_cosines` fields describe that counterfactual, not the executed full residual. Added `executed_residual_norm7`, `scaled_full_over_parent`, `scaled_removed_over_parent` and `removed_norm_full32` distinguish what is executed and what projection would remove. Diagnostics on divergent trajectories are not fixed-input treatment comparisons. The first input is shared exactly across methods for all80 goal/start combinations.

`summary.json` preserves separate unique first arrival, placed150, terminal400, ever, grasp, old-three/all-four joint counts, Front->Back lifted-region chronology, and paired gained/lost IDs. `ever150`, its IDs and each first placement step are a post-run phase audit to separate intervention effects from post-handback effects; they do not replace prespecified endpoints or choose gain/starts. Both full-controller durations have bitwise identical first150 prefixes on all80 goal/start pairs. Early Back placement improves0->3 with full residual (IDs805,813,814), but none remains valid at400 in either full arm. Parallel150's eight Back-ever successes all occur AFTER its intervention, during native F. The report retains the eventual opposite trend as well.

`runtime.json` and `final_inventory.json` bind the exact imported code, 28 shared dependencies, package versions, checkpoint files, normalizers and metadata before/after execution (47 final file identities). Models, datasets and caches remain on the server. The original complete run is `/home/lzx/cl_binding/.server/k25_whole4_20261124_01`, alongside the reference `k25_parallel4_20261124_01` run. Existing original environment entry is `/home/lzx/cl_binding/.server/k1_screen_20261101_01/code/env.sh` (source it before changing to the run directory); `$PY` selects the isolated project environment. Logs and the queue source record actual worker commands and GPU assignment.

## Offline verification

From repository root, with earlier committed evidence archives present:

```bash
python -B docs/results/2026-11-24/k25_whole4/summarize_whole4.py docs/results/2026-11-24/k25_whole4 --check
python scripts/k25/plot_whole4.py docs/results/2026-11-24/k25_whole4/summary.json docs/figures/2026-11-24/k25_whole4
python docs/results/2026-11-24/k25_whole4/make_report.py
python -m unittest discover -s tests -p 'test_k25_whole4.py' -v
python -m unittest discover -s tests -p 'test_stage1_protocol.py' -v
```

The audit loads no policy weights and runs no simulation. It verifies the sibling reference artifact manifest, 320 reference trajectories, 160 new complete trajectories, 80 real-tensor gates, unchanged trainable-parameter fingerprints, every actual action hash, initial state signatures and the41-signature record count, full/removed/projected norm identities, 80 shared initial inputs, and 80 exact duration-prefix pairs (including first16 state signatures). Frozen shared dependency files are in `../../2026-11-22/k25_order_restore/dependencies`. `analysis.log` and `analysis_check.log` record the offline audit; `formula_tests.log` records three passing formula checks on the server; `stage1_tests.log` records21 passing protocol checks. The same three formula checks also passed locally.

## Interpretation limits

The experiment isolates retaining/removing an entire response component at fixed parent-mean definition and gain, with full32D padding retained in the full arm. It does not locate individual semantic dimensions, prove a Back-specific or necessary information store, or separate mean mismatch, phase-dependent gain needs and the known weak native Back completion. Early Back and Right benefits coexist with Left/Front losses. Full residual does not solve final Back placement. There is one existing training history and one continuation seed, using the same20 earlier test starts; this is not a fresh physical holdout or universal four-task recovery claim. Both methods use the parent and are two-model control diagnostics.
