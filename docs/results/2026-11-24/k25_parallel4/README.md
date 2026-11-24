# K25 parallel duration and four-command evidence

This archive contains480 new complete400-control evaluations. Part1 is160 full-duration rollouts on the exact four previously evaluated two-old-goal profiles. Part2 is320 rollouts (four arms × four commands × twenty physical starts) on the existing three-old-goal K19 parent continued to original selected K26 Back3000. Training history covers four commands; no new training was run. Ten160-control development prefixes calibrate the four-command gain independently of test behaviour.

## Protocol and method

`analysis_plan.json` was saved before evaluations. `full_protocol.json` reuses the original models, gains and prompts, and records the exact three textual changes to the original episode function: intervention cutoff15 to40 action blocks and matching reporting boundaries. The actual original K22 `combine2('par')` function remains unchanged. Previously completed150->native traces stay in the earlier archive.

`four_protocol.json` fixes three old references (Left, Front, Right) and the original Back3000 continuation. The projector fits the actual F residual onto the parent's centered-old-response span over50x7, using a fixed Helmert contrast basis and float64 SVD (rcond1e-6). The same coefficients reconstruct the full32-dimensional velocity. Allfour real instructions pass through this projector; there is no true-label Back bypass, discrete K26 routing or parent Back reference. ParentP and nativeF baselines each execute allfour instructions. This is a two-model diagnostic, not independent recovery of F.

`four_calibration.json` contains130 norm ratios: dev states0–9, firstEuler input at controls30,40,...150 while nativeF performs a160-control Back prefix. Alpha is the upper median capped10, using centered-old-response Frobenius norms. No test outcomes choose alpha. `history/` preserves original training manifests, completion, READY records and code. Checkpoint files stay on the server.

## Trace schema

- `full_<profile>_<lane>.jsonl.gz`:160 full-duration records; matching completion marker stores original tensor gates, weights and file SHA. Allfirst150 control actions, signatures, states and denoising diagnostics must exactly match the prior traces.
- `four_<lane>.jsonl.gz`:one physical state, allfour arms and allfour commands,16 rollouts per file. Arms are parent, native, parallel150 and parallel400. Every failure is retained. `decisions` has15 blocks forparallel150 and40 forparallel400, each containing10 SVD diagnostics.
- Every record has400-step object/EEF positions,7D clipped actual actions, contact masks, region masks and placement masks;40 block hashes and41 signatures. Sites bit order is left, front, right, back. Contact mask is finger1 +2*finger2. First arrival requires lift>3cm and five consecutive controls in the earliest unique region; final placement is the simulator predicate at control400.
- `summary.json` separates first arrival, success at150, final400, ever and grasp. Counts use twenty matched physical starts. Old-three and allfour joint counts, Back Front->Back passage, and gained/lost start IDs are retained. The mixed old-first + Back-final count is descriptive, alongside strict original endpoints.
- `rank`, `condition`, `coefficients`, `off_axis_fraction`, `orthogonality_error` and `old_axis_cosines` describe each actual projector input. No classification or sign is forced into execution. `actual_query_exact` in four-command gates is true only for the three old canonical reference queries; the Back false flag means that old-reference equality is not applicable, not a failed query. Independent normal-equations and self-reconstruction checks are separately recorded.
- `runtime.json`, `final_inventory.json` bind source, environment, shared dependencies, checkpoint files, normalizers and metadata. Worker before/after fingerprints cover155 originally trained parameters; full checkpoint and partial checkpoint file SHA values are separately verified. No model weights, data assets or caches are committed.

## Offline verification

From repository root, with earlier committed archives present:

```bash
python -B docs/results/2026-11-24/k25_parallel4/summarize_parallel4.py docs/results/2026-11-24/k25_parallel4 --check
python scripts/k25/plot_parallel4.py docs/results/2026-11-24/k25_parallel4/summary.json docs/figures/2026-11-24/k25_parallel4
python docs/results/2026-11-24/k25_parallel4/make_report.py
```

This does not load policy weights or run simulation. Previous150-window traces are `../../2026-11-23/k25_parallel`; shared dependency snapshots remain in `../../2026-11-22/k25_order_restore/dependencies`. `analysis.log` and `analysis_check.log` record full audits. `stage1_tests.log` records21 passing protocol tests; `subspace_tests.log` records5 passing mathematical tests on the server. No extra reverse, V, gain, phrasing or seed experiments were added.

Limitations: the same20 physical starts are reused; one three-old parent and one Back training seed; weak native Back final placement known before this run. Three centered old responses have rank<=2 by construction, which is not a claim about the model's intrinsic representation dimension. Four-command comparisons also change the parent and training history relative to earlier two-old profiles. They do not isolate instruction count as a causal variable or establish universal recovery.
