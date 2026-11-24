# K25 whole4-new evidence

This reruns the original K25 whole4 comparison with a fresh Back continuation from the SAME three-old K19 parent. It contains400 NEW formal rollouts (native, parallel150, whole150, parallel400, whole400; four goals x20 matched physical starts), plus separate Back development qualification, and reuses80 frozen parent rollouts from `../k25_parallel4`. No new training was launched in this evaluation round. The earlier Back-only24000-step training has finished; training completion alone is not qualification.

`plan.json` was saved before evaluations. `preparation.json` binds all13 saved candidate weights, reference sources, test/dev cases and source hashes. Native Back development uses20 fresh reset states labelled640-659, physically distinct from the frozen800-819 formal states. Selection checks saved checkpoints latest-to-earliest and stops at the first >=16/20 final and ever successes. If none passes, the terminal24000-step model is evaluated as an explicitly UNQUALIFIED diagnostic because the user requested the new-model comparison. All failures remain in the denominator. `selection.json` records the outcome and complete checked development scores; no repair outcomes select a model.

`calibration.json` uses the exact original rule: native selectedF Back prefixes on independent development IDs0-9, first Euler input at30,40,...150,130 centered-old-response norm ratios, upper median capped10. Gain is recalibrated for the NEW F, then fixed identically across whole and parallel. There is no post-projection renormalization or gain sweep. Cross-model differences also change training budget, sampling, seed and learning schedule, and cannot isolate one training change.

The frozen original parallel4 episode source executes unchanged. Whole replaces only its field callback with parent mean + alpha times the FULL actualF residual, all32 dimensions. Parallel fits active50x7 onto the parent three-old span using SVD rcond1e-6 and lifts to32 using the same coefficients. Both construct in float64 and cast to native dtype. Actual complete text goes only toF; P queries only the three fixed old references. No Back label bypass is used. Window1-150 returns to nativeF at151; window1-400 intervenes throughout; every rollout runs400 steps.

`new_<lane>.jsonl.gz` holds20 complete rollouts from one frozen test start; its COMPLETE marker binds hashes,155-parameter before/after fingerprints and four gates for each controller. Traces retain all object/EEF positions, physical clipped actions, contacts, region/placement masks,40 action-block hashes and41 state signatures. First arrival requires a lifted unique five-control region run; placement uses the separate simulator predicate. Ever, placement150, final400, Front->Back chronology, grasp, paired gained/lost IDs, old-three/all-four joint counts and final50-control placement fractions are separately retained in summary. Projection diagnostics in WHOLE logs are counterfactual values on the whole controller's input, not the executed full residual; four extra fields describe the executed full and removed norms.

`dev_<step>_<lane>` files retain every checked qualification rollout and marker. `history/` binds the completed training, all13 checkpoint READY identities, and compressed full24000-step training log. Independent training audit verifies every sample index, learning rate, finite loss/gradient and source identity. No model weights, optimizer state, dataset, cache or private teammate material is committed.

`start_identity.json` and `final_identity.json` verify the same60 model/normalizer/source/dependency files and package versions before/after the evaluation phase, including28 shared dependencies. Formula gates, unchanged trainable fingerprints and original reference artifact manifests provide separate checks. Reused parent trajectories are audited against their original artifact manifest and rescored on the same cases. The first150 prefixes are bitwise equal for both duration variants of BOTH methods (160 goal/start pairs); original first-input projection diagnostics match between whole and parallel in both windows (160 pairs). Source/run folder on server: `/home/lzx/cl_binding/.server/k25_whole4_new_20261124_01`.

## Offline verification

From repository root, with earlier committed evidence archives present:

```bash
python -B docs/results/2026-11-24/k25_whole4_new/summarize_whole4_new.py docs/results/2026-11-24/k25_whole4_new --check
python scripts/k25/plot_whole4_new.py docs/results/2026-11-24/k25_whole4_new/summary.json docs/figures/2026-11-24/k25_whole4_new
python docs/results/2026-11-24/k25_whole4_new/make_report.py
```

The offline audit runs no inference or simulation. `analysis.log` and `analysis_check.log` record the outcome. `formula_tests.log` contains3 passing existing full-residual checks and `stage1_tests.log` contains21 passing protocol checks. This remains one training history/seed, one new training configuration, reused20 formal starts and a parent-dependent two-model diagnostic. Qualification failure is never relabelled as a strong Back baseline.
