# K25 parallel projection reproduction — archived evidence

Only one method was evaluated: original K22 parent mean + parallel projection of the continued model's actual-query residual onto the matched parent's difference axis. Four existing model profiles, twenty reused physical starts (800–819), two canonical old instructions: 160 complete 400-control rollouts. Intervention is controls 1–150; controls 151–400 are native F. No V, sign-only, orthogonal-component, gain-sweep or reverse-sign rollouts were added.

Independent calibration adds twenty 160-control development prefixes (not evaluation samples), ten per1000 profile. The two12000 gains are original values. Check `analysis_plan.json`, `preparation.json`, both `calibration_*.json` and `protocol.json`.

## Files and data fields

- `*_0` through `*_7.jsonl.gz` and corresponding `_COMPLETE.json`: all160 evaluation traces, actions, endpoints and per-denoising diagnostics, with32 completion markers. Each episode stores400 steps,40 action-block hashes and41 state signatures.
- `trace.regions`: bit mask of lifted geometric occupancy; `trace.success`: LIBERO placement success mask; bits follow left, front, right, back. `contacts` uses finger1 +2*finger2. Final placement is step400 only. First arrival requires five consecutive controls at a unique earliest target.
- `decisions[].denoising`: fifteen intervention blocks × ten denoising inputs. `c1` is the unscaled signed projection coefficient, `coefficient` is alpha*c1, `cos` is actual residual vs parent axis. B has negative expected sign; that sign is used only for analysis, never to force execution. `late_norm`, `parent_norm`, `mean_gap_norm`, `residual_norm` use50x7. `perp_over_old` is the scaled off-axis norm relative to the old norm. Off-axis does not imply harmful noise.
- `history/k21.py`, `k22.py`, `k22p3.py`, `original_lf_alpha.json`: exact server originals. Execution calls the original `Lab2.combine2('par')`; coefficient is50x7, output isfull32.
- `runtime.json`, `final_inventory.json`: environment, dependency hashes and checkpoint identities before/after the run. No model, simulator asset or environment cache is committed. Trainable-weight before/after fingerprints cover155 named parameters, separately from full checkpoint file hashes.
- `source_amendment.json`, `technical_attempt_01/`: the independent tensor-check formula initially grouped float32 multiplication/division differently from the original. No completed shard existed. The only edit changes expected-value arithmetic grouping; all interruption logs and readable prefixes are retained. Actual method, calibration rule, alpha and intervention window remain unchanged. Calibration source hash belongs to the original preparation source; the exact one-location patch is audited.
- `summary.json`: independent full-trace rescoring of all160 records, original implementation gates, gains and descriptive diagnostics, plus references reused from the frozen parent0 qualification and1000 native archives. References are not new rollouts.
- `formula_tests.log`: four server formula tests passed. `validation.json` records21 stage1 protocol tests and local/archived verification. `analysis.log`, `analysis_check.log`: full offline audits.

## Offline reproduction of the summary

Run from repository root after obtaining this commit and the already tracked previous archives:

```bash
python -B docs/results/2026-11-23/k25_parallel/summarize_parallel.py docs/results/2026-11-23/k25_parallel --check
python scripts/k25/plot_parallel.py docs/results/2026-11-23/k25_parallel/summary.json docs/figures/2026-11-23/k25_parallel/parallel_results
python docs/results/2026-11-23/k25_parallel/make_report.py
```

Dependency snapshots referenced by SHA remain in `docs/results/2026-11-22/k25_order_restore/dependencies`; parent qualification is `../k25_anchor`, and original1000 native traces are in `../../2026-11-22/k25_v_timing`. Nothing is downloaded or simulated by offline rescoring.

This is a two-model diagnostic: P supplies the mean and old control axis, F supplies a scalar. There is one matched parent per training order. The existing1000 streams and original12000 endpoints are different branches, so their difference is not a controlled training-depth effect. These twenty starts were reused; no fresh holdout, independent seed, new-task retention or paraphrase claims are supported. No optimal gain or causal benefit of removing an orthogonal component is established by this one-method reproduction. The teammate's private report remains local and is not included in the archive or uploaded.
