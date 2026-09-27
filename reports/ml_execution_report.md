# ML Execution Report — Business Entity Resolution (Amazon ML Challenge 2026)

**Team:** Codeconscious · **Date:** 2026-09-27 · **All metrics measured on real data. Nothing invented.**

## 1. Baseline (measured)

SGD logistic regression, caps 80/60, thr 0.88: **val F0.5 0.7270** (P 0.823 / R 0.576), test slice 0.7277. 66.5M candidates (38.4/row). Runtime 3,713 s.

## 2. Best configuration (validated)

**LightGBM, caps 200/60, thr 0.96: val F0.5 0.7942** (P 0.877 / R 0.654), test slice 0.7940. Blocking recall 0.704. Final inference runtime 3,586 s, 12 workers.

## 3. Model comparison (validation macro F0.5, same 50k calib slice)

| Model | Thr | Val F0.5 | P | R | Test | Recall_fit | Runtime |
|---|---|---|---|---|---|---|---|
| sgd | 0.88 | 0.7270 | 0.823 | 0.576 | 0.7277 | 0.659 | 1,752 s |
| hgb | 0.94 | 0.7701 | 0.861 | 0.619 | 0.7707 | 0.659 | 1,957 s |
| **lightgbm** | **0.94→0.96** | **0.7728→0.7942** | 0.877 | 0.654 | 0.7940 | 0.704 | 1,455/1,561 s |
| xgboost | 0.94 | 0.7696 | 0.861 | 0.618 | 0.7702 | 0.659 | 1,622 s |
| catboost | 0.94 | 0.7886 | 0.870 | 0.653 | 0.7886 | 0.704 | 1,175 s |

GBM spread is ±0.004 except sgd; all share the blocking ceiling. CatBoost fastest to fit.

## 4. Blocking experiments (10k val queries, ground truth)

| Config | Recall | cands/row | Verdict |
|---|---|---|---|
| 80/60 (baseline) | 0.658 | 37.8 | baseline |
| 120/60, 150/60, **200/60** | 0.692/0.699/**0.702** | 43.7/45.2/46.2 | **200/60 selected** |
| df_cap 120 (any cap) | 0.59–0.70 | 50–83 | REJECTED: noisy groups flood budget, row-id truncation drops good pairs |
| Per-key-type df caps | identical | identical | REJECTED |
| Soundex phonetic keys | zero gain | — | REJECTED (df-cap eats y: keys) |
| Per-token keys (t:/x:) | zero gain (0.65774 vs 0.65780) | — | REJECTED (rare pairs already covered by l:/f:/n:) |
| Soft-cap (whole groups) | 0.7009 at cap 80 | 40.6 | implemented `--soft-cap` (off); F0.5 equivalence unvalidated |

Miss autopsy (baseline, 7,004 pairs): HIT 65.9%, NO_SHARED_KEY 12.5%, DF_CAPPED 16.9%, BUDGET_CUT 4.8%.

## 5. Gate / graph (10k slice, same fitted lightgbm)

| Arm @ thr 0.70 | F0.5 | Singleton FP | Verdict |
|---|---|---|---|
| baseline | 0.73706 | 174 | — |
| evidence_gate | 0.73802 (157 rows changed, 175 dropped) | 142 | +0.001; vacuous at selected 0.94 → OFF |
| graph_expand | 0.73706 (0 rows changed) | 174 | REJECTED, kept in code |

## 6. Dense retrieval probe

MiniLM + FAISS on 50k/2k slice (resident pairs n=60): sparse 0.733 / dense 0.967 / union 0.967. Direction solid; full-corpus flat index infeasible (18 GB > RAM). IVF-PQ design documented as future work. Full-corpus embedding aborted at measured 142/s (9.5 h/file) — out of session budget.

## 7. Final outputs

`output/matching_results.tsv` (72.9 MB) + `output/candidate_pairs.tsv` (1,050.6 MB): 1,732,544 rows each. Matched 1,439,496 / singletons 293,048. Max links/row 34.

## 8. Verification

- Bundled validator: **PASS** · Official validator: **PASS** (1,732,544 rows)
- matches ⊆ candidates: **0 violations** in 1,732,544 rows
- Singleton format (empty 2nd col, `\t\n`): correct by construction
- pytest: **35/35 passed**
- Prior correction math: verified vs closed form (opt-in, off in final config)

## 9. Evidently diagnostics (`reports/evidently/`)

Train→test drift (50k samples): 3/6 columns drifted — country 0.255 (France 15% of test, unseen in train), addr_len 0.197, addr_toks 0.150. Prediction rates by country: FR 83.6%, IN 77.2%, US 90.0% — France handled by open-set binary country_match, no collapse.

## 10. Leakage / fair-play

Local-only (numpy/sklearn/lightgbm/rapidfuzz + stdlib; no network calls). Fit/val/test slices disjoint and positional. No test labels used. Model: LightGBM <1M params, MIT/Apache-2.0 deps only. France emitted by construction (rows driven by test_source1.tsv).

## 11. 0.90/0.95 verdict (honest)

Not reached. Sparse recall ceiling 0.702 caps F0.5 ≈ 0.85 even with perfect keys ported (per-token measured zero gain). Best measured: **0.7942 val / 0.76 leaderboard** (consistent gap). 0.90+ needs dense retrieval at full scale (recall ≥0.93 at P ≥0.88) — IVF-PQ design documented, ~31 h CPU encoding out of session budget.

## 12. Artifacts

- Outputs: `output/*.tsv` (+ `reports/baseline_outputs/` SGD backup)
- Model: `reports/final_model.json` + booster sidecar
- Experiments: `reports/experiment_results.csv` (31 rows), per-run JSONs
- State: `checkpoints/state.json`, `reports/best_config.json`, `reports/live_status.md`
- Tests: `code/.../tests/` (35 tests) · Requirements: pinned + optional GBM section

## 13. Exact final commands

```bash
python code/business_entity_resolution/src/run_pipeline.py --data-dir student_resource --mode test --model-type lightgbm --max-candidates 200 --output-dir output --report reports/final_test_report.json --save-model reports/final_model.json
python code/business_entity_resolution/src/run_pipeline.py --mode validate --data-dir student_resource --output-dir output
python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
python -m pytest code/business_entity_resolution/tests -q
```

## 14. Recommendation

Ship `output/matching_results.tsv` (LightGBM caps-200). Next program: (1) soft-cap F0.5 validation (recall-equal at lower volume), (2) IVF-PQ dense blocking for the recall wall, (3) dense-cosine features for precision. Stop conditions met: all reasonable in-session paths measured, best selected on validation, submission valid.
