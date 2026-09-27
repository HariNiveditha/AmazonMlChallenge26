# Emergency Baseline Checkpoint (11:40 IST deadline program)

**Frozen:** 2026-09-27 ~22:23 IST. Never overwritten by emergency runs.

## Best validated pipeline
- Model: LightGBM (`tree_lightgbm`), caps 200/60, threshold 0.96
- Validation macro F0.5: **0.7942** (P 0.877 / R 0.654), test slice 0.7940
- Leaderboard: 0.767 (consistent mix-shift gap, root-caused)
- Blocking recall: 0.704 · 46.05 cands/row · 79.8M candidates

## Preserved artifacts (`reports/emergency_baseline/`)
- matching_results.tsv (72.9 MB, 1,732,544 rows)
- candidate_pairs.tsv (1,050.6 MB, 1,732,544 rows)
- final_test_report.json, final_model.json + booster pkl, best_config.json

## Live outputs
- `output/*.tsv` currently hold the same best configuration (final inference).
- Emergency runs write to `output_emergency/` only.

## Verdicts already measured (do not re-run)
- df_cap 120 hurts recall · phonetic/per-token/per-key-df zero gain · graph 0 fired · gate +0.001 below 0.90 only · CatBoost 0.7886 < LightGBM · dense full-corpus infeasible at 142/s.
