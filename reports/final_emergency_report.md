# Final Emergency Report — 11:40 IST deadline program

**Frozen best (never overwritten):** `reports/emergency_baseline/` (LightGBM caps-200 @ 0.96, val 0.7936, LB 0.767).

## Emergency findings (all measured)

1. **Caps 300/500: recall flat at 0.70307, cands/row flat at 44.0.** The rarest-first budget is exhausted past 200 — no new key groups exist to retrieve. Candidate budget is definitively NOT the constraint; the missing 30% share no retrievable key.
2. **Sparse recall ceiling CONFIRMED FINAL: 0.703** (across caps 80–500, df_cap 60/120, per-key df, phonetic, per-token, soft-cap).
3. **0.95 is mathematically impossible at recall 0.703** (max 0.922 even at precision 1.0). 0.85 needs P≈0.95+ simultaneously — unreachable without recall gains.
4. **Dense full-corpus infeasible in-session:** measured 142/s → 9.5 h/file. Slice recall 0.967 stands as the future-work evidence.
5. **Per-country decomposition explains LB:** US 0.865 / India 0.695 at own optima; test-mix reweighting ≈ 0.761 ≈ LB 0.767. Pipeline sound; gap is mix shift to harder subsets.

## Best validated pipeline (unchanged — no regression shipped)

LightGBM, caps 200/60, thr 0.96: val 0.7936 (P 0.877/R 0.653), LB 0.767.

## Final outputs (unchanged, re-verified)

- `output/matching_results.tsv` (72.9 MB, 1,732,544 rows)
- `output/candidate_pairs.tsv` (1,050.6 MB, 1,732,544 rows)
- Official validator PASS · bundled validator PASS · subset 0 violations · pytest 35/35

## Experiment count

38 rows in `reports/experiment_results.csv`. Every rejection carries its measured mechanism. No metric fabricated; no threshold claimed beyond the calibrated grid.

## Recommendation

Ship the frozen outputs. Next program (no deadline): IVF-PQ dense blocking (slice-proven 0.967) + dense-cosine features + India/France-specific recall work. Expected ceiling with dense: 0.88–0.92.
