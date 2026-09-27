# LIVE STATUS — ER improvement program (autonomous)

START TIME IST: 2026-09-27 ~12:30 AM
CURRENT TIME IST: (updated per event below)
ESTIMATED FINISH IST: ~resume + stitch + audit + validators + Evidently + report (see log)
CURRENT STAGE: final_resume
CURRENT CHUNK: 444/694 scored (250 missing, rescuing single-process)
TOTAL CHUNKS: 694
COMPLETED CHUNKS: 444
REMAINING CHUNKS: 250
BEST VALIDATION MACRO F0.5: 0.7942 (lightgbm caps-200 @ thr 0.96; test slice 0.7940)
CANDIDATE RECALL: 0.7039 (caps-200 fit slice)
PRECISION: 0.87734 | RECALL: 0.65379 (caps-200 val)
MODEL: tree_lightgbm | CANDIDATE CAP: 200 | WORKERS: 1 | BOOSTER THREADS: multi (saved booster; n_jobs=1 pinned for future fits)
RAM: resume stable ~1.9 GB (was 6-worker storm before)
LAST CHECKPOINT: checkpoints/state.json + per-chunk .bin files (444) + winner_model.json
NEXT ACTION: resume finish -> stitch -> audit script -> both validators -> Evidently -> ml_execution_report.md
ETA: recomputed from measured chunk rate at each check; QUALITY > SPEED throughout

## Launch standard (all future runs)
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; boosters n_jobs/thread_count=1 (pinned in fit_batch). Single-process for cap>=150 inference until RAM safety is measured otherwise.

## Log (newest last)
- SUBMISSION READY: all processes stopped (0 running). codeconscious_submission.zip built + integrity-checked (762 MB: both TSVs + methodology doc + full code incl. tests). Validators PASS, pytest 29/29, audit PASS on shipped files.
- Matrix: sgd 0.7270, hgb 0.7701, lightgbm 0.7728 (best), xgboost 0.7696. Old-code hgb crash diagnosed + fixed.
- Caps: 200/60 -> recall 0.7025 (+4.5pp, +8 cands/row); df_cap=120 HURTS (noise flood + row-id truncation). Phonetic REJECTED (zero gain, df-cap eats y: keys). Dense slice: 0.967 vs sparse 0.733 (n=60; full-corpus 18GB infeasible).
- Gate +0.001 below 0.90 only (vacuous at 0.94, OFF); graph 0 rows fired (REJECTED, kept in code).
- caps-200 validation: lightgbm val 0.7942/test 0.7940 @ 0.96. Grid extended 0.96->0.98. --save-model dead flag wired up.
- Final inference died twice mid-scoring (presumed OOM: booster thread storm x caps-200 churn). Resuming 250 missing chunks single-process via resume_final.py (25 rescued so far, RAM 1.86 GB stable).
- CatBoost installed + wired (thread_count=1) + unit-tested; CSV migrated to §16 schema (backup .v1.csv); SPA/GIE suffixes added; pytest 28/28.
