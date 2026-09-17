# Metrics

Purpose: Machine-readable metrics produced by executed evaluations.

Files:
- `baseline_summary.csv` — one validated seed-42 row per approved baseline.
- `hosp_ablation.csv` — supported Phase 8 validation-only HOSP variants and frozen FULL result.
- `final_model_comparison.csv` — five-model test metrics and bootstrap intervals.
- `hosp_phase9_test_ledger.json` — one-time HOSP test contract and prediction digest.
- `hosp_conformal_calibration.json` — frozen Phase 9 conformal threshold and coverage.

Columns include AUROC, AUPRC, balanced accuracy, F1, sensitivity, specificity, MCC, parameters and latency.

Flow: sealed predictions → cohort/checksum validation → metrics and intervals.

Validate Phase 9: `.venv\Scripts\python.exe scripts\12_evaluate.py --config configs\evaluation.yaml`

Per-run predictions and complete metadata remain in `experiments/`.

Do not edit values manually; use `Not run`, `Pending` or `Unavailable` when appropriate.
