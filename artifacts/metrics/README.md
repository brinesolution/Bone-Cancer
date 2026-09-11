# Metrics

Purpose: Machine-readable metrics produced by executed evaluations.

Files:
- `baseline_summary.csv` — one validated seed-42 row per approved baseline.

Columns include AUROC, AUPRC, balanced accuracy, F1, sensitivity, specificity, MCC, parameters and latency.

Flow: complete run `metrics.json` → strict validation → atomic summary update.

Validate: `.\scripts\run_phase.ps1 -Action phase5`

Per-run Brier/ECE, confusion matrices and predictions remain in `experiments/baselines/`.

Do not edit values manually; use `Not run`, `Pending` or `Unavailable` when appropriate.
