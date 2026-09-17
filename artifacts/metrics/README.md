# Metrics

Purpose: Machine-readable metrics produced by executed evaluations.

Files:
- `baseline_summary.csv` — one validated seed-42 row per approved baseline.
- `hosp_ablation.csv` — supported Phase 8 validation-only HOSP variants and frozen FULL result.

Columns include AUROC, AUPRC, balanced accuracy, F1, sensitivity, specificity, MCC, parameters and latency.

Flow: complete run `metrics.json` → strict validation → atomic summary update.

Validate Phase 8: `.venv\Scripts\python.exe scripts\11_tune_hosp_bonenet.py --config configs\optimization.yaml`

Per-run predictions and complete metadata remain in `experiments/`.

Do not edit values manually; use `Not run`, `Pending` or `Unavailable` when appropriate.
