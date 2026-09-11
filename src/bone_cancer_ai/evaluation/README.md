# Evaluation

Purpose: Honest binary metrics, latency measurement and atomic reports.

Files:
- `classification.py` — AUROC, AUPRC, threshold metrics, Brier and ECE.
- `reports.py` — JSON/CSV writers, confusion matrices, latency and summary updates.
- Other modules — later segmentation, calibration and open-set work.

Main libraries: PyTorch, NumPy, Pandas, scikit-learn.

Flow: saved logits/targets → fixed threshold 0.5 → metrics and reports.

Validate: `\.venv\Scripts\python.exe scripts\09_train_baseline.py --validate-artifacts`

Undefined metrics remain null; never enter unavailable or estimated results.
