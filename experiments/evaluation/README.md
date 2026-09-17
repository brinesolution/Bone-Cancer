# Phase 9 Evaluation Runs

Purpose:
Store frozen evaluation provenance and derived HOSP calibration/test records.

Files:
- `phase9-*/status.json` — checksum manifest and final summary.
- `calibration_predictions.csv` — held-out conformal calibration predictions.
- `hosp_test_predictions_enriched.csv` — sealed test predictions plus uncertainty.
- `conformal_summary.json` — threshold, coverage and set-size summary.

Main libraries:
PyTorch, Pandas, scikit-learn and Matplotlib.

Flow:
Frozen aliases + immutable splits → sealed predictions → metrics, plots and galleries.

Run:
`.venv\Scripts\python.exe scripts\12_evaluate.py --config configs\evaluation.yaml`

After completing `visual_qa.json`, publish with the same command plus `--finalize-visual-qa`.

Do not edit generated run files or repeat test inference for the same checkpoint.
