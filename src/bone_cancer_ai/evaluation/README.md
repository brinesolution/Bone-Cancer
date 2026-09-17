# Evaluation

Purpose: Frozen binary evaluation, uncertainty intervals and comparison figures.

Files:
- `classification.py`, `bootstrap.py` — metrics and stratified confidence intervals.
- `calibration.py`, `comparison.py` — uncertainty helpers and required plots.
- `phase9.py` — one-time HOSP test ledger, comparison and gallery workflow.
- `reports.py` — atomic JSON/CSV persistence and latency measurement.

Main libraries: PyTorch, NumPy, Pandas, scikit-learn.

Flow: frozen/saved predictions → cohort checks → metrics/CIs → plots and explanations.

Run: `.venv\Scripts\python.exe scripts\12_evaluate.py --config configs\evaluation.yaml`

Finalize reviewed galleries: append `--finalize-visual-qa`.

The test ledger fails closed on changed contracts or checksums and prevents repeat evaluation.
