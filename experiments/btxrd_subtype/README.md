# BTXRD subtype runs

Purpose: isolated Phase B HOSP-BoneNet training and fresh evaluation artifacts.

Files per run:
- `contract.json` — runtime, split checksums and execution mode.
- `best.pt`, `last.pt` — validated resumable checkpoints.
- `calibration_*.csv`, `test_*.csv` — fresh masked predictions.
- `metrics.json`, `status.json` — calibration/evaluation evidence.

Flow: `workspace/btxrd` manifests → masked trainer → BTXRD calibration → BTXRD test.

Run: `.venv\Scripts\python.exe scripts\18_train_btxrd_subtype.py`

Do not edit run artifacts manually; rerun the runner with the same contract to reuse a complete run.
