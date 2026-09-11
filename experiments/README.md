# Experiments

Purpose: Immutable, uniquely named outputs from executed model runs.

Folders:
- `baselines/` — four Phase 5 model families and run bundles.
- `hosp_bonenet/`, `ablations/`, `tuning/` — later-phase outputs.

Each complete run contains config, environment/Git evidence, history, best/last checkpoints, predictions, confusion matrix, metrics and notes.

Flow: resolved config + verified manifests → training → immutable run directory.

Run: `\.venv\Scripts\python.exe scripts\09_train_baseline.py --config <config>`

Do not rename, overwrite or manually edit run artifacts.
