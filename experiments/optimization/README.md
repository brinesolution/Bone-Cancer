# Optimization Experiments

Purpose: Immutable, fingerprinted Phase 8 run bundles.

Files:
- `hosp_bonenet/<run>/study.db` — resumable Optuna study.
- `hosp_bonenet/<run>/trials.csv` — terminal trial summary.
- `hosp_bonenet/<run>/hosp_ablation.csv` — supported validation ablations.
- `hosp_bonenet/<run>/selection.json` — frozen validation-only decision.

Main libraries: Optuna, PyTorch, Pandas.

Flow: verified inputs/config/code → resumable stages → validated selection → checkpoint registry.

Run: `.venv\Scripts\python.exe scripts\11_tune_hosp_bonenet.py --config configs\optimization.yaml`

Do not rename or edit run files. Use `--force-new-run` only for an intentional fresh run.
