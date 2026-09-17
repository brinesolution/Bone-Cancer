# Optimization

Purpose: Phase 8 validation-only data screening, tuning, ablation, selection and reuse.

Files:
- `contracts.py` — strict configs, hashes, ranking and subset contracts.
- `execution.py` — reusable training/evaluation stages.
- `workflow.py` — ordered runner, Optuna study and transactional publication.
- `frozen.py` — checksum-verified selected model loading.

Main libraries: PyTorch, Optuna, Pandas, PyYAML.

Flow: Phase 7 alias + fit/validation cache → screens → 10 trials → supported ablations → frozen alias.

Run: `.venv\Scripts\python.exe scripts\11_tune_hosp_bonenet.py --config configs\optimization.yaml`

Do not edit generated run bundles or frozen metadata manually. Test data is excluded from Phase 8.
