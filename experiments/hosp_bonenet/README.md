# HOSP-BoneNet Experiments

Purpose:
Immutable training runs for the primary HOSP-BoneNet model.

Files:
- `<run>/config.yaml` — resolved run settings
- `<run>/stage_A|stage_D/` — checkpoints, histories, curves and validation outputs
- `<run>/selection.json` — validation-selected checkpoint and artifact hashes
- `<run>/status.json` — completion and test-access status

Main libraries:
PyTorch, timm, TorchMetrics, Matplotlib.

Flow:
fit_train → Stage A → Stage D → validation-AUROC selection

Run:
`.venv\Scripts\python.exe scripts\10_train_hosp_bonenet.py --config configs\hosp_bonenet.yaml`

Validate:
`.venv\Scripts\python.exe scripts\10_train_hosp_bonenet.py --validate-bundle <run_dir>`

Do not edit generated run artifacts or evaluate test before Phase 9.
