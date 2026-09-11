# Models

Purpose: Reusable model registry and implementations for baselines and HOSP-BoneNet.

Files:
- `registry.py` — approved aliases and timm backbone specifications.
- `baselines/` — shared Phase 5 binary classifier.
- `hosp_bonenet/` — Phase 6 architecture modules.

Main libraries: PyTorch, timm.

Flow: model alias → registry → model factory → logits.

Run: `\.venv\Scripts\python.exe scripts\09_train_baseline.py --config configs\baseline_densenet121.yaml`

Do not add competing baseline pipelines or edit generated checkpoints manually.
