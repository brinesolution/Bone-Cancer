# Phase handoff

Phase completed:
Phase 7 — Full Training and Checkpoint Selection

Important outputs:
- `experiments/hosp_bonenet/20260912T121852779982Z-a371f643`
- `experiments/hosp_bonenet/20260912T121852779982Z-a371f643/stage_D/best.pt`
- `configs/hosp_train.yaml`
- `scripts/10_train_hosp_bonenet.py`

Tests:
- PASS — 353 pytest tests
- PASS — Ruff, compileall, pip check, environment/CUDA
- PASS — bundle integrity and exact Stage A/D checkpoint reload
- PASS — source and split preservation
- CLEAN — independent review

Important result:
- Stage D epoch 6 selected by validation AUROC 0.958355
- Validation AUPRC 0.964236; F1 0.902389
- Test: Not run; reserved for Phase 9

Outstanding issue:
- Optional supervised branches remain gated; test evaluation is pending Phase 9.

Do not rerun:
- Phase 7 full run or earlier audits/baselines unless relevant inputs or code change.

Next model:
SOL

Next phase:
Phase 8 — Optimization and Accuracy Improvement

Start command:
`Get-Content PROJECT_STATE.md`

Resume from:
`PROJECT_STATE.md`
