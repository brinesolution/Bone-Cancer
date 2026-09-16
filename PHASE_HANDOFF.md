# Phase handoff

Phase completed:
Phase 7 — Full Training and Checkpoint Selection

Important outputs:
- `experiments/hosp_bonenet/20260912T121852779982Z-a371f643`
- `artifacts/checkpoints/registry.json` (five verified aliases)
- `workspace/cache/resized/cache_index.csv` (6,324 verified entries)

Tests:
- PASS — 411 pytest tests
- PASS — Ruff, compileall, pip check, environment/CUDA
- PASS — bundle/registry integrity and five no-training reruns
- PASS — exact cache reuse and cached DataModule batches
- PASS — source and split preservation
- CLEAN — independent registry/cache review

Important result:
- Stage D epoch 6 selected by validation AUROC 0.958355
- Validation AUPRC 0.964236; F1 0.902389
- Test: Not run; reserved for Phase 9

Outstanding issue:
- Optional supervised branches remain gated; test evaluation is pending Phase 9.

Do not rerun:
- Phase 5/7 training or cache rebuild unless verified inputs/config/code change; default commands reuse.

Next model:
SOL

Next phase:
Phase 8 — Optimization and Accuracy Improvement

Start command:
`Get-Content PROJECT_STATE.md; .venv\Scripts\python.exe tools/refresh_checkpoint_registry.py`

Resume from:
`PROJECT_STATE.md`
