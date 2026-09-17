# Phase handoff

Phase completed:
Phase 8 — Optimization, Ablations and Accuracy Improvement

Important outputs:
- `experiments/optimization/hosp_bonenet/phase8-5e55d4bddd5b`
- `configs/hosp_bonenet_phase8_final.yaml`
- `artifacts/metrics/hosp_ablation.csv`
- alias `hosp_bonenet.phase8.selected`

Tests:
- PASS — 445 pytest tests
- PASS — Ruff, compileall, pip check, CUDA/AMP
- PASS — completed-run reuse and bundle/checksum validation
- PASS — frozen CUDA load/forward and source preservation
- CLEAN — independent Phase 8 code/artifact review

Important result:
- B2 selected; validation AUROC/AUPRC 0.958522/0.964397
- 10 Optuna trials; test Not run

Outstanding issue:
- Phase 9 owns one-time test evaluation, explainability and visual QA.

Do not rerun:
- Phase 8 training unless verified inputs/config/code change; default command reuses.

Next model:
ASTRA

Next phase:
Phase 9 — Final Evaluation, Explainability and Visual QA

Start command:
`Get-Content PROJECT_STATE.md; Get-Content scripts\12_evaluate.py`

Resume from:
`PROJECT_STATE.md`
