# Project state

Current phase: Phase 8 — Optimization, Ablations and Accuracy Improvement COMPLETE
Completed phases: 1–8
Next phase: Phase 9 — Final Evaluation, Explainability and Visual QA

## Environment

- Python: 3.11.15 (`.venv`)
- PyTorch: 2.14.0+cu130; torchvision: 0.29.0+cu130; timm: 1.0.29
- CUDA: 13.0 available
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB
- Branch: `phase1-scaffold`

## Dataset

- Immutable source: `dataset/Main`
- Source: 12,557 images; 0=`normal` 6,827, 1=`bone_tumour` 5,730
- Source preservation: 12,560 files / 1,120,418,007 bytes; exact Phase 1 match after Phase 8
- Derived splits: fit 4,871; calibration 857; validation 1,453; test 1,882
- Leakage: zero record/sample/group overlap; split generation unchanged
- Inputs: verified 384×384 fit/validation cache; dynamic fit augmentation

## Supervision gates

- Verified training supervision: binary only
- B3 concepts, B4 supervised spatial grounding and B5 hierarchy were not run
- Subtype, risk, concepts and supervised masks remain disabled
- Human/clinical verification: Unavailable

## Phase 8 result

- Run: `experiments/optimization/hosp_bonenet/phase8-5e55d4bddd5b`
- Parent: `hosp_bonenet.phase7.selected`
- Data: 384, current mild augmentation, weighted sampler, verified cache
- Optuna: 10 terminal trials; 4 complete, 6 pruned; trial 1 selected
- Tuned: lr 3e-6, weight decay 1e-4, warmup 0, dropout 0.3, label smoothing 0
- Supported warm-start ablations: B0 plain, B1 localization attention, B2 gated global-local fusion
- Selected architecture: B2; best confirmation epoch: 2/3
- Validation AUROC/AUPRC/F1: 0.958522 / 0.964397 / 0.897196
- Validation balanced accuracy/ECE/Brier: 0.893758 / 0.103580 / 0.103251
- Test: Not run; reserved for Phase 9

## Latest models

- DenseNet121 test AUROC/AUPRC: 0.96315 / 0.96519
- EfficientNet-B3 test AUROC/AUPRC: 0.97810 / 0.97612
- ConvNeXt-Tiny test AUROC/AUPRC: 0.96401 / 0.96394
- Swin-Tiny test AUROC/AUPRC: 0.96517 / 0.96355
- HOSP-BoneNet Phase 8 validation AUROC/AUPRC: 0.958522 / 0.964397; test Not run

## Artifact reuse

- Selected alias: `hosp_bonenet.phase8.selected`
- Checkpoint SHA-256: `c24ed27c83c00aba245d2e28d7a39c3cd9fb027b311049625e3c915b0a711ec5`
- Frozen config: `configs/hosp_bonenet_phase8_final.yaml`
- Exact completed reruns validate and reuse; interrupted stages resume; `--force-new-run` starts fresh
- Registry and frozen loader reject changed or checksum-invalid artifacts

## Verification

- `pytest -q`: 445 passed
- Ruff, compileall, pip check, CUDA/AMP environment check: PASS
- Completed-run reuse and full bundle validation: PASS
- Frozen checkpoint strict CUDA load/forward: PASS
- Source preservation and absence of test artifacts: PASS
- Independent code review: CLEAN

## Important paths

- Phase 8 config: `configs/optimization.yaml`
- Frozen config: `configs/hosp_bonenet_phase8_final.yaml`
- Runner: `scripts/11_tune_hosp_bonenet.py`
- Core: `src/bone_cancer_ai/optimization/`
- Run: `experiments/optimization/hosp_bonenet/phase8-5e55d4bddd5b`
- Ablations: `artifacts/metrics/hosp_ablation.csv`
- Registry: `artifacts/checkpoints/registry.json`

## Open issues

- Test evaluation, explainability galleries and statistical intervals belong to Phase 9.
- Classification-guided lesion maps have no claimed segmentation accuracy.
- Multi-seed and clinical validation remain Pending/Unavailable.

## Next action

- Use ASTRA for Phase 9. Load `hosp_bonenet.phase8.selected`; evaluate frozen models once on test and generate explainability/QA artifacts.
