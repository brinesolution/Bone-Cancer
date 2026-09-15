# Project state

Current phase: Phase 7 — Full Training and Checkpoint Selection COMPLETE
Completed phases: 1–7
Next phase: Phase 8 — Optimization and Accuracy Improvement

## Environment

- Python: 3.11.15 (`.venv`)
- PyTorch: 2.14.0+cu130; torchvision: 0.29.0+cu130; timm: 1.0.29
- CUDA: 13.0 available
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB
- Branch: `phase1-scaffold`

## Dataset

- Immutable source: `dataset/Main`
- Source images/labels: 12,557; class 0=`normal` 6,827, class 1=`bone_tumour` 5,730
- Source snapshot: 12,560 files, 1,120,418,007 bytes; exact Phase 1 match after Phase 7
- Derived splits: fit 4,871; calibration 857; validation 1,453; test 1,882
- Split generation: `de64a489fff7202ec39d442e4b26a5a5c0d7e2c7db8ed7a8587601ef1cd28590`
- Leakage: zero record/sample/group overlap; all split hashes preserved
- Inputs: grayscale-replicated RGB, aspect-preserving 384×384 pad, ImageNet normalization

## Supervision gates

- Verified training supervision: binary only
- Stages B/C skipped: spatial labels are not released; subtype/concept labels are unavailable
- Human project reviews: 0; clinical verification: Unavailable
- Optional subtype, risk, concept and supervised mask heads remain disabled

## Phase 7 training

- Run: `experiments/hosp_bonenet/20260912T121852779982Z-a371f643`
- Input: fit_train only; checkpoint selection: validation AUROC only; test: Not run
- Stage A: 35 epochs, early stopped; best epoch 27; validation AUROC 0.956851
- Stage D: 10 epochs at 1e-5; selected epoch 6; validation AUROC 0.958355
- Selected validation AUPRC: 0.964236; F1: 0.902389; balanced accuracy: 0.901817
- Selected confusion matrix: TN=649, FP=62, FN=81, TP=661
- Selected checkpoint: `stage_D/best.pt`
- SHA-256: `a6380739e6527a085434866c83a135b86ca118e82f8d5a8bf75d934d25ce268e`
- Stage A/D best-checkpoint logits, targets and sample IDs reload exactly
- Stage A resumed from intact epoch-29 state after a host-memory interruption; provenance is recorded
- Training histories and curves exist for both executed stages; no unexplained NaNs

## Latest models

- DenseNet121 test AUROC/AUPRC: 0.96315 / 0.96519
- EfficientNet-B3 test AUROC/AUPRC: 0.97810 / 0.97612
- ConvNeXt-Tiny test AUROC/AUPRC: 0.96401 / 0.96394
- Swin-Tiny test AUROC/AUPRC: 0.96517 / 0.96355
- HOSP-BoneNet validation AUROC/AUPRC: 0.95835 / 0.96424; test: Not run

## Verification

- `pytest -q`: 353 passed
- Ruff, compileall, pip check, environment/CUDA check: PASS
- Run-bundle validation and both-stage exact checkpoint reload: PASS
- Source and split preservation: PASS
- Independent Phase 7 review: CLEAN

## Important paths

- Training config: `configs/hosp_train.yaml`
- Model config: `configs/hosp_bonenet.yaml`
- Trainer: `src/bone_cancer_ai/training/trainer.py`
- Entry point: `scripts/10_train_hosp_bonenet.py`
- Selected run: `experiments/hosp_bonenet/20260912T121852779982Z-a371f643`
- Preservation evidence: `logs/source_after_phase7.json`, `logs/phase7_split_preservation.json`

## Open issues

- Test evaluation is reserved for Phase 9.
- Optional task supervision remains gated by unavailable/unreleased verified labels.
- Classification-guided lesion maps have no claimed segmentation accuracy.
- Multi-seed and clinical validation are Pending/Unavailable.

## Next action

- Run ordered Phase 8 optimization and ablations using validation data only. Preserve the selected Phase 7 run, test isolation, source data, and split generation. Use SOL.
