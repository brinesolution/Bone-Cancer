# Project state

Current phase: Phase 6 — HOSP-BoneNet Implementation COMPLETE
Completed phases: 1–6
Next phase: Phase 7 — Full HOSP-BoneNet Training

## Environment

- Python: 3.11.15 (`.venv`)
- PyTorch: 2.14.0+cu130; torchvision: 0.29.0+cu130; timm: 1.0.29
- CUDA: 13.0 available
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB
- Branch: `phase1-scaffold`

## Dataset

- Immutable source: `dataset/Main`
- Source images/labels: 12,557; class 0=`normal` 6,827, class 1=`bone_tumour` 5,730
- Source snapshot: 12,560 files, 1,120,418,007 bytes; exactly matches Phase 5
- Derived splits: fit 4,871; calibration 857; validation 1,453; test 1,882
- Split generation: `de64a489fff7202ec39d442e4b26a5a5c0d7e2c7db8ed7a8587601ef1cd28590`
- Leakage: zero record/sample/group overlap; all four split files match Phase 5 hashes
- Inputs: grayscale-replicated RGB, aspect-preserving 384×384 pad, ImageNet normalization

## Annotation gates

- Verified training supervision: binary only
- Valid source spatial records: 1,825; not released into preserved training manifests
- Human project reviews: 0; clinical verification: Unavailable
- Subtype, risk, concept and supervised mask branches remain disabled

## Phase 6 implementation

- ConvNeXt-Tiny multistage encoder with discovered channel/reduction metadata
- Input-aligned lightweight lesion FPN, global pooling and FP32 mask-weighted local pooling
- Learned global/local fusion and configuration-gated concept/subtype/risk heads
- Strict factory derives optional dimensions only from enabled label-schema sections
- Masked binary/multiclass, Dice+BCE, concept, grounding and hierarchy losses
- FP32 AMP reductions; valid labels cannot be silently dropped through missing outputs
- Label-free inference returns only configuration-enabled tensor keys

## Phase 6 smoke result

- Split: fit_train only; 16 runtime-valid images, balanced 8/8
- Backbone: ConvNeXt-Tiny; pretrained false; 96×96; CUDA AMP
- Initial/final loss: 0.687840 / 0.104092; ratio 0.151331
- Initial/final accuracy: 0.4375 / 1.0000; 10 epochs
- Trainable parameters: 28,922,338; peak allocated CUDA memory: 648,881,152 bytes
- This is an overfit acceptance check, not validation/test performance

## Latest models

- DenseNet121: test AUROC 0.96315; AUPRC 0.96519
- EfficientNet-B3: test AUROC 0.97810; AUPRC 0.97612
- ConvNeXt-Tiny: test AUROC 0.96401; AUPRC 0.96394
- Swin-Tiny: test AUROC 0.96517; AUPRC 0.96355
- HOSP-BoneNet checkpoint/validation/test metrics: Not run; Phase 7

## Verification

- `pytest -q`: 337 passed
- Ruff, compileall, pip check, environment/CUDA check: PASS
- Real 384×384 ConvNeXt HOSP forward/backward in CUDA AMP: PASS
- Binary, subtype, mask and full multitask loss tests: PASS
- Tiny overfit: PASS; implementation/config hashes recorded
- Source/split preservation: PASS
- Independent review: CLEAN; zero Critical/Important findings

## Important paths

- Config: `configs/hosp_bonenet.yaml`
- Architecture: `src/bone_cancer_ai/models/hosp_bonenet/`
- Losses: `src/bone_cancer_ai/losses/`
- Tests: `tests/test_hosp_bonenet.py`, `tests/test_hosp_losses.py`
- Overfit: `scripts/phase6_smoke_overfit.py`
- Evidence: `artifacts/verification/phase6_tiny_overfit.json`, `logs/phase6_verification.json`

## Open issues

- Full HOSP-BoneNet training/checkpoint selection is pending Phase 7.
- Optional task supervision and open-set/conformal calibration remain unavailable until their gates open.
- Classification-guided lesion maps have no claimed segmentation accuracy.
- Patient/study IDs and qualified clinical verification are unavailable.

## Next action

- Implement Phase 7 staged binary HOSP training on fit_train, select on validation only, and preserve test isolation. Use SOL. Do not rerun Phase 5 baselines, audits, splits, annotation conversion, or Phase 6 overfit unless inputs/code change. Never alter `dataset/Main`.
