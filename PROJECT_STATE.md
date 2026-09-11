# Project state

Current phase: Phase 5 — Standard Comparison Baselines COMPLETE
Completed phases: 1–5
Next phase: Phase 6 — HOSP-BoneNet Implementation

## Environment

- Python: 3.11.15 (`.venv`)
- PyTorch: 2.14.0+cu130
- torchvision: 0.29.0+cu130
- timm: 1.0.29
- CUDA: 13.0 available
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB
- Branch: `phase1-scaffold`

## Dataset

- Immutable source: `dataset/Main`
- Source labels/images: 12,557; class 0=`normal` 6,827, class 1=`bone_tumour` 5,730
- Source snapshot: 12,560 files, 1,120,418,007 bytes; Phase 5 exactly matches Phase 4
- Derived splits: fit 4,871; calibration 857; validation 1,453; test 1,882
- Split generation: `de64a489fff7202ec39d442e4b26a5a5c0d7e2c7db8ed7a8587601ef1cd28590`
- Leakage: zero record/sample/group overlap; zero held-out records moved to training
- Preprocessing: grayscale-replicated RGB, aspect-preserving 384×384 pad, ImageNet normalization
- Labels used in Phase 5: verified binary only; filename-derived diagnosis is not an input

## Annotation gates

- Accepted Phase 4 source generation: `c204d1f6873644948e4f355f26b399ae`
- Valid source spatial records: 1,825; conflicts: 16 JSONs / 8 exact-image groups; unmatched: 26
- Human project reviews: 0; clinical verification: Unavailable
- Subtype, risk, concept and lesion heads remain disabled until valid labels are joined to preserved splits

## Phase 5 implementation

- One four-name registry and `TimmClassifier`; no architecture-specific training copies
- Shared weighted-BCE AMP trainer, AdamW 1e-4, weight decay 1e-4, cosine schedule, warmup 2
- Seed 42, max 40 epochs, patience 8, gradient clip 1.0, validation-AUROC selection
- DenseNet/EfficientNet/Swin: batch 8, accumulation 1
- ConvNeXt: measured memory exception, microbatch 4 × accumulation 2 = effective batch 8
- Unique run bundles include config and provenance; checkpoints and report files are written atomically
- Test is evaluated once only after loading the best validation checkpoint

## Latest models and executed test metrics

| Model | Best/last epoch | AUROC | AUPRC | Balanced acc. | F1 | MCC | Params | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 | 28/36 | 0.96315 | 0.96519 | 0.89795 | 0.88989 | 0.79305 | 6,954,881 | 10.925 |
| EfficientNet-B3 | 23/31 | 0.97810 | 0.97612 | 0.92302 | 0.91624 | 0.84379 | 10,697,769 | 11.100 |
| ConvNeXt-Tiny | 28/36 | 0.96401 | 0.96394 | 0.89865 | 0.89052 | 0.79462 | 27,820,897 | 6.671 |
| Swin-Tiny | 10/18 | 0.96517 | 0.96355 | 0.89901 | 0.88916 | 0.80643 | 27,520,123 | 8.168 |

Final run IDs:
- DenseNet121: `20260910T201824489363Z-8b08661d`
- EfficientNet-B3: `20260911T035833073159Z-4d5b4f29`
- ConvNeXt-Tiny: `20260911T150346748828Z-840945d2`
- Swin-Tiny: `20260911T175501792702Z-21542b12`

## Verification

- `pytest -q`: 316 passed
- Ruff: PASS; compileall: PASS; pip check: PASS
- Phase 5 artifact validator: 4 models / 4 runs / valid
- All saved test metrics recomputed exactly from 1,882 ordered predictions per model
- All eight best/last checkpoints reloaded with exact model/config identity
- Source snapshot and split fingerprint/count/integrity checks: PASS
- Independent review: no Critical or Important findings

## Important paths

- Summary: `artifacts/metrics/baseline_summary.csv`
- Runs: `experiments/baselines/<model>/<run-id>/`
- Configs: `configs/train.yaml`, `configs/baseline_*.yaml`
- Training CLI: `scripts/09_train_baseline.py`
- Evidence: `logs/phase5_run_verification.json`, `logs/source_after_phase5.json`, `logs/phase5_split_preservation.json`
- Memory evidence: `artifacts/verification/phase5_convnext_batch8_memory_failures.json`

## Open issues

- Multi-seed comparison: Pending; only seed 42 was run.
- DenseNet honestly records `dirty=true`; the exact launch diff is unavailable, while run/metric/checkpoint integrity passed.
- Patient/study IDs and qualified clinical verification are unavailable.
- Baseline probabilities are not yet conformally calibrated; calibration remains Phase 8/9 work.

## Next action

Implement Phase 6 HOSP-BoneNet modules and masked losses, then pass shape/AMP/gradient and 16–32 sample overfit acceptance tests. Use SOL. Do not rerun the four final baselines, data audit, split generation or source annotation conversion unless inputs/configuration change. Never alter `dataset/Main`.
