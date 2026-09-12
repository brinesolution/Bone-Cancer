# Phase 7 HOSP-BoneNet Training Plan

**Goal:** Train the first full binary-supervised HOSP-BoneNet and select a reproducible checkpoint by validation AUROC.

## Constraints

- Consume only preserved `fit_train` and `val` manifests; do not evaluate test/calibration.
- Run Stage A binary global/local learning, then initialize Stage D from Stage A best with a fresh lower-rate optimizer.
- Keep Stages B/C disabled because masks/subtypes/concepts are not released.
- Use CUDA AMP, effective batch 8, seed 42, and validation-only early stopping.

## Tasks

1. Extend the shared trainer for HOSP output mappings, per-stage monitoring and resume-safe checkpoints.
2. Add strict HOSP training config resolution and an immutable run-bundle CLI.
3. Add tests for stage transitions, metrics, no-test access, artifacts and checkpoint reload.
4. Run one-batch CUDA smoke and checkpoint reload before full training.
5. Run the complete 384-pixel training, reload best, save validation metrics/curves, and validate artifacts.
6. Review, fix, run the full suite and source/split preservation checks, then update concise handoff docs.

## Acceptance

- Training finishes or reaches configured early stopping without NaNs.
- History records train/validation loss, AUROC, AUPRC, F1, LR, GPU memory and epoch time.
- Both stage checkpoints reload with exact identity; final selection uses the larger validation AUROC with an earlier-stage tie break.
- Test/calibration loaders are never read; optional tasks remain disabled.
