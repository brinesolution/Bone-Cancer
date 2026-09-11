# Phase 5 Standard Comparison Baselines Work Notes

**Status:** COMPLETE — implementation, real runs, review and verification passed.

**Authoritative source:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md`, Sections 10–12 and Phase 5. Detailed execution plan: `docs/superpowers/plans/2026-09-10-phase-5-baselines.md`.

## IDEA

Phase 5 establishes fair empirical reference models before HOSP-BoneNet is optimized. Inputs are the unchanged 384-pixel preprocessing, fit/validation/test manifests, verified binary semantics, disabled optional-label gates and CUDA environment. Outputs are four real run bundles, checkpoints and a common summary. Risks are test-set selection, architecture-specific pipelines, unequal preprocessing/budgets, pretrained-weight/API mismatch, CUDA OOM, stale checkpoints, missing/invalid sample labels and accidental invented metrics.

## DESIGN

- One `TimmClassifier` and frozen four-name registry; no competing custom baseline implementations.
- One shared config/trainer/evaluator. Model configs select only architecture; memory adjustments must be measured and recorded.
- Fit uses only fit_train. Validation AUROC selects best checkpoint and early stopping. Calibration stays reserved. Test runs exactly after best selection and never changes training decisions.
- BCEWithLogits uses the fit-only positive weight. Invalid samples are masked. Metrics use a fixed 0.5 probability threshold; undefined values are null.
- Each run has a unique immutable directory and complete config/environment/Git/history/checkpoint/metric/confusion/prediction evidence. Summary rows derive from metrics JSON only.
- Long training requires CUDA and AMP; OOM fails with explicit guidance. No silent CPU fallback or lowered image resolution.

## PLAN / phase loop

- [x] Task 1: registry, wrapper, configs, tests and four-model CUDA memory preflight.
- [x] Task 2: binary metrics, latency/report writers and tests.
- [x] Task 3: shared AMP trainer/checkpoints/early stopping and synthetic overfit tests.
- [x] Task 4: CLI, run bundles, summary aggregation and orchestration tests.
- [x] Task 5: sequential real training, independent review, verification, documentation and handoff.

Executed results are in `artifacts/metrics/baseline_summary.csv`; full run, source and split verification is in `logs/phase5_*`. All four final bundles passed, 316 tests passed, and independent review found no Critical/Important issue.
