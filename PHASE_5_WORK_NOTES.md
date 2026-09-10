# Phase 5 Standard Comparison Baselines Work Notes

**Status:** IDEA / PLAN / DESIGN complete; BUILD in progress.

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
- [ ] Task 4: CLI, run bundles, summary aggregation and orchestration tests.
- [ ] Task 5: sequential real training, independent review, verification, documentation and handoff.

The user explicitly authorized Phase 5 under the master directive and selected the recommended SOL route. The directive already approves the architecture, model list, settings, fairness contract and acceptance criteria, so implementation proceeds without a second design decision. The repository is clean on the existing non-main `phase1-scaffold` branch at commit `7282c0f`; work remains in the user-specified project root.
