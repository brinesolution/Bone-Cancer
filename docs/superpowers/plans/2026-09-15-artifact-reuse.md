# Artifact Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verified checkpoint aliases, idempotent run reuse/resume, and a production preprocessing cache.

**Constraints:** Preserve immutable experiment runs and `dataset/Main`; keep test labels isolated; never commit large checkpoints or cached images.

## Task 1 — Checkpoint registry

- [ ] Add failing tests for safe promotion, alias resolution, idempotence and tamper rejection.
- [ ] Implement `src/bone_cancer_ai/artifacts/registry.py` with atomic JSON updates and hard-link/copy promotion.
- [ ] Add a compact artifact-folder README and Git allow-list for registry metadata.
- [ ] Run focused tests and Ruff.

## Task 2 — HOSP rerun policy

- [ ] Add failing CLI tests for exact completed-run reuse, automatic failed-run resume and `--force-new-run`.
- [ ] Integrate run discovery, bundle validation and selected-checkpoint registration into script 10.
- [ ] Preserve explicit `--resume-run` and reject conflicting flags.
- [ ] Run HOSP CLI/training tests.

## Task 3 — Baseline and future reuse

- [ ] Add tests for registering the four selected Phase 5 checkpoints.
- [ ] Add a registry refresh/validation command and wire successful future baseline runs to the common API.
- [ ] Configure inference to use the selected HOSP alias without performing Phase 9 evaluation.
- [ ] Run baseline and inference-focused tests.

## Task 4 — Validated preprocessing cache

- [ ] Add failing tests for idempotent cache reuse, tamper/staleness rejection and cached/uncached tensor equivalence.
- [ ] Extend the cache builder for selected split manifests and prior-index reuse.
- [ ] Add strict DataModule cache loading for fit_train/val before dynamic augmentation/normalization.
- [ ] Build the complete fit_train/val cache and verify every entry.

## Task 5 — Final verification and handoff

- [ ] Refresh the registry from completed Phase 5/7 artifacts.
- [ ] Run full pytest, Ruff, compileall, pip check, bundle validation and source/split preservation.
- [ ] Update concise state, guide, handoff and changelog entries.
- [ ] Review, fix findings, and commit verified milestones.
