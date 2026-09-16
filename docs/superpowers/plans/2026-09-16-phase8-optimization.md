# Phase 8 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select and freeze the best validation-only binary HOSP-BoneNet configuration through ordered data screening, compact tuning, and supported ablations.

**Architecture:** A Phase 8 runner resolves the checksum-verified Phase 7 parent, uses only `fit_train` and `val`, resumes deterministic experiment stages, and publishes a validated Phase 8 alias. Core selection, transfer, and subset contracts live under `src/`; the script only orchestrates.

**Tech Stack:** Python 3.11, PyTorch, Optuna, pandas, PyYAML, pytest.

**Spec:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md` — Phase 8.

## Global Constraints

- Never read or write `dataset/Main` except through existing read-only dataset loaders.
- Never load the test manifest or test metrics during Phase 8.
- Selection metric is validation AUROC with deterministic secondary validation metrics.
- Binary labels are the only verified supervision; B3, B4, and B5 stay disabled.
- Preserve Phase 7 artifacts and only keep best/last checkpoints per new stage.

---

### Task 1: Ablation and training contracts

**Files:** Modify `src/bone_cancer_ai/models/hosp_bonenet/model.py`, `src/bone_cancer_ai/training/trainer.py`; test `tests/test_hosp_bonenet.py`, `tests/test_training_smoke.py`.

**Interfaces:** `HOSPConfig.spatial_path` accepts `auto|none|attention|gated`; `BinaryTrainer` accepts `label_smoothing` in `[0, 1)`.

- [ ] Add failing tests for local/global/gated feature routing and label smoothing.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement strict feature routing and smoothed binary training targets.
- [ ] Run focused and regression tests.

### Task 2: Reusable Phase 8 experiment primitives

**Files:** Create `src/bone_cancer_ai/optimization/__init__.py`, `src/bone_cancer_ai/optimization/phase8.py`; test `tests/test_phase8_optimization.py`.

**Interfaces:** Validate configuration, select deterministic stratified subsets, transfer compatible parent weights, rank validation candidates, reject test references, and atomically persist strict artifacts.

- [ ] Write failing tests for deterministic subsets, validation-only enforcement, transfer checks, and metric ranking.
- [ ] Run tests to verify failure for missing interfaces.
- [ ] Implement the smallest strict primitives.
- [ ] Run focused tests and ruff.

### Task 3: Resumable optimization runner

**Files:** Replace `scripts/11_tune_hosp_bonenet.py`; create `configs/optimization.yaml`; test `tests/test_phase8_cli.py`.

**Interfaces:** `run_phase8(config_path, force_new_run=False)` creates or resumes one fingerprinted bundle; CLI supports `--config` and `--force-new-run`.

- [ ] Add failing CLI contract tests using tiny injected dependencies.
- [ ] Implement data screening, ten Optuna trials, B0/B1/B2 ablations, full-fit confirmation, and incumbent comparison.
- [ ] Persist hashes, metrics, predictions, SQLite study, final YAML, status, and no test artifacts.
- [ ] Reuse completed stages and resume exact interrupted training from `last.pt`.

### Task 4: Execute, freeze, and hand off

**Files:** Create `artifacts/metrics/hosp_ablation.csv`; update concise status, guide, changelog, and handoff documents.

- [ ] Run the real CUDA Phase 8 workflow against verified cached data.
- [ ] Validate every bundle, selected checkpoint checksum, source snapshot, and absence of test access.
- [ ] Run full pytest, ruff, compileall, environment verification, and a checkpoint reload/forward smoke test.
- [ ] Request code review, fix findings, rerun verification, then commit the verified phase.
