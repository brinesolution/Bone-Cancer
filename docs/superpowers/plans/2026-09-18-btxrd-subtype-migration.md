# BTXRD Subtype Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add BTXRD subtype metadata as validated, masked supervision while preserving the frozen binary pipeline and immutable source datasets.

**Architecture:** Phase A creates a read-only audit, exact-hash linkage, conflict report, and deterministic grouped manifests under `workspace/btxrd/`. Phase B consumes only those validated artifacts through a separate HOSP configuration and run namespace, enabling the existing optional subtype head with masked loss and fresh calibration/evaluation.

**Tech Stack:** Python 3.11, pandas, openpyxl, Pillow, imagehash, scikit-learn, PyTorch, PyYAML, PyTest.

**Spec:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md`; BTXRD official release: https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398

## Global constraints

- `dataset/Main` and `dataset/BTXRD` are read-only; all outputs are derived under `workspace/`, `experiments/`, `artifacts/`, or `logs/`.
- Exact SHA-256 and perceptual-hash evidence is recorded before any subtype training.
- BTXRD is linked metadata, not silently appended to the old binary image pool.
- Normal rows keep binary label 0 and an invalid/blank subtype target; only validated subtype rows contribute subtype loss.
- Existing Phase 5–10 checkpoints, test metrics, aliases, and manifests remain frozen. No old binary test metric is reused for BTXRD.
- No fabricated metrics or subtype semantics; ambiguous/conflicting rows remain masked and reported.

## Phase A — provenance audit and manifests

1. Add `src/bone_cancer_ai/data/btxrd_migration.py` with required-column validation, safe image resolution, SHA-256/pHash inventory, subtype conflict detection, exact linkage to the Main manifest, and deterministic grouped split assignment.
2. Add `scripts/16_prepare_btxrd.py` and `configs/btxrd_data.yaml`; write only derived `workspace/btxrd/` CSV/JSON artifacts and a source-preservation summary.
3. Add focused tests for schema validation, read-only behavior, exact linkage, conflict masking, duplicate groups, deterministic splits, and rerun idempotence.
4. Run the audit on the local 3,746-image BTXRD release; acceptance requires complete image/metadata accounting, explicit conflict counts, no source changes, and disjoint grouped fit/calibration/validation/test manifests.

## Phase B — subtype-enabled HOSP run and evaluation

1. Add `configs/hosp_btxrd_subtype.yaml` and a separate runner that consumes only Phase A manifests, enables the existing masked subtype head, and records a new fingerprinted experiment namespace.
2. Validate binary/subtype masks, group-disjoint splits, checkpoint reload, CUDA forward/backward, and masked subtype loss before full training.
3. Train/reuse only within the new BTXRD run; calibrate binary and subtype uncertainty on the new calibration split and evaluate once on its held-out test split.
4. Publish fresh metrics and checkpoint aliases only after checksum validation; do not alter the frozen production alias. Document unavailable subtype classes or unresolved conflicts explicitly.

## Verification and handoff

- Focused tests → full `pytest -q` → Ruff → compileall → real BTXRD audit → source-tree hash comparison.
- Phase A may be marked complete only when its artifacts and checks pass. Phase B remains pending until a validated audit exists.
- Keep `PROJECT_STATE.md` and `PHASE_HANDOFF.md` concise; report actual counts and use `Not run`/`Pending` for training metrics until executed.
