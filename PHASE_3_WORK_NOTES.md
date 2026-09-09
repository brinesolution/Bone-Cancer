# Phase 3 Cleaning, Preprocessing and Data Loaders Implementation Plan

**Goal:** Turn the immutable audited source into leakage-safe manifest partitions and deterministic, partial-supervision-aware PyTorch batches without training a model.

**Authoritative requirements:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md`, especially Sections 0, 1, 5, 8, 9, 13 Phase 3, 18–23. Resume evidence comes from `PROJECT_STATE.md`, `PHASE_HANDOFF.md`, the Phase 2 generated manifests, data/config/workspace READMEs, and the existing src/script scaffolds.

## IDEA

Phase 2 proved that the official folders contain extensive cross-split duplication. Phase 3 must prevent those relationships from crossing fit/calibration/validation/test while preserving the source bytes and source assignments as audit metadata. It must also create one image pipeline for grayscale/RGB radiographs, preserve aspect ratio, align masks, expose missing-task masks, and make corrupt-runtime samples nonfatal.

Inputs already exist:

- immutable `dataset/Main` images and CSVs;
- `dataset_master.csv`, `duplicate_groups.csv`, `quality_report.csv`, and `label_audit.csv`;
- verified binary names and disabled optional-head gates in `label_schema.yaml`;
- Phase 2 source-integrity baseline and summaries.

Required outputs:

- all-record clean manifest plus exclusion/review evidence;
- `fit_train.csv`, `calibration.csv`, `val.csv`, and `test.csv`;
- reusable preprocessing, transforms, Dataset, sampler, and DataModule modules;
- smoke-tested batches from every final split and a visually reviewed batch grid;
- Phase 3 summaries, tests, documentation, state, and handoff.

Main failure risks are moving source files, allowing one connected duplicate group into two final splits, moving held-out records into training, using test labels to tune preprocessing/calibration, silently resolving label conflicts, misaligning masks, emitting filenames as predictive inputs, nondeterministic validation, or crashing workers on a changed/corrupt file.

## PLAN AND DESIGN

### Task 1 — Clean-manifest policy and contracts

**Files:** modify `configs/data.yaml`, `data/schema.py`, `data/quality.py`; create `tests/test_splits.py`; activate `scripts/04_build_clean_manifest.py`.

**Policy:**

- A record's `split_group_id` is its Phase 2 connected duplicate group or an opaque stable singleton ID.
- Held-out priority is test, then validation, then official training. In a cross-split group, keep records already in the highest-priority official split and exclude lower-priority copies. Never reassign an official test/validation record into training.
- Exclude unreadable/blank records, missing binary labels, and every group with an exact-byte binary-label conflict. Preserve source rows and write explicit reasons.
- Keep near-blank records as review candidates because both Phase 2 examples contain anatomy.
- Keep perceptual-only conflict groups together and flag them for review; pHash alone cannot rewrite a label or justify silent deletion.

**Contracts:** clean rows add `sample_id`, `duplicate_group`, `split_group_id`, `clean_pool`, `assigned_split`, `is_eligible`, `review_required`, `review_reason`, and `split_correction_reason`. `dataset_clean.csv` retains all rows; `exclusions.csv` records every excluded row. Summary JSON records original-to-clean movements, classes, reasons, and policy.

- [x] Add failing unit/CLI tests for group mapping, held-out priority, exact-conflict exclusions, quality exclusions, singleton IDs, deterministic ordering, and safe destinations.
- [x] Implement the contracts and clean-manifest builder.
- [x] Verify all source records appear exactly once as eligible or excluded and source paths are unchanged.

### Task 2 — Grouped internal calibration split

**Files:** add split functions to `data/samplers.py`; activate `scripts/06_make_internal_splits.py`; extend `tests/test_splits.py`.

**Design:** official held-out records remain validation/test. Only the clean `train_pool` is split into fit/calibration. Search deterministic `GroupShuffleSplit` candidates using training labels/groups only and select the candidate closest to configured total/class calibration targets. Never inspect validation/test labels for calibration assignment. Write full self-contained split manifests atomically.

- [x] Add failing tests for deterministic assignment, class representation, complete coverage, disjoint records, and zero group overlap.
- [x] Implement grouped split selection, validation, summaries, and four CSV outputs.
- [x] Execute against the full clean manifest and confirm no known group spans final splits.

### Task 3 — Aspect-preserving preprocessing and radiograph-safe transforms

**Files:** implement `data/preprocessing.py`, `data/transforms.py`; replace `tests/test_transforms.py`.

**Interfaces:** `decode_radiograph`, `resize_and_pad`, `load_binary_mask`, and `RadiographTransform`. Decode applies EXIF orientation in memory, optionally converts all radiographs to grayscale, replicates to RGB, pads centrally without stretching, resizes to 384, and applies ImageNet normalization. Validation is deterministic. Training may use only small affine rotation/translation/scale plus mild brightness/contrast, gamma, and low Gaussian noise; horizontal/vertical flips and crop are absent. Masks use nearest-neighbour geometry and shared random transforms.

- [x] Add failing tests for grayscale/RGB/RGBA decoding, exact aspect padding, output tensor shape/range, deterministic validation, seeded training sequences, and mask alignment.
- [x] Implement using the locally verified Albumentations 2.0.8 API and OpenCV interpolation constants.
- [x] Render synthetic checks and inspect the exact pad/mask geometry.

### Task 4 — Partial-label Dataset, samplers, and DataModule

**Files:** implement `data/dataset.py`, `data/samplers.py`, `data/datamodule.py`; extend `tests/test_transforms.py` and `tests/test_splits.py`.

**Batch contract:** every batch contains image `[B,3,384,384]`, binary/subtype/risk labels with `-1` for missing, concept tensor, zero placeholder mask `[B,1,384,384]`, nested `valid_task_mask`, and nonpredictive metadata. The top-level batch never exposes source path, image filename, or canonical ID. Current optional tasks remain invalid because their Phase 4 activation gates are closed.

Runtime decode failure returns a zero image with `sample_valid=false`, all task masks false, and a logged error type instead of pairing another image with the row's label.

- [x] Add failing tests for batch shape, missing/disabled labels, opaque metadata, no filename leakage, corrupt-runtime containment, deterministic loader order, and optional weighted sampling.
- [x] Implement the Dataset, inverse-frequency sampler, worker seeding, and split-aware DataModule.
- [x] Load one real batch from every final split and confirm labels/masks/shapes.

### Task 5 — Optional deterministic cache and visual QA

**Files:** activate `scripts/05_build_cache.py`; update cache modules/READMEs; generate `artifacts/figures/preprocessing_batch_grid.png`.

**Design:** cache creation is optional and off by default. It writes only aspect-padded derived PNGs plus a content/config fingerprint index under `workspace/cache/resized`; filenames are opaque sample IDs. Phase 3 verifies the cache implementation on a bounded sample rather than duplicating the complete source without need.

- [x] Add a temporary-fixture cache test for safe paths, deterministic names/content, source hash traceability, and idempotent rebuild.
- [x] Implement and smoke-test a bounded cache build.
- [x] Generate and visually inspect a grid drawn from all four real loaders for anatomy, padding, orientation, channels, and mask/label display.

### Task 6 — Review, verification, documentation, and handoff

**Files:** update relevant READMEs, root guide/state/changelog/handoff, PowerShell/VS Code tasks, and this plan.

- [x] Request an independent read-only review of code, tests, policies, and full summaries; fix every Critical/Important finding.
- [x] Run fresh full CLIs, `pytest -q`, Ruff, compileall, pip check, manifest invariants, loader smoke tests, visual review, and Phase 1 source-snapshot comparison.
- [x] Update documentation with executed split/exclusion counts and limitations; retain all model metrics as Not run.
- [x] Commit the verified Phase 3 milestone and stop before Phase 4.

## Acceptance mapping

- Batches from every split: Task 4 full smoke test.
- Corrupt image cannot crash an epoch: clean exclusion plus runtime invalid-sample fallback.
- Exclusions logged: Task 1 all-record manifest, exclusions CSV, summary.
- No duplicate group overlap: Tasks 1–2 invariant validation against all four files.
- Tests pass: Task 6 fresh repository suite.
- Anatomically reasonable visual batch grid: Task 5 local image inspection with findings documented.

## Scope boundary

No annotation UI, annotation conversion, baseline/model implementation, training, tuning, evaluation metrics, or clinical claims belong to Phase 3. Phase 4 will validate spatial/subtype/risk annotations; Phase 5 begins baseline training only after all Phase 3 gates pass.
