# Phase 2 Dataset Audit and Leakage Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index every image in `dataset/Main`, verify its labels against local provenance, report quality and duplicate/leakage risks, and generate the four required manifests plus three required figures without changing source files.

**Architecture:** `schema.py` defines stable tabular contracts; `indexer.py` inventories and decodes each source image once; `audit.py` validates CSV and provenance evidence and produces reports/figures; `dedupe.py` builds deterministic exact/canonical/perceptual duplicate candidates and connected groups. Three thin scripts run the stages. Every output is written atomically beneath configured `workspace` or `artifacts`; only `dataset/Main` is indexed, while other local datasets are read solely as provenance evidence.

**Tech Stack:** Python 3.11, pathlib, dataclasses, hashlib, Pillow, imagehash, NumPy, Pandas, scikit-learn neighbor search, Matplotlib, openpyxl, PyTest.

**Spec:** `BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md`, especially Sections 0, 1, 5, 8, 9, 13 Phase 2, 16, 18, 19 and 23.

## Global Constraints

- `E:/Bone Cancer/dataset/Main` is read-only. No rename, deletion, rewrite, recompression or generated file is permitted there.
- Index only configured official splits `train`, `val`, `test`; do not merge other local datasets into Main.
- Source filenames and paths are metadata for integrity/provenance only and never future model inputs.
- Missing or unverified labels remain empty. Do not infer subtype, risk, box, mask or concepts from filename text.
- Label semantics require deterministic evidence. Current preliminary provenance mapping covers all 12,557 Main images: 0 maps to `normal`; 1 maps to `bone_tumour`, with benign/malignant subclasses in one local source.
- Near duplicates are candidates, not automatic exclusions. Phase 2 reports; Phase 3 decides clean manifests/splits.
- Test/validation data are audited for integrity but never used for preprocessing/training decisions based on labels.
- No model training, checkpoints or performance metrics in Phase 2.
- Use atomic output replacement in derived directories. Preserve Phase 1 source snapshots and verify source integrity again after all Phase 2 work.

---

### Task 1: Audit contracts and provenance reader

**Files:**
- Modify: `requirements/base.txt`
- Modify: `src/bone_cancer_ai/utils/config.py`
- Modify: `src/bone_cancer_ai/data/schema.py`
- Create: `src/bone_cancer_ai/data/provenance.py`
- Modify: `configs/data.yaml`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_config(path) -> dict`; configured project/source roots.
- Produces: `ManifestColumns`, `QualityStatus`, `canonical_source_id(name: str) -> str`, `record_id(split: str, image_id: str) -> str`, `load_main_labels(dataset_root: Path, splits: Sequence[str]) -> LabelInventory`, `load_local_provenance(config: dict, project_root: Path) -> pd.DataFrame`, `atomic_write_csv(frame, destination) -> None`.

- [x] Write tests using a temporary train/val/test fixture: duplicate YAML keys fail; exactly one label CSV per split is required; CSV root columns must be exactly `image_id,binary_label`; duplicate/blank IDs, non-integer labels and values outside `{0,1}` become explicit audit issues; canonical IDs remove only a terminal Roboflow `.rf.<32hex>` and export-format suffix; duplicate output IDs fail schema validation.
- [x] Run `pytest tests/test_dataset.py -q` and confirm failures are caused by missing Phase 2 interfaces.
- [x] Add `openpyxl` to `requirements/base.txt`, install it, and record it in the environment freeze after `pip check`.
- [x] Implement strict readers and typed schemas. Use a custom config section for provenance roots and audit thresholds; all external-to-Main provenance stays optional and read-only. Write derived CSVs to a sibling temporary file and replace only after serialization succeeds.
- [x] Run Task 1 tests and Ruff; retain the red/green output in `logs/`.

### Task 2: Deterministic image indexing and quality measurements

**Files:**
- Modify: `src/bone_cancer_ai/data/indexer.py`
- Modify: `src/bone_cancer_ai/data/quality.py`
- Modify: `scripts/01_index_dataset.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `LabelInventory`, source image paths, configured extensions and quality thresholds.
- Produces: `inspect_image(path, official_split, binary_label, thresholds) -> dict`, `index_dataset(config) -> tuple[pd.DataFrame, pd.DataFrame, dict]`; master fields include the directive-required columns plus a globally unique `record_id=official_split/image_id`, `pil_mode`, `is_rgb_encoded_grayscale`, `intensity_mean`, `intensity_std`, `entropy`, `dark_fraction`, `bright_fraction`, `decode_error`.

- [x] Add synthetic grayscale/RGB/RGBA/constant/truncated-image tests. Literal expectations verify dimensions/channels, SHA-256, perceptual hash length, canonical ID, RGB-encoded-grayscale detection, blank/near-blank status and decode-error containment. Test label rows missing images and images missing label rows.
- [x] Run the targeted tests and confirm expected missing-interface failures.
- [x] Implement one-pass Pillow decoding: open, capture mode/size, EXIF-transpose in memory for analysis, load pixels, compute SHA-256 from original bytes, pHash, grayscale thumbnail statistics and entropy. Catch per-file decode exceptions so one corrupt image cannot abort indexing. Use a bounded thread pool with deterministic final sorting.
- [x] Implement CLI flags `--config`, `--workers`, `--output`, `--quality-output`, `--summary-output`; validate every destination with `ProjectPaths.output_path`; never write under source; emit JSON summary and nonzero exit on structural label errors while still reporting them.
- [x] Run the test fixture end to end and Task 2 tests; inspect output headers and row counts.

### Task 3: Label/provenance audit and required figures

**Files:**
- Modify: `src/bone_cancer_ai/data/audit.py`
- Modify: `scripts/02_audit_dataset.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Consumes: dataset master, label issues, optional local provenance sources.
- Produces: `audit_labels(master, provenance, existing_issues=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]`, returning the enriched master, label-audit rows and summary; `save_audit_figures(master, output_dir, seed=42) -> list[Path]`; output `label_audit.csv`, updated `dataset_master.csv`, `dataset_class_distribution.png`, `dimension_distribution.png`, `sample_grid.png`, and `dataset_audit_summary.json`.

- [x] Add provenance fixtures for cancer/normal and benign/malignant/normal folders. Verify exact/canonical matches, conflict reporting and conservative semantics: 0=`normal`, 1=`bone_tumour`; `risk_label` may use benign/malignant only where directly matched; subtype remains empty unless the BTXRD spreadsheet explicitly identifies the Main image.
- [x] Add figure tests asserting readable PNGs with nonzero dimensions and deterministic sample selection from audited readable images; charts label filesystem/audit counts accurately.
- [x] Run tests to observe missing audit/figure implementations.
- [x] Implement provenance loading: Roboflow folder classes, three-class folder labels, BTXRD spreadsheet columns, annotation/image inventories, and nine-class YAML/label inventory. Record evidence coverage/conflicts in `label_audit.csv` and summary; never copy diagnosis tokens out of Main filenames.
- [x] Implement headless Matplotlib figures. The sample grid shows images with anonymized manifest row keys and verified binary class/split only, not diagnostic filename text or patient identifiers.
- [x] Run Task 3 tests and the audit CLI against the synthetic fixture.

### Task 4: Duplicate and leakage candidate grouping

**Files:**
- Modify: `src/bone_cancer_ai/data/dedupe.py`
- Modify: `scripts/03_find_duplicates.py`
- Modify: `tests/test_dedupe.py`

**Interfaces:**
- Consumes: master rows with globally unique `record_id`, `image_id`, `official_split`, `sha256`, `phash`, `canonical_source_id`, `binary_label`, dimensions.
- Produces: `hamming_distance(left, right) -> int`, `find_duplicate_edges(master, max_phash_distance=4) -> pd.DataFrame`, `build_duplicate_groups(master, edges) -> pd.DataFrame`, `summarize_leakage(groups) -> dict`; CSV records `duplicate_group`, `record_id`, `image_id`, `other_record_id`, `other_image_id`, `match_type`, `phash_distance`, `official_split`, `other_split`, `cross_split`, `label_conflict`.

- [x] Add hand-checked hash tests for Hamming distance, exact duplicate pairs, canonical pairs, perceptual candidates, transitive connected groups, deduplication of an edge matched by several rules, deterministic group IDs, label conflicts and cross-split flags. A perceptual candidate with same hash but implausibly different aspect ratio must remain reportable with its dimensions; it is never silently excluded.
- [x] Run the targeted tests and confirm missing behavior failures.
- [x] Implement exact/canonical grouping with grouped combinations. Generate perceptual candidates using a deterministic BK-tree over 64-bit pHashes to avoid quadratic all-pairs comparison. Merge evidence types per unordered pair, then use union-find for connected components.
- [x] Implement CLI `--manifest`, `--output`, `--summary-output`, `--max-phash-distance`; validate fields and safe output destinations; emit cross-split counts and label-conflict counts.
- [x] Run duplicate tests, including a moderate synthetic scaling case, then Ruff.

### Task 5: Full local execution, visual review, documentation and handoff

**Files:**
- Modify: `workspace/annotations/label_schema.yaml`
- Modify: `README.md`, `PROJECT_GUIDE.md`, `PROJECT_STATE.md`, `PHASE_HANDOFF.md`, `CHANGELOG.md`
- Modify: `src/bone_cancer_ai/data/README.md`, `workspace/manifests/README.md`, `scripts/README.md`, `dataset/README.md`, `configs/README.md`, `artifacts/README.md`, `artifacts/figures/README.md`
- Modify: `scripts/run_phase.ps1`, `.vscode/tasks.json`
- Create derived outputs: four required CSVs, four audit JSON summaries (index/audit/duplicate/verification), and three PNG figures.

**Interfaces:**
- Consumes: verified Phase 2 commands/results and unchanged source snapshot.
- Produces: durable audit evidence and latest-only Phase 3 handoff.

- [x] Run `01_index_dataset.py`, inspect every returned count/error, and stop to diagnose any structural failure rather than weakening validation.
- [x] Run `02_audit_dataset.py`, inspect provenance coverage, label meaning evidence, subtype/risk/spatial availability, image quality and required PNGs. Visually review the sample grid for orientation, padding absence, unreadable tiles, burned-in text risk and gross inversion; record findings without making clinical judgments.
- [x] Run `03_find_duplicates.py`, inspect exact/canonical/perceptual/cross-split groups and label conflicts. Phase 2 reports only; no deletion or source split edits.
- [x] Compare `dataset_master.csv` paths with source image inventory and Main label rows. Require every image indexed and every CSV label row mapped, with every mismatch reported. Re-hash all Main files and compare with Phase 1 snapshot to prove source preservation.
- [x] Request independent review of Phase 2 code, tests and generated summaries; investigate and fix every Critical/Important issue, then rerun affected commands.
- [x] Run fresh `pytest -q`, `ruff check .`, `python -m compileall -q src scripts app tools`, all three Phase 2 CLIs, manifest invariant verification and source-integrity comparison. Store command logs.
- [x] Update verified binary schema only when provenance coverage/conflicts justify it. Record any BTXRD subtype/risk/spatial evidence as available metadata without enabling training branches before Phase 3/4 contracts.
- [x] Update all relevant documentation with actual executed counts and limitations. Commit one verified Phase 2 milestone. Handoff must recommend SOL for Phase 3 and retain all model metrics as Not run.

## Plan self-review

- Spec coverage: every Phase 2 task, required output, acceptance criterion, source-safety rule and handoff field maps to a task above.
- Placeholder scan: no implementation placeholder instructions; terms such as Pending are reserved only for truthful future-phase status in final documentation.
- Type consistency: indexing produces the exact hashes, canonical IDs, labels and dimensions consumed by audit/dedupe; audit produces the reports/figures consumed by documentation; all CLI outputs pass through `ProjectPaths.output_path`.
- Scope: code stops at reporting. Clean-manifest exclusion and corrected splits remain Phase 3; annotation conversion remains Phase 4; training remains Phase 5+.

## Execution choice

The user explicitly requested immediate execution in this session, so use the inline executing-plans workflow. Independent subagent use is reserved for the required read-only code review, and any subagent name follows the repository's Mahabharata/Ramayana naming rule.
