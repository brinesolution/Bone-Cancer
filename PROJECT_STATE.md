# Project state

Updated: 2026-09-10 (Asia/Calcutta).

**PHASE 4 COMPLETE — Annotation and Labeling System.**
Completed phases: 1–4. Phases 5–10 Pending. No model training, checkpoints or performance metrics exist.

Phase 4 delivered the local metadata app, native LabelMe integration, immutable revisions, exact-image/conflict-aware source conversion, masks/boxes/overlays, master CSVs and validation. The full suite passed 126 tests. The accepted source generation contains 1,825 unique images with masks and boxes; all 1,825 reconvert with zero validation issues. Sixteen conflicting source JSONs and 26 unmatched JSONs are excluded explicitly. Human project reviews remain zero and clinical verification remains Unavailable. See PHASE_4_WORK_NOTES.md and PHASE_HANDOFF.md for evidence and limits.

Authoritative specification: [master directive](BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md). Engineering records: [Phase 3](PHASE_3_WORK_NOTES.md), [Phase 4](PHASE_4_WORK_NOTES.md). Latest-only transition: [PHASE_HANDOFF.md](PHASE_HANDOFF.md).

## Environment

Use the existing project `.venv`: Python 3.11.15, torch 2.14.0+cu130, torchvision 0.29.0+cu130, CUDA available, NVIDIA GeForce RTX 4070 Laptop GPU. Phase 1 commit: `1c7e79b`. Phase 2 added openpyxl 3.1.5 plus et_xmlfile 2.0.0 to the verified Windows freeze; `pip check` passes. Do not recreate the environment on ordinary resume.

## Audited Main dataset

Source: `E:\Bone Cancer\dataset\Main`, immutable.

| Official split | Class 0: normal | Class 1: bone_tumour | Images / CSV rows |
|---|---:|---:|---:|
| train | 4,779 | 4,010 | 8,789 |
| val | 1,024 | 860 | 1,884 |
| test | 1,024 | 860 | 1,884 |
| **Total** | **6,827** | **5,730** | **12,557** |

All 12,557 images have one real CSV row; every CSV row maps to one image. Manifest record IDs are unique, all files decode, and no missing/extra label rows exist. The three source CSVs contain exactly `image_id,binary_label`.

Binary semantics are verified with independent local metadata plus image-content evidence: every Main canonical group has at least one exact SHA-256 provenance anchor. Class 0 means `normal`; class 1 conservatively means `bone_tumour`, not malignant. Main diagnostic filename words were not parsed into labels and must never become predictive inputs.

| Property | Executed result |
|---|---|
| Extensions | 8,838 `.jpg`; 3,719 `.jpeg` |
| PIL modes / channels | 9,662 RGB/3-channel; 2,895 L/1-channel |
| RGB-encoded grayscale | 9,109 images |
| Dimensions | width 153–3,594, median 640; height 311–4,881, median 640; 3,149 shape pairs |
| Quality | 12,555 readable; 2 near-blank/low-exposure; 0 blank; 0 unreadable |
| Patient/study ID | Unavailable; canonical/duplicate group is the strongest grouping identifier |

The two near-blank flags retain visible anatomy and are review candidates, not automatic exclusions. Visual review shows mixed stored orientation, photographic film captures, variable framing/exposure and burned-in text/markers. Phase 3 preserved anatomy/aspect ratio and documented these shortcut risks for later evaluation.

## Partial supervision inventory

Configured read-only provenance produced 18,170 evidence rows. Exactly 18,160 match a Main image byte-for-byte within their canonical group. Ten transformed subtype-detection images are nonexact and are excluded from direct evidence consumption; exact BTXRD records provide the agreeing metadata.

- Risk metadata: 1,867 tumour images (1,525 benign; 342 malignant).
- Unambiguous subtype metadata: 1,866 images across nine source classes.
- Source annotations: 1,867 images with boxes and polygons.
- Concepts: unavailable.
- `IMG001276.jpeg` activates two workbook subtype columns. Its subtype remains blank and the conflict is retained in `label_audit.csv` for Phase 4.

Subtype, risk, box and mask branches remain disabled. Phase 4 mechanically validated a conflict-free spatial subset and representative overlays, but did not release optional labels into the fingerprinted training manifests. A deliberate validated supervision release must join accepted annotations without changing existing partition assignments; unsupported branches and unresolved labels must stay masked. Clinical/expert verification remains Unavailable.

## Duplicate and leakage findings

`duplicate_groups.csv` reports pair evidence and stable connected groups. It does not delete files or change official folders.

| Finding | Count |
|---|---:|
| Connected duplicate candidate groups | 2,045 |
| Images in candidate groups | 6,968 |
| Pair edges | 13,225 |
| Cross-split groups | 1,215 |
| Cross-split pair edges | 6,112 |
| Cross-split exact-byte edges | 508 |
| Cross-split canonical-ID edges | 5,061 |
| Cross-split perceptual-hash edges | 2,046 |
| Pair edges with label conflicts | 6 |
| Exact-byte label-conflict edges | 2 |

Evidence totals are 1,080 SHA-256 matches, 11,016 canonical-ID matches and 4,300 pHash candidates; one edge can carry multiple evidence types. Perceptual matches at Hamming distance ≤4 are candidates, while SHA matches establish identical bytes. Phase 3 resolved training eligibility and partition leakage through derived manifests so no known connected group spans fit/calibration/validation/test. Never move or rename source files.

## Clean manifest and final partitions

Phase 3 keeps every source record in `dataset_clean.csv` and records eligibility rather than modifying source folders. Connected groups retain records only in the highest-priority official partition present: test, then validation, then training. This excluded 3,490 lower-priority cross-split copies. Four records across two exact-byte label-conflict groups were excluded. The 37 records in three pHash-only conflict groups remain together and are flagged for review; the two near-blank images also remain eligible and flagged.

| Final status/split | Class 0: normal | Class 1: bone_tumour | Records |
|---|---:|---:|---:|
| fit_train | 2,197 | 2,674 | 4,871 |
| calibration | 385 | 472 | 857 |
| validation | 711 | 742 | 1,453 |
| test | 1,023 | 859 | 1,882 |
| **Eligible total** | **4,316** | **4,747** | **9,063** |
| excluded | 2,511 | 983 | 3,494 |

The calibration set is 14.9616% of the 5,728-record clean official-training pool and contains 703 of its 4,681 connected groups. Selection uses training groups/labels only. Invariant verification reports zero record overlap, zero connected-group overlap, zero unassigned eligible rows, and zero official validation/test rows moved into fit/calibration. Patient/study IDs remain unavailable.

## Preprocessing and batch contract

All inputs are decoded in memory with EXIF orientation, converted to grayscale-replicated RGB, resized with one aspect-preserving scale, centrally padded to 384×384, and ImageNet-normalized. Fit-only augmentation uses mild affine/intensity/noise transforms; flips and crops are absent. Calibration/validation/test transforms are deterministic. Image and mask geometry share interpolation-aware transforms.

Every batch includes image, binary/subtype/risk targets, concept targets, mask, nested `valid_task_mask`, and opaque metadata. Source filenames/paths/canonical IDs are excluded from predictive batch fields. Current real batches mark binary/sample valid and subtype/risk/concept/mask tasks invalid. Runtime corrupt images return a zero image and all-false validity masks with a logged error type. Real batches of eight loaded successfully from all four splits with image shape `[8,3,384,384]` and mask shape `[8,1,384,384]`.

The optional cache is off by default. A bounded smoke run generated 16 deterministic opaque PNGs under `workspace/cache/resized`, each with verified source/cache hashes and preprocessing fingerprint. Visual review of `preprocessing_batch_grid.png` found recognizable anatomy, consistent grayscale and expected black padding across all splits; mixed orientation, burned-in markers and film/capture artifacts remain documented shortcut risks.

## Implemented through Phase 4

- Strict source CSV contracts, stable record/canonical IDs and atomic CSV/JSON writers.
- Threaded one-pass image indexing with SHA-256, 64-bit pHash, dimensions, channels, pixel statistics and contained decode errors.
- Local provenance readers for binary folders, three-class folders, BTXRD workbook/images/LabelMe inventory and nine-class detection export.
- Content-gated, idempotent label audit that clears stale enrichment and preserves source conflicts.
- Deterministic class/dimension figures and anonymized manifest-title sample grid; source pixels remain unchanged.
- BK-tree pHash search, grouped exact/canonical comparisons, evidence merging and union-find duplicate components.
- Safe Phase 2 CLI sequence in `scripts/run_phase.ps1 -Action phase2` and VS Code.
- All-record clean-manifest policy with stable opaque sample/singleton IDs, explicit exclusion/review reasons and held-out priority.
- Group-respecting deterministic internal calibration split and optional seeded inverse-frequency sampling.
- Safe radiograph decoder, aspect-preserving image/mask geometry, deterministic validation transforms and conservative fit transforms.
- Partial-supervision `BoneCancerDataset`, four-split `BoneCancerDataModule`, corrupt-decode containment and real-batch visual QA.
- Optional content/config-fingerprinted resized cache and ordered Phase 3 PowerShell/VS Code command.
- Unknown-aware Streamlit metadata app, transactional SQLite history, optimistic revisions, CSV export and progress counts.
- Local LabelMe 6.3.1 staging, source hash bindings, polygon/box conversion, immutable spatial reviewer history and overlays.
- Exact-image duplicate source-annotation agreement checks, conflict quarantine, unique master indexes and full artifact/provenance revalidation.

Generated local evidence:

- `workspace/manifests/dataset_master.csv`
- `workspace/manifests/quality_report.csv`
- `workspace/manifests/label_audit.csv`
- `workspace/manifests/duplicate_groups.csv`
- `workspace/manifests/dataset_clean.csv`
- `workspace/manifests/exclusions.csv`
- `workspace/splits/fit_train.csv`, `calibration.csv`, `val.csv`, `test.csv`
- `workspace/cache/resized/cache_index.csv` and 16 smoke PNGs
- `artifacts/figures/dataset_class_distribution.png`
- `artifacts/figures/dimension_distribution.png`
- `artifacts/figures/sample_grid.png`
- `artifacts/figures/preprocessing_batch_grid.png`
- `logs/dataset_index_summary.json`, `dataset_audit_summary.json`, `duplicate_summary.json`, `phase2_verification.json`, `source_after_phase2.json`
- `logs/clean_manifest_summary.json`, `internal_split_summary.json`, `cache_summary.json`, `phase3_data_verification.json`, `source_after_phase3.json`
- `workspace/annotations/source_annotations.csv`, `source_annotation_issues.csv`, `image_labels.csv`, `reviews.sqlite3`
- Accepted annotation generation `workspace/annotations/source_imports/c204d1f6873644948e4f355f26b399ae/`
- `artifacts/figures/phase4_annotation_overlay_grid.png`
- `logs/phase4_annotation_validation.json`, `phase4_source_import.json`, `phase4_pytest.txt`, `source_after_phase4.json`, `phase4_split_preservation.json`

These local data-derived files are Git-ignored and must be regenerated through the scripts rather than edited manually.

## Phase 3 verification (retained evidence)

Executed Phase 3 CLIs completed with exit 0. Final data invariant verification reports `ok: true`: clean eligibility/exclusions cover all 12,557 records; the four split manifests cover all eligible rows exactly once; record/group overlap and held-out-to-training movement are zero; all manifest paths stay under the source; and cache hashes/locations are safe. The Phase 3 source snapshot exactly equals the Phase 2 baseline: 12,560 files, 1,120,418,007 bytes, identical paths/sizes/mtimes/SHA-256.

Phase 3 executed commands:

```powershell
python -m pytest -q
ruff check .
python -m compileall -q src scripts app tools
python -m pip check
./scripts/run_phase.ps1 -Action phase3
python scripts/05_build_cache.py --limit 16
```

The final full suite passed 89 tests; all 30 focused Phase 3 tests passed. They cover split policy, no filename leakage, shapes, missing/malformed labels, aligned masks, deterministic transforms/loaders/cache, group overlap, stale/mixed generation rejection, outside-source path rejection, and a complete mixed valid/corrupt loader epoch. Ruff, compileall and pip check passed. Independent review found one Important split-boundary issue and two Minor cache/test gaps; all were fixed and reverified. Visual batch QA passed. No training or inference ran; all model metrics remain `Not run`.

## Phase 4 verification and coverage

`python scripts/08_validate_annotations.py --import-source` exited 0 after inspecting all 1,867 BTXRD JSONs. The accepted index has 1,825 unique sample IDs, 1,825 masks and 1,825 box records; full reconversion reports valid=true, zero issues. Sixteen JSONs across eight exact-image groups have conflicting drawings and are quarantined; 26 sources lack an eligible byte-exact match. One accepted record still has unknown subtype. Source clean-pool coverage is train_pool 1,284, validation 281, test 260; train_pool is not a final fit assignment and must still join the preserved fit/calibration manifests before training.

All 126 tests passed, including synthetic Streamlit autosave/navigation/unknown handling, native LabelMe load/edit/save/reload, stale revisions, geometry, duplicate disagreement, provenance drift and invalid-artifact exit codes. Ruff, compileall and pip check passed. Independent review found no remaining material issues after fixes. Three native-resolution overlays plus the 12-image grid were inspected for coordinate consistency; this is technical QA, not clinical verification. The local server and real-data AppTest loaded successfully. Browser automation reported no browser available; interactive browser visual QA is not claimed.

The Phase 4 source snapshot equals Phase 3 exactly: 12,560 files, 1,120,418,007 bytes, unchanged paths/sizes/mtimes/SHA-256. DataModule setup revalidated all split fingerprints and preserved fit 4,871 / calibration 857 / validation 1,453 / test 1,882. No source labels, source images or training split files were rewritten.

The initial rejected generation `dd8bf4e772474d6b89ee7b2d0e69b541` is retained only as failed-run evidence in `phase4_initial_import_rejected.json`; do not consume it. Latest source indexes point to the accepted generation above. Human annotation coverage remains zero: synthetic test reviews are isolated under logs. The 16 conflicting source drawings, ambiguous subtype, pHash-only binary conflicts and clinical verification still require qualified human input; they are not fabricated or silently resolved.

## Next action

Manually select **SOL** before Phase 5 — Standard Comparison Baselines. Automatic primary-model switching is unavailable. Use one configurable implementation for DenseNet121, EfficientNet-B3, ConvNeXt-Tiny and Swin-Tiny, preserving existing verified splits, binary semantics, preprocessing and fair training budgets. Verify CUDA/model forward-backward before training; Phase 5 has not started. Do not rerun environment setup, dataset audit, split generation or source conversion unless inputs/configuration change. Launch annotation review with `.\.venv\Scripts\python.exe scripts/07_launch_annotation_tool.py` as needed. Never alter `dataset/Main`.
