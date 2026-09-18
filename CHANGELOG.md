# Changelog

2026-09-18 — Added direct web-app launcher and read-only model-results dashboard.

2026-09-17 — Completed one-time frozen Phase 9 test comparison, bootstrap intervals, conformal uncertainty, Grad-CAM/lesion galleries and technical visual QA.
2026-09-17 — Completed Phase 10 frozen inference API, Streamlit demo, CLI workflows and final QA wiring.

2026-09-17 — Completed validation-only Phase 8 data screening, 10-trial tuning and supported HOSP ablations; froze B2 at validation AUROC 0.958522 with reusable checksum-verified artifacts.

2026-09-16 — Added checksum-backed selected-checkpoint aliases and validated Phase 5/7 rerun reuse.
2026-09-16 — Added the versioned 6,324-image fit/validation cache with strict integrity checks and dynamic augmentation.

2026-09-14 — Completed staged HOSP-BoneNet training; selected Stage D epoch 6 by validation AUROC 0.958355 with exact checkpoint reload and no test access.

2026-09-12 — Implemented and verified configuration-gated HOSP-BoneNet, masked multitask losses, CUDA AMP checks, and a real 16-image overfit smoke test.

2026-09-12 — Completed four shared-pipeline seed-42 baselines with real checkpoints/metrics, strict artifact validation, an evidenced ConvNeXt 4×2 memory exception, and preserved source/splits.

## 2026-09-10 — Phase 4 verified

- Added local Streamlit metadata review with unknown-aware controls, actual reviewer/status/notes, autosave, optimistic SQLite revisions, navigation, progress and master CSV export.
- Integrated pinned LabelMe 6.3.1 using opaque exact-byte copies, source bindings and immutable reviewer-owned spatial revisions.
- Added native-coordinate polygon unions, xyxy boxes, overlays and strict label/geometry/hash/provenance validation. Boxes never fabricate segmentation masks.
- Inspected all 1,867 BTXRD JSONs; accepted 1,825 unique mask/box records with zero validation errors, quarantined 16 conflicting source drawings and reported 26 unmatched sources.
- Fixed conflicting exact-image annotation overwrite, duplicate index rows, false-success validation exits, audit/overlay drift checks and excessive preflight mask retention. Independent review closed all material findings.
- Passed 126 tests, Ruff, compileall and pip check; native LabelMe and Streamlit AppTest workflows passed. Inspected the 12-image overlay grid and three full overlays. Browser automation was unavailable; no browser visual QA is claimed.
- Source snapshot and all four split counts/fingerprints remain unchanged. No human/clinical verification, training or performance metrics were fabricated. Optional supervision release remains pending.

## 2026-09-09 — Phase 3 verified

- Added an all-record clean manifest with opaque sample/group IDs, explicit exclusions/review flags, and held-out priority that never moves validation/test records into training.
- Logged 3,490 lower-priority cross-split duplicate exclusions and four records from two exact-byte label-conflict groups; kept pHash-only conflicts together for review.
- Created deterministic connected-group partitions: fit 4,871; calibration 857; validation 1,453; test 1,882.
- Added a full-bundle generation fingerprint and fail-closed DataModule checks for stale/mixed files, record/sample/group overlap, eligibility, held-out movement and paths outside `dataset/Main`.
- Implemented safe grayscale/RGB/RGBA decoding, aspect-preserving 384-pixel padding, aligned masks, deterministic evaluation transforms and conservative fit-only augmentation without flips/crops.
- Implemented partial-label Dataset/DataModule contracts, corrupt-runtime invalid samples, seeded shuffle/weighted sampling and no source filename/path predictive fields.
- Added an optional traceable image cache; verified 16 opaque PNGs with matching hashes and no source edits.
- Loaded real batches from every split and visually verified recognizable anatomy, grayscale consistency and expected padding.
- Preserved the exact source snapshot: 12,560 files and 1,120,418,007 bytes unchanged. No model training or performance results were produced.

## 2026-09-09 — Phase 2 verified

- Indexed all 12,557 Main images and mapped every label row without modifying source files.
- Verified 0=`normal` and 1=`bone_tumour` using canonical metadata gated by SHA-256 content anchors.
- Added strict schema/label readers, threaded image/quality indexing, provenance audit, required figures and atomic derived outputs.
- Added deterministic exact/canonical/pHash duplicate evidence and connected-group leakage reports.
- Reported 2,045 duplicate groups and 6,112 cross-split edges; preserved six pair label conflicts for Phase 3 resolution.
- Inventoried partial BTXRD subtype/risk/box/polygon metadata while keeping optional branches disabled.
- Fixed audit rerun accumulation/stale-value behavior and excluded nonexact provenance rows from direct evidence.
- Added openpyxl to declared/frozen dependencies; passed 59 tests, Ruff, compileall, pip check, independent review and full source-preservation comparison.
- Generated no models, checkpoints or performance/clinical results.

## 2026-09-09 — Phase 1 verified

- Read directive; inventoried prerequisites and source files.
- Captured source hashes, sizes and modification times.
- Created modular scaffold, inactive commands, header-only tables and unverified schema.
- Created Python 3.11 environment; verified a CUDA tensor operation.
- Added safe YAML/output-path, JSON logging, RNG/device, source hashing and environment utilities.
- Verified all 40 declared dependency imports; recorded 237 installed versions after checks.
- Used maintained grad-cam and one opencv-python-headless distribution to avoid cv2 namespace conflicts.
- Passed 31 tests, Ruff, compileall, pip check, lock dry-run and PowerShell workflow.
- Matched all 12,560 source files by path, SHA-256, size and mtime; all 201 directive paths and authored-folder READMEs present.
- Completed independent review and addressed its two minor test suggestions.
- Updated state/guide/handoff; stopped before Phase 2 for manual SOL selection. All model results remain Not run.
2026-09-18 — Added direct web-app launcher and read-only model-results dashboard.
