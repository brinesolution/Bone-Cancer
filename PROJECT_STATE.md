# Project state

Updated: 2026-09-09 (Asia/Calcutta).

**PHASE 2 COMPLETE — Dataset Audit, Label Verification and Leakage Analysis.**
Completed phases: 1–2. Phases 3–10: Pending. No model training, checkpoints or performance metrics exist.

Authoritative specification: [master directive](BONE_CANCER_10_PHASE_BUILD_DIRECTIVE.md). Phase 2 engineering record: [PHASE_2_WORK_NOTES.md](PHASE_2_WORK_NOTES.md). Latest-only transition: [PHASE_HANDOFF.md](PHASE_HANDOFF.md).

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

The two near-blank flags retain visible anatomy and are review candidates, not automatic exclusions. Visual review shows mixed stored orientation, photographic film captures, variable framing/exposure and burned-in text/markers. Phase 3 must prevent shortcuts from borders/text and validate orientation/preprocessing choices.

## Partial supervision inventory

Configured read-only provenance produced 18,170 evidence rows. Exactly 18,160 match a Main image byte-for-byte within their canonical group. Ten transformed subtype-detection images are nonexact and are excluded from direct evidence consumption; exact BTXRD records provide the agreeing metadata.

- Risk metadata: 1,867 tumour images (1,525 benign; 342 malignant).
- Unambiguous subtype metadata: 1,866 images across nine source classes.
- Source annotations: 1,867 images with boxes and polygons.
- Concepts: unavailable.
- `IMG001276.jpeg` activates two workbook subtype columns. Its subtype remains blank and the conflict is retained in `label_audit.csv` for Phase 4.

Subtype, risk, box and mask branches remain disabled. Availability does not establish spatial alignment or clinical/expert verification; Phase 4 owns annotation conversion, overlays and activation.

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

Evidence totals are 1,080 SHA-256 matches, 11,016 canonical-ID matches and 4,300 pHash candidates; one edge can carry multiple evidence types. Perceptual matches at Hamming distance ≤4 are candidates, while SHA matches establish identical bytes. Phase 3 must resolve label conflicts and create manifest-based corrected splits so no known connected group spans fit/calibration/validation/test. Never move or rename source files.

## Implemented Phase 2 components

- Strict source CSV contracts, stable record/canonical IDs and atomic CSV/JSON writers.
- Threaded one-pass image indexing with SHA-256, 64-bit pHash, dimensions, channels, pixel statistics and contained decode errors.
- Local provenance readers for binary folders, three-class folders, BTXRD workbook/images/LabelMe inventory and nine-class detection export.
- Content-gated, idempotent label audit that clears stale enrichment and preserves source conflicts.
- Deterministic class/dimension figures and anonymized manifest-title sample grid; source pixels remain unchanged.
- BK-tree pHash search, grouped exact/canonical comparisons, evidence merging and union-find duplicate components.
- Safe Phase 2 CLI sequence in `scripts/run_phase.ps1 -Action phase2` and VS Code.

Generated local evidence:

- `workspace/manifests/dataset_master.csv`
- `workspace/manifests/quality_report.csv`
- `workspace/manifests/label_audit.csv`
- `workspace/manifests/duplicate_groups.csv`
- `artifacts/figures/dataset_class_distribution.png`
- `artifacts/figures/dimension_distribution.png`
- `artifacts/figures/sample_grid.png`
- `logs/dataset_index_summary.json`, `dataset_audit_summary.json`, `duplicate_summary.json`, `phase2_verification.json`, `source_after_phase2.json`

These local data-derived files are Git-ignored and must be regenerated through the scripts rather than edited manually.

## Verification

Executed Phase 2 CLIs completed with exit 0. Final invariant verification reports `ok: true`: master paths equal the full source-image inventory, manifest IDs equal all label rows, required columns/figures exist, and the Phase 2 source snapshot exactly equals the Phase 1 baseline: 12,560 files, 1,120,418,007 bytes, identical paths/sizes/mtimes/SHA-256.

Latest checks:

```powershell
python -m pytest -q
ruff check .
python -m compileall -q src scripts app tools
python -m pip check
python scripts/01_index_dataset.py
python scripts/02_audit_dataset.py
python scripts/03_find_duplicates.py
```

Independent Phase 2 review found no remaining Critical, Important or Minor code findings after fixes. It also compared the BK-tree result with brute-force Hamming pairs. No training or inference ran.

## Next action

Use **SOL** for Phase 3 — Cleaning, Manifest-Based Splits, Preprocessing and Loaders. Automatic model switching is unavailable, so manually select SOL before continuing. Start from the generated manifests and keep duplicate connected components intact across new partitions. Do not rerun environment creation, dependency installation or scaffold generation. Rerun Phase 2 only if its source/configuration changes, and never alter `dataset/Main`.
