# src/bone_cancer_ai/data

## Purpose
Manifest validation, indexing, quality, deduplication and reusable image loading.

## Files
`schema.py` defines manifest/issues contracts and strict CSV readers; `indexer.py` decodes and hashes; `quality.py` measures grayscale thumbnails; `provenance.py` reads separate local metadata; `audit.py` verifies/enriches labels and renders figures; `dedupe.py` detects and groups duplicate evidence. Preprocessing, transforms, datasets, samplers and datamodule remain Phase 3 scaffolds.

## Inputs
Immutable Main images/CSVs, configured provenance roots and audit thresholds.

## Outputs
Audited manifests, summaries and figures. Safe tensor batches are Phase 3 outputs.

## Main Libraries
pathlib, hashlib, Pillow, ImageHash, NumPy, Pandas, Matplotlib, PyYAML and openpyxl.

## Important Functions / Classes
`load_main_labels`, `inspect_image`, `index_dataset`, `analyze_pixels`, `load_local_provenance`, `audit_labels`, `save_audit_figures`, `find_duplicate_edges`, `build_duplicate_groups` and `summarize_leakage` are active. `record_id=split/image_id` remains unique when source names repeat.

## Theory
Byte hashes prove exact identity; pHash Hamming distance screens visually similar images; union-find turns pair evidence into stable groups. Quality thresholds create review flags and never delete data.

## Execution
Run scripts 01–03 in order. Tests: `python -m pytest tests/test_dataset.py tests/test_dedupe.py -q`.

## Notes
Do not turn pHash candidates into exclusions without a logged Phase 3 rule. Do not derive labels from filenames or write into Main.
