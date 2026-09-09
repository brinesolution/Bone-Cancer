# src/bone_cancer_ai/data

## Purpose
Manifest validation, indexing, quality, deduplication, preprocessing and reusable image loading.

## Files
`schema.py` defines manifest/issues contracts, opaque IDs and atomic writers; `indexer.py` decodes and hashes; `quality.py` measures thumbnails and builds the clean manifest; `provenance.py` reads separate local metadata; `audit.py` verifies labels; `dedupe.py` groups duplicate evidence; `preprocessing.py` decodes, pads and optionally caches; `transforms.py` owns aligned image/mask transforms; `dataset.py` returns partial-supervision batches; `samplers.py` creates grouped calibration splits and optional class weights; `datamodule.py` owns the four DataLoaders and visual smoke grid.

## Inputs
Immutable Main images/CSVs, Phase 2 audit manifests, label schema, and data configuration.

## Outputs
Audited/clean manifests, grouped split tables, optional derived cache entries, structured tensors and batch QA summaries/figures.

## Main Libraries
pathlib, hashlib, Pillow, OpenCV, Albumentations, NumPy, Pandas, scikit-learn, PyTorch and Matplotlib.

## Important Functions / Classes
Phase 3 uses `build_clean_manifest`, `assign_internal_splits`, `split_generation_id`, `decode_radiograph`, `resize_and_pad`, `RadiographTransform`, `BoneCancerDataset`, `BoneCancerDataModule`, `make_weighted_sampler`, `cache_manifest_images`, and `smoke_test_loaders`. `valid_task_mask` prevents missing optional labels from entering losses; runtime decode failure returns an invalid zero sample and keeps the associated row identity/label isolated. The DataModule fingerprints all persisted fields and jointly revalidates the four manifests at the training boundary.

## Theory
Connected duplicate groups are the strongest available independence unit because patient/study IDs are absent. Official test has highest retention priority, followed by validation and training; lower-priority copies are excluded in derived manifests. Aspect-preserving resize uses one scale and central padding. Images become grayscale-replicated RGB tensors for pretrained encoders; masks share geometry with nearest-neighbour interpolation. Validation transforms are deterministic, while training uses mild affine/intensity noise without flips or crops.

## Execution
Run scripts 01–04 then 06, or `./scripts/run_phase.ps1 -Action phase3` after Phase 2 exists. Optional bounded cache: `python scripts/05_build_cache.py --limit 16`. Tests: `python -m pytest tests/test_transforms.py tests/test_splits.py -q`.

## Notes
Do not edit generated CSVs/cache entries or derive labels from filenames. pHash-only label conflicts remain flagged for Phase 4 review; exact-byte label-conflict groups are excluded. Optional task gates in `label_schema.yaml` must remain authoritative.
