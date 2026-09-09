# workspace/cache/resized

## Purpose
Derived resized radiographs.

## Files
Opaque `sample_*.png` images plus `cache_index.csv`. Phase 3 locally generated 16 smoke entries; generated files are Git-ignored.

## Inputs
Eligible clean-manifest paths, source hashes, and `configs/data.yaml` preprocessing fields.

## Outputs
Aspect-padded, grayscale-replicated RGB PNGs with 384×384 dimensions and index traceability.

## Main Libraries
Pillow, OpenCV, hashlib and Pandas.

## Important Functions / Classes
`cache_manifest_images` in `data/preprocessing.py`; `scripts/05_build_cache.py` is the producer.

## Theory
One scale factor preserves anatomy, central padding makes a square canvas, and source/cache hashes make derivation auditable.

## Execution
`python scripts/05_build_cache.py --limit 16`.

## Notes
Do not edit generated images/indexes manually or treat cached pixels as new labels. The source remains authoritative.
