# workspace/cache

## Purpose
Derived image cache.

## Files
resized/: aspect-preserving resized data; normalized/: optional validated derived arrays.

## Inputs
Eligible rows from `dataset_clean.csv`, verified source SHA-256 values, and preprocessing config.

## Outputs
Optional derived 384×384 PNGs and `cache_index.csv` under `resized/`. Phase 3 verified a bounded 16-image cache; the DataLoader still reads originals unless a future explicit cache integration is validated.

## Main Libraries
Pillow, OpenCV, Pandas and hashlib.

## Important Functions / Classes
`cache_manifest_images` validates each source hash, applies the same decode/resize/pad contract, uses opaque filenames, writes atomically, and records source/cache hashes plus a preprocessing fingerprint.

## Theory
Cache validity depends on both source bytes and transformation configuration. Cache files are derived performance aids and cannot replace the audit manifest.

## Execution
Optional smoke run: `python scripts/05_build_cache.py --limit 16`. Omit `--limit` only when a complete cache is intentionally needed.

## Notes
Do not edit cache files/indexes or copy them into `dataset/Main`. Delete/rebuild stale derived cache only through a reviewed workflow; source files remain immutable.
