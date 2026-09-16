# workspace/cache/resized

Purpose: Local pre-augmentation cache for `fit_train` and `val`.

Files:
- `sample_*.png` — 384×384 lossless derived images.
- `cache_index.csv` — source/cache hashes, split, geometry, and format fingerprint.

Flow: verified source bytes → deterministic grayscale/resize/pad → cached pixels.

Libraries: Pillow, OpenCV, Pandas, hashlib.

Run: `.venv\Scripts\python.exe scripts/05_build_cache.py`

The DataModule rejects missing, stale, changed, or checksum-invalid entries. Generated files are Git-ignored; do not edit them manually.
