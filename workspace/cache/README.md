# workspace/cache

Purpose: Verified derived pixels that avoid repeated decode/resize work.

Files:
- `resized/` — versioned 384×384 PNGs and `cache_index.csv`.
- `normalized/` — reserved; currently unused.

Inputs: `fit_train.csv`, `val.csv`, source SHA-256, and preprocessing config.

Flow: immutable source → hash/decode/resize/pad → verified PNG → dynamic augmentation.

Libraries: Pillow, OpenCV, Pandas, hashlib.

Run: `.venv\Scripts\python.exe scripts/05_build_cache.py`

Do not edit cache files/indexes or place derived files in `dataset/Main`.
