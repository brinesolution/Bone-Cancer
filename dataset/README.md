# dataset

Purpose:
Immutable local input collection. `Main/` is read-only and supplies audited images and labels.

Files:
- `Main/` — official train/validation/test source files.
- `BTXRD/`, `archive*/`, and folder exports — provenance evidence consumed by audit scripts.

Flow:
Source datasets → read-only audit → derived manifests/cache under `workspace/`.

Run:
`python scripts/01_index_dataset.py` and `python scripts/02_audit_dataset.py --config configs/data.yaml`

Do not edit, rename, move, overwrite or delete files under `Main/`.
