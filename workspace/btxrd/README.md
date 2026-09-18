# BTXRD Migration Artifacts

Purpose: derived audit/linkage outputs for the read-only `dataset/BTXRD` release.

Files:
- `btxrd_manifest.csv` — hashed image rows, masked subtype targets, Main linkage, groups and splits.
- `fit_train.csv`, `calibration.csv`, `val.csv`, `test.csv` — grouped derived partitions.
- `subtype_conflicts.csv` — rows excluded from subtype supervision.
- `duplicates.csv` — exact/perceptual duplicate evidence.
- `audit_summary.json` — counts, checksums and validation status.

Flow: `dataset/BTXRD` + Main manifest → `scripts/16_prepare_btxrd.py` → these files.

Run: `.venv\Scripts\python.exe scripts\16_prepare_btxrd.py --config configs/btxrd_data.yaml`

Do not edit these outputs or either source dataset manually; rerun the script to refresh them.
