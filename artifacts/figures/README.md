# artifacts/figures

## Purpose
Audit/evaluation figures.

## Files
`dataset_class_distribution.png`, `dimension_distribution.png` and `sample_grid.png` are Phase 2 audit figures. Later evaluation figures remain pending.

## Inputs
Audited `dataset_master.csv`.

## Outputs
Executed class-count, stored-dimension and deterministic source-preview figures.

## Main Libraries
Matplotlib, Pillow and NumPy.

## Important Functions / Classes
`save_audit_figures` in `data/audit.py` owns these files.

## Theory
Plots summarize executed audit rows. The sample grid withholds manifest IDs but keeps source pixels unchanged, so burned-in text may remain visible and must be treated as a leakage/privacy risk.

## Execution
`python scripts/02_audit_dataset.py` regenerates all three figures.

## Notes
Do not edit generated figures manually or present them as clinical review. Inspect stored orientation, exposure, text markers and capture artifacts before Phase 3 preprocessing decisions.
