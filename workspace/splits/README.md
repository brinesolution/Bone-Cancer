# workspace/splits

## Purpose
Manifest-based partitions.

## Files
`fit_train.csv`, `calibration.csv`, `val.csv`, and `test.csv`: self-contained eligible record manifests.

## Inputs
Phase 3 `dataset_clean.csv`, duplicate-group IDs, verified binary labels, and the configured seed/calibration fraction.

## Outputs
Executed counts are fit 4,871, calibration 857, validation 1,453, and test 1,882. Every one of the 9,063 eligible records occurs exactly once.

## Main Libraries
Pandas and scikit-learn `GroupShuffleSplit` through `data/samplers.py`.

## Important Functions / Classes
`assign_internal_splits` searches seeded group-respecting training candidates; `validate_final_splits` rejects overlap, missing assignments, and held-out-to-training movement. `split_generation_id` fingerprints every persisted field except itself. `BoneCancerDataModule` jointly validates all four files and refuses mixed/stale generations or paths outside `dataset/Main`.

## Theory
Calibration is held out from clean official training after duplicate grouping. Official validation/test semantics are preserved, except lower-priority duplicate copies are excluded to stop leakage. Test labels never select calibration assignments or model decisions.

## Execution
`python scripts/06_make_internal_splits.py` or `./scripts/run_phase.ps1 -Action phase3`.

## Notes
Do not edit generated records manually or train from the official folders. Regenerate after any upstream audit/policy change. No patient/study identifier exists, so duplicate connected groups are the strongest available grouping unit.
