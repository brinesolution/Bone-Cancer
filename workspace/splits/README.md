# workspace/splits

## Purpose
Manifest-based partitions.

## Files
fit_train.csv, calibration.csv, val.csv, test.csv: header-only split scaffolds.

## Inputs
Future audited source/duplicate groups.

## Outputs
No assigned records yet.

## Main Libraries
None in Phase 1.

## Important Functions / Classes
Future split builder assigns groups without moving images.

## Theory
Calibration is held out from official training after duplicate grouping; test never tunes models.

## Execution
Pending Phase 3.

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
