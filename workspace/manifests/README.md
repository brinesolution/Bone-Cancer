# workspace/manifests

## Purpose
Dataset audit tables.

## Files
dataset_master.csv: image index; duplicate_groups.csv: grouping; quality_report.csv: quality findings; label_audit.csv: row/label issues.

## Inputs
Pending full local audit.

## Outputs
Header-only CSV scaffolds; no audited records.

## Main Libraries
None in Phase 1.

## Important Functions / Classes
Future indexer/audit/dedupe modules own these outputs.

## Theory
Hashes, grouping and label-row integrity precede training.

## Execution
Pending Phase 2.

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
