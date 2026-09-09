# workspace/manifests

## Purpose
Dataset audit tables.

## Files
dataset_master.csv: image index; duplicate_groups.csv: grouping; quality_report.csv: quality findings; label_audit.csv: row/label issues.

## Inputs
Immutable Main images, three source CSVs and configured provenance metadata.

## Outputs
`dataset_master.csv` and `quality_report.csv` contain 12,557 rows. `label_audit.csv` retains one source metadata conflict. `duplicate_groups.csv` is a pair-edge report with stable connected-component IDs, evidence types, split flags and label conflicts.

## Main Libraries
Pandas, Pillow, ImageHash and NumPy.

## Important Functions / Classes
`index_dataset` owns master/quality rows; `audit_labels` enriches verified labels; `find_duplicate_edges` and `build_duplicate_groups` own duplicate evidence.

## Theory
SHA-256 confirms identical bytes. Canonical IDs connect exported variants. A 64-bit perceptual hash supplies review candidates under the configured Hamming distance. Connected groups must stay within one final split to prevent leakage.

## Execution
`python scripts/01_index_dataset.py`; `python scripts/02_audit_dataset.py`; `python scripts/03_find_duplicates.py`.

## Notes
Do not edit these generated CSVs manually. Perceptual matches are candidates, while exact hashes are confirmed byte duplicates. Phase 3 must create a clean manifest and resolve label conflicts without moving source files.
