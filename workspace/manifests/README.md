# workspace/manifests

## Purpose
Dataset audit tables.

## Files
`dataset_master.csv`: image index; `duplicate_groups.csv`: pair/group evidence; `quality_report.csv`: quality findings; `label_audit.csv`: row/label issues; `dataset_clean.csv`: all source records plus Phase 3 eligibility/assignment fields; `exclusions.csv`: excluded rows and reasons.

## Inputs
Immutable Main images/CSVs, configured provenance metadata and Phase 3 split policy.

## Outputs
`dataset_master.csv`, `quality_report.csv`, and `dataset_clean.csv` contain 12,557 rows. The clean manifest marks 9,063 eligible and 3,494 excluded records; `exclusions.csv` holds those 3,494 decisions. `duplicate_groups.csv` is a pair-edge report with stable connected-component IDs, evidence types, split flags and label conflicts.

## Main Libraries
Pandas, Pillow, ImageHash and NumPy; Phase 3 policy code uses stable SHA-derived opaque IDs.

## Important Functions / Classes
`index_dataset` owns master/quality rows; `audit_labels` enriches verified labels; `find_duplicate_edges` and `build_duplicate_groups` own duplicate evidence; `build_clean_manifest` adds eligibility, review, group and clean-pool fields without changing the source.

## Theory
SHA-256 confirms identical bytes. Canonical IDs connect exported variants. A 64-bit perceptual hash supplies review candidates. Phase 3 retains records from the highest-priority official split in each connected group and excludes lower-priority copies. Exact-byte label-conflict groups are excluded; pHash-only conflicts stay grouped and flagged.

## Execution
Run scripts 01–03, then `python scripts/04_build_clean_manifest.py`.

## Notes
Do not edit these generated CSVs manually. Regenerate them from the scripts. Source paths are audit metadata and must never be model inputs.
