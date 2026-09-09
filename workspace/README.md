# workspace

## Purpose
Derived working data.

## Files
manifests/: audit records; splits/: manifest assignments; cache/: derived images; annotations/: reviewed labels/spatial targets.

## Inputs
Read-only Main images/CSVs and configured local provenance metadata.

## Outputs
Local reproducible audit, cleaning, split and optional cache files.

## Main Libraries
Pandas, Pillow, ImageHash, OpenCV, Albumentations, scikit-learn, PyTorch, NumPy, PyYAML and openpyxl.

## Important Functions / Classes
`index_dataset`, `audit_labels`, and `find_duplicate_edges` produce Phase 2 evidence. `build_clean_manifest`, `assign_internal_splits`, and `cache_manifest_images` produce Phase 3 derived data through atomic writers.

## Theory
Separating source from derived data makes exclusions and leakage corrections reviewable and reversible without touching original images/folders.

## Execution
Run scripts 01–03 for the audit, then scripts 04 and 06 for Phase 3. Script 05 is an optional cache.

## Notes
Do not edit generated records manually. Regenerate them from source evidence/configuration. Phase 4 owns annotation revisions and optional-task activation.
