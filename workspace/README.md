# workspace

## Purpose
Derived working data.

## Files
manifests/: audit records; splits/: manifest assignments; cache/: derived images; annotations/: reviewed labels/spatial targets.

## Inputs
Read-only Main images/CSVs and configured local provenance metadata.

## Outputs
Local reproducible derived files.

## Main Libraries
Pandas, Pillow, ImageHash, NumPy, PyYAML and openpyxl.

## Important Functions / Classes
`index_dataset`, `audit_labels`, `find_duplicate_edges`, and atomic schema writers produce the Phase 2 records. Phase 3 will consume them.

## Theory
Separating source from derived data enables reversible cleaning.

## Execution
Run scripts 01, 02 and 03 in order; see `scripts/README.md`.

## Notes
Do not edit generated records manually. Regenerate them from the source and configuration. Phase 3 owns cleaned split assignments; Phase 4 owns annotation revisions.
