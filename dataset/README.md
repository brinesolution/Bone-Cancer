# dataset

## Purpose
Immutable local input collection.

## Files
`Main/` contains the immutable official train/val/test images and label CSVs. `bone cancer detection.v1i.folder/` and `archive(2)/` provide independent folder-class evidence for binary/risk meanings. `BTXRD/` contains the workbook and LabelMe annotations; `archive(3)/` contains the nine-class detection export. ZIPs and `archive/` remain source material.

## Inputs
Original local datasets.

## Outputs
Read-only inputs to the Phase 2 audit; no generated outputs belong here.

## Main Libraries
None in this directory.

## Important Functions / Classes
No implementation or generated files belong here.

## Theory
Preserving original bytes makes later cleaning and split decisions traceable.

## Execution
`python scripts/01_index_dataset.py` indexes only `Main`. `python scripts/02_audit_dataset.py` reads configured provenance sources without merging or changing them.

## Notes
Keep every source dataset file untouched. Labels are joined through canonical IDs for audit only; diagnostic words in filenames are never model features or inferred labels.
