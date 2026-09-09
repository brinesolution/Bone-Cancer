# workspace/annotations

## Purpose
Verified annotation records.

## Files
`label_schema.yaml` records audited binary meanings and partial-supervision availability. `image_labels.csv`, `subtype_labels.csv`, `concept_labels.csv`, `boxes/` and `masks/` remain Phase 4 review/export targets.

## Inputs
Phase 2 metadata audit; future project-team review.

## Outputs
Header-only tables, empty spatial folders; no claimed annotations.

## Main Libraries
PyYAML reads the schema. Pandas/openpyxl read the Phase 2 provenance tables.

## Important Functions / Classes
Binary semantics come from `audit_labels`. Review/storage/overlay validation remains Phase 4.

## Theory
Unknown differs from negative. Partial subtype/risk/spatial coverage requires validity masks, and availability does not establish annotation quality.

## Execution
Read `label_schema.yaml`; rerun script 02 when provenance inputs change. Annotation commands are Pending Phase 4.

## Notes
Do not enable subtype, risk, box or mask supervision before Phase 4 validates annotations and overlays. Never invent missing labels.
