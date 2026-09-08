# workspace/annotations

## Purpose
Verified annotation records.

## Files
label_schema.yaml: unverified schema, every optional task disabled; image_labels.csv: binary reviews; subtype_labels.csv: subtype/risk reviews; concept_labels.csv: long-form concept reviews; boxes/ and masks/: spatial targets.

## Inputs
Future source documentation and human review.

## Outputs
Header-only tables, empty spatial folders; no claimed annotations.

## Main Libraries
PyYAML reads schema in future phases.

## Important Functions / Classes
Review/storage/validation functions are Pending Phase 4.

## Theory
Unknown differs from negative; absence must not create a false label.

## Execution
Pending Phase 4.

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
