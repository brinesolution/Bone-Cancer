# tools

## Purpose
Local annotation and verified-artifact maintenance tools.

## Files
`annotation_app/`: Phase 4 app/storage/validator modules.
`refresh_checkpoint_registry.py`: validates existing Phase 5/7 selections and registers
their missing or idempotent stable checkpoint aliases without training; it does not replace
an alias that identifies a different selection.

## Inputs
Reviewed annotation state and completed, validated training bundles.

## Outputs
Checkpoint registry metadata and local content-addressed checkpoint links.

## Main Libraries
Python standard library plus the Phase 5/7 validators and artifact registry.

## Important Functions / Classes
`refresh_checkpoint_registry` composes the baseline and HOSP validation/registration APIs.

## Theory
Annotation tools must preserve unknown labels and reviewer history.

## Execution
Run `python tools/refresh_checkpoint_registry.py` to validate and register all selected
Phase 5/7 checkpoints without loading training data or starting training.

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
