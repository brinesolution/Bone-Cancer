# artifacts

## Purpose
Model and evaluation outputs.

## Files
checkpoints/: learned weights; metrics/: real metrics; predictions/: outputs; figures/: charts; explainability/: overlays; model_cards/: limitations.

## Inputs
Future completed runs and frozen evaluation config.

## Outputs
Audit figures, checksum-addressed selected checkpoints, baseline metrics and Phase 8 HOSP ablations.

## Main Libraries
PyTorch, Pandas, Matplotlib and Pillow.

## Important Functions / Classes
The checkpoint registry verifies SHA-256 aliases; training/evaluation modules publish generated metrics.

## Theory
Artifact claims must be backed by executed code.

## Execution
`python scripts/02_audit_dataset.py` for Phase 2 figures.

## Notes
Do not edit generated artifacts manually or add fabricated results. Regenerate artifacts from recorded inputs/configs.
