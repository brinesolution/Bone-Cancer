# artifacts

## Purpose
Model and evaluation outputs.

## Files
checkpoints/: learned weights; metrics/: real metrics; predictions/: outputs; figures/: charts; explainability/: overlays; model_cards/: limitations.

## Inputs
Completed runs and frozen evaluation configuration.

## Outputs
Audit figures, checksum-addressed checkpoints, final metrics, statistical plots and explanations.

## Main Libraries
PyTorch, Pandas, Matplotlib and Pillow.

## Important Functions / Classes
The checkpoint registry verifies SHA-256 aliases; training/evaluation modules publish generated metrics.

## Theory
Artifact claims must be backed by executed code.

## Execution
Use the owning phase script; Phase 9 uses `scripts/12_evaluate.py`.

## Notes
Do not edit generated artifacts manually or add fabricated results. Regenerate artifacts from recorded inputs/configs.
