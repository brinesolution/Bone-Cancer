# artifacts

## Purpose
Model and evaluation outputs.

## Files
checkpoints/: learned weights; metrics/: real metrics; predictions/: outputs; figures/: charts; explainability/: overlays; model_cards/: limitations.

## Inputs
Future completed runs and frozen evaluation config.

## Outputs
Phase 2 dataset audit figures exist under `figures/`. No model artifacts or performance metrics exist.

## Main Libraries
Matplotlib/Pillow for current audit figures; later producers remain pending.

## Important Functions / Classes
`save_audit_figures` is active. Model/evaluation producers are implemented in later phases.

## Theory
Artifact claims must be backed by executed code.

## Execution
`python scripts/02_audit_dataset.py` for Phase 2 figures.

## Notes
Do not edit generated artifacts manually or add fabricated results. Regenerate artifacts from recorded inputs/configs.
