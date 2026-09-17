# src/bone_cancer_ai

## Purpose
Reusable research pipeline.

## Files
__init__.py: version; utils/: safety; data/: audit/loaders; annotations/: review; models/losses/training/: HOSP and baselines; optimization/: Phase 8 contracts, execution, reuse and frozen loading.

## Inputs
Config, future verified manifests and tensors.

## Outputs
Verified manifests/loaders, model runs, frozen configs and checksum-bound selections.

## Main Libraries
See utils/README.md for actual APIs.

## Important Functions / Classes
Package imports do not instantiate networks or train; entry points call reusable modules.

## Theory
Module boundaries separate data integrity, model computation and evaluation.

## Execution
python -c "import bone_cancer_ai; print(bone_cancer_ai.__version__)"

## Notes
Do not edit generated records manually. Keep original dataset files untouched.
