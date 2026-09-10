# src/bone_cancer_ai

## Purpose
Reusable research pipeline.

## Files
__init__.py: version; utils/: environment/config/source safety; data/: audit, leakage control, preprocessing and loaders; annotations/: review, revisions, LabelMe conversion and validation. Models/losses/training/evaluation/explainability/inference remain future phases.

## Inputs
Config, future verified manifests and tensors.

## Outputs
Verified manifests, loaders and local annotation artifacts; model outputs Not run.

## Main Libraries
See utils/README.md for actual APIs.

## Important Functions / Classes
Package imports do not instantiate networks or train.

## Theory
Module boundaries separate data integrity, model computation and evaluation.

## Execution
python -c "import bone_cancer_ai; print(bone_cancer_ai.__version__)"

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
