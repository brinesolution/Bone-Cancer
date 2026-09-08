# src

## Purpose
Installable source layout.

## Files
bone_cancer_ai/: reusable package and all future pipeline modules.

## Inputs
Project config and permitted inputs.

## Outputs
Importable package after editable install.

## Main Libraries
setuptools packages the source.

## Important Functions / Classes
Public APIs belong in the package, not script-only implementations.

## Theory
A src layout avoids accidental imports from an uninstalled working directory.

## Execution
python -m pip install -e . --no-deps

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
