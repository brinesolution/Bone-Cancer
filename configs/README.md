# configs

## Purpose
Configuration contracts.

## Files
data.yaml: authoritative paths and data defaults; train.yaml: common future training budget; baseline_*.yaml: four model identities; hosp_bonenet.yaml: approved architecture defaults; inference.yaml: inactive checkpoint/threshold settings.

## Inputs
Approved directive; future verified labels.

## Outputs
Explicit YAML mappings consumed by the package.

## Main Libraries
PyYAML via utils.config.

## Important Functions / Classes
load_config validates mappings. Referenced *_config paths are relative to this folder; inheritance is not implemented in Phase 1.

## Theory
Unverified labels use null/empty mappings. All classifiers await verified class counts.

## Execution
python scripts/00_verify_environment.py --config configs/data.yaml

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
