# configs

## Purpose
Configuration contracts.

## Files
`data.yaml` contains immutable source paths, provenance roots, image extensions, audit thresholds and preprocessing defaults. `train.yaml`, `baseline_*.yaml`, `hosp_bonenet.yaml` and `inference.yaml` remain later-phase contracts.

## Inputs
Approved directive and Phase 2 verified label meanings.

## Outputs
Explicit YAML mappings consumed by the package.

## Main Libraries
PyYAML via utils.config.

## Important Functions / Classes
`load_config` rejects duplicate keys/non-mappings. `ProjectPaths` resolves the project and blocks outputs that overlap `dataset/Main`. Referenced `*_config` paths are relative to this folder; config inheritance is not implemented.

## Theory
Audit parameters control the deterministic pHash distance and conservative blank-image screen. Binary semantics are verified; model-training configs remain gated until cleaned splits exist.

## Execution
`python scripts/01_index_dataset.py --config configs/data.yaml` followed by scripts 02 and 03.

## Notes
Change thresholds only through reviewed configuration and rerun all three audit stages. Never point outputs into the source dataset.
