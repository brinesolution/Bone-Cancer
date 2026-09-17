# configs

## Purpose
Configuration contracts.

## Files
`data.yaml` owns data policy; `baseline_*.yaml` defines Phase 5; `hosp_bonenet.yaml` defines Phase 6/7; `optimization.yaml` and its frozen output define Phase 8; `evaluation.yaml` defines Phase 9.

## Inputs
Approved directive, Phase 2 verified label meanings/duplicate evidence, and Phase 3 safety decisions.

## Outputs
Explicit YAML mappings consumed by the package.

## Main Libraries
PyYAML via utils.config.

## Important Functions / Classes
`load_config` rejects duplicate keys/non-mappings. `ProjectPaths` resolves the project and blocks outputs that overlap `dataset/Main`. Referenced `*_config` paths are relative to this folder; config inheritance is not implemented.

## Theory
Audit parameters control pHash/quality screening. The Phase 3 policy gives test then validation then train retention priority for connected groups, excludes exact-byte conflicts, retains pHash-only conflicts for review, and reserves 15% of clean official training for grouped calibration. Preprocessing uses grayscale-replicated RGB, aspect padding to 384 and ImageNet normalization. Training transforms prohibit flips/crops.

## Execution
Scripts 01–06 use `data.yaml`; 07/08 use `annotation.yaml`; 11 uses `optimization.yaml`; 12 uses `evaluation.yaml`.

## Notes
Policy action names are validated and unsupported silent-relabel behavior fails. After a reviewed change, rerun every affected upstream/downstream stage. Never point outputs into the source dataset.
