# configs

## Purpose
Configuration contracts.

## Files
`data.yaml` owns immutable paths and data policy. `train.yaml` and `baseline_*.yaml` define the shared Phase 5 runs. `hosp_bonenet.yaml` defines the verified Phase 6 architecture and binary-only supervision gate. `inference.yaml` points to the verified Phase 7 HOSP alias while Phase 9 evaluation and Phase 10 inference remain gated.

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
Run scripts 01–04 and 06 with `--config configs/data.yaml`; script 05 is optional. Scripts 07/08 use `annotation.yaml`, which references data.yaml plus annotation root, database, eligible manifest, loopback port, tri-state concept names and allowed spatial classes. Source class vocabulary stays in label_schema.yaml. Annotation review never silently changes training gates.

## Notes
Policy action names are validated and unsupported silent-relabel behavior fails. After a reviewed change, rerun every affected upstream/downstream stage. Never point outputs into the source dataset.
