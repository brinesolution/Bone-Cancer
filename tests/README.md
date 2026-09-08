# tests

## Purpose
Phase-scoped behavioral tests.

## Files
test_utils.py: YAML, paths, RNG, logging, hashes; test_environment.py: installed CLI, CUDA and inactive training entry points; remaining test_*.py: future-phase docstring shells.

## Inputs
Synthetic temporary files/tensors and installed environment.

## Outputs
pytest results; temporary data under logs/pytest_tmp.

## Main Libraries
pytest, NumPy, torch, subprocess.

## Important Functions / Classes
Tests never train models. CUDA checks use tiny synthetic tensors.

## Theory
Tests must fail for broken behavior, not simply check that files exist.

## Execution
pytest -q

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
