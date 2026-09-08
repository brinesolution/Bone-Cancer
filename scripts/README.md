# scripts

## Purpose
Phase command entry points.

## Files
00_verify_environment.py: active verifier; 01–03: Phase 2 audit; 04–06: Phase 3 preparation; 07–08: Phase 4 annotations; 09: Phase 5 baseline; 10: Phase 7 training; 11: Phase 8 tuning; 12–14: Phase 9 evaluation/explanations; 15: inference; run_phase.ps1: Phase 1 commands.

## Inputs
Configuration and installed package; later phases need verified artifacts.

## Outputs
Verifier JSON on stdout; diagnostics on stderr; --output writes a new safe report exclusively.

## Main Libraries
argparse/json and package utils.

## Important Functions / Classes
main validates paths before imports/output. run_phase.ps1 propagates command exit codes.

## Theory
Inactive commands fail explicitly rather than claiming results.

## Execution
python scripts/00_verify_environment.py --require-cuda; ./scripts/run_phase.ps1 -Action test

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
