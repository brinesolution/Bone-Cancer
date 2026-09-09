# .vscode

## Purpose
Editor setup.

## Files
settings.json: project interpreter, pytest and excluded data watchers; launch.json: environment debugger; tasks.json: verify/test/lint/compile plus ordered Phase 2 audit and Phase 3 data pipeline.

## Inputs
Workspace root and .venv.

## Outputs
Editor commands using the project interpreter.

## Main Libraries
None; editor JSON configuration.

## Important Functions / Classes
No Python functions. Paths use workspaceFolder expansion.

## Theory
Consistent interpreter selection prevents accidental use of system Python.

## Execution
Use VS Code Run Task → Run Phase 3 data pipeline after Phase 2 evidence exists.

## Notes
Do not edit generated records manually. Keep original dataset files untouched.
