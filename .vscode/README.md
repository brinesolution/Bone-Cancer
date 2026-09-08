# .vscode

## Purpose
Editor setup.

## Files
settings.json: project interpreter, pytest and excluded data watchers; launch.json: environment debugger; tasks.json: verify/test/lint/compile.

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
Use VS Code Run Task → Verify environment.

## Notes
Do not edit generated records manually. Keep original dataset files untouched. Future-phase scaffolds are not completed functionality; update this README when implementation becomes active.
