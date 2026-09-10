# Annotation modules

## Purpose and interfaces

- `context.py`: load_annotation_context resolves config/schema/eligible manifest; image_path checks containment and SHA-256.
- `schema.py`: validate_metadata enforces vocabulary, reviewer/status and unknown-aware values.
- `storage.py`: AnnotationStore provides transactional save/latest/history/export/progress; expected_revision rejects stale saves.
- `spatial.py`: convert_labelme validates native geometry; save_spatial writes 0/1 masks and xyxy boxes; render_overlay produces QA images. Polygon regions union; boxes never fabricate masks.
- `labelme_bridge.py`: stage_labelme creates opaque hash-bound copies; import_staged verifies identity and returns geometry plus the exact JSON bytes converted; verify_equivalent compares exact file content.
- `workflow.py`: prepare/launch LabelMe, save immutable spatial revisions and export latest project metadata/spatial records.
- `source_import.py`: import and validate BTXRD annotations by exact Main-image matches in separate generation directories.
- `validation.py`: revalidate all project revision history against source identity, schema and geometry.

Inputs: explicit configs, Paths, manifest rows and LabelMe JSON. Outputs: local SQLite history, CSV indexes, staged copies, revision artifacts and validation reports. Errors fail explicitly for unsafe paths, changed images, stale revisions, unknown classes or invalid geometry. Source labels/partitions are not rewritten.

## Libraries, theory and execution

The standard library provides SQLite, hashing, UUIDs and subprocesses. Pandas/PyYAML handle tables/configs. Pillow, NumPy and OpenCV handle native-resolution geometry. Unknown denotes missing supervision, not class zero. Only exact identity establishes automatic geometry reuse; partial labels require later masked losses.

Scripts 07/08 orchestrate the modules. Run `.\.venv\Scripts\python.exe -m pytest tests/test_annotations.py tests/test_annotation_app.py tests/test_annotation_spatial.py tests/test_annotation_workflow.py tests/test_annotation_source.py tests/test_labelme_runtime.py -q` from the project root. See [workflow documentation](../../../workspace/annotations/README.md) for commands and limits. Do not edit generated revisions or source files manually.
