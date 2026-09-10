# Editable LabelMe copies

Purpose: local manual polygons/boxes on copied radiographs. Inputs: an eligible source record and verified SHA-256. `stage_labelme` creates a per-sample image, JSON and binding sidecar; LabelMe edits only that working copy. Outputs become immutable revisions through `import_spatial_review`.

Run the annotation app, Prepare editable copy, Open LabelMe, draw/save, then Import saved spatial revision. Libraries: LabelMe/PyQt5, Pillow, standard-library hashing/copying. Exact byte identity protects coordinate reuse; filenames are opaque here. Never alter bindings or replace staged images; source changes require a new audit. See the parent workflow README for commands and theory.
