# Spatial revision history

Each unique revision directory holds saved LabelMe JSON, native masks/boxes, overlay and a reviewer/timestamp record. `import_spatial_review` writes artifacts then publishes revision.json; `spatial_frame` chooses the latest timestamp per sample for export while retaining history. `validate_project_annotations` checks every revision against source image identity and geometry.

Inputs: hash-bound staged drawings and actual reviewer text. Outputs: project annotations, not clinical verification. Libraries: Pillow, OpenCV, NumPy, Pandas and standard library. Polygon coverage is a 0/1 union; boxes alone provide no mask. Use scripts07/08 via the parent workflow; never edit generated artifacts or erase older revisions manually.
