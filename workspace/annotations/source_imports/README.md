# Source conversion generations

`import_source_annotations` writes each BTXRD conversion into a new generation directory containing native masks, boxes, representative overlays and immutable CSV evidence. Latest workspace source indexes reference a generation; they remain separate from human review exports. Only exact image bytes and compatible source drawings permit reuse; conflicting drawings are quarantined.

Inputs: read-only BTXRD JSON/images and eligible Main manifest. Outputs: mechanically validated annotations and issues. Libraries: Pillow/OpenCV/NumPy for rasterization, Pandas for indexes, SHA-256 for identity. `validate_source_import` checks geometry and provenance against the source. Run script08 --import-source to rebuild deliberately; script08 alone validates. Do not edit generation artifacts, guess labels or treat source provenance as new expert verification. See the parent workflow README.
