# Annotation workflow

## Purpose, inputs and files

This workspace holds derived research annotations. Inputs are the audited class schema, eligible Phase 3 clean manifest, read-only Main images and BTXRD JSONs. `label_schema.yaml` records vocabulary and supervision gates. `reviews.sqlite3` stores metadata history; `image_labels.csv` exports latest project metadata/spatial reviews. `source_annotations.csv` separately indexes mechanically validated source annotations; `source_annotation_issues.csv` records rejected or unmatched source files. Source annotations never become human reviews automatically.

`labelme/<sample_id>/` contains an opaque editable image copy, working JSON and SHA-256 binding. `spatial_reviews/<revision_id>/` preserves each imported drawing, mask/boxes, overlay and reviewer record. `source_imports/<generation_id>/` preserves source-conversion generations. Generated/private files stay out of Git. Do not manually edit SQLite, bindings, exported indexes, masks or revision records. Source files under `dataset/` remain read-only.

## Metadata review

Run from `E:\Bone Cancer` using the existing environment:

```powershell
.\.venv\Scripts\python.exe scripts/07_launch_annotation_tool.py --check
.\.venv\Scripts\python.exe scripts/07_launch_annotation_tool.py
```

Open `http://127.0.0.1:8502`. The app displays the image, sample ID, original image ID and original binary label separately. Review fields begin unknown. Filename words never supply a diagnosis. Binary review uses 0=normal, 1=bone_tumour, plus unknown; subtype/risk use the audited schema vocabulary.

1. Enter your actual reviewer name. Each committed widget change autosaves a new revision with UTC timestamp, status and notes. Text changes commit on Enter or focus change.
2. Select only supported labels. For concepts, unchecked **Assessed** means unknown. Assessed with Present unchecked means absent; both checked means present. Unknown never becomes a negative training target.
3. Use draft, reviewed or needs_review. Reviewed means project review; it does not assert radiologist, pathology or clinical verification.
4. Use Previous/Next; saved values reload on return. Failed saves block navigation. Correct the field or use Reload saved revision to discard unsaved edits. Stale revisions reject concurrent overwrites.
5. Export master annotation CSV downloads current metadata and spatial paths. History retains earlier versions. Progress counts metadata saves/reviews and spatial coverage separately.

## Draw polygons and boxes

Expand **Lesion polygons and boxes — LabelMe**. Prepare editable copy, then Open LabelMe. The installed LabelMe 6.3.1 GUI opens on this computer with an explicit output directory; previous-image label copying is disabled.

Create a polygon around a lesion or a rectangle around its extent. Use the exact class `lesion`, with multiple shapes for multiple lesions. Use manual drawing tools; predictions are not ground truth. Save with Ctrl+S. Select existing shapes to edit them; reopening the staged image resumes the working drawing.

Return to the app, enter your reviewer name/notes, and Import saved spatial revision. The importer checks source/staged hashes, dimensions, classes, geometry, finite coordinates, bounds and EXIF orientation. It saves a new immutable revision and displays an overlay: red polygon coverage, yellow boxes. Inspect alignment, correct mistakes in LabelMe and import again. Older revisions remain intact. An empty drawing means missing spatial supervision, not a negative segmentation label.

After preparing a copy, a manual launch is also possible:

```powershell
.\.venv\Scripts\python.exe -m labelme <staged-image-path> --output <staged-folder> --labels lesion --validate-label exact --config "{keep_prev: false, auto_save: true, with_image_data: false}"
```

Version 6.3.1 requires an output directory, not a JSON filename. Source images/JSONs must never be the editing destination.

## Import and validate source annotations

```powershell
.\.venv\Scripts\python.exe scripts/08_validate_annotations.py --import-source
.\.venv\Scripts\python.exe scripts/08_validate_annotations.py
```

The first command creates a new generation from `dataset/BTXRD/Annotations`, matching source images to eligible Main rows by SHA-256 and rechecking actual bytes. A filename pairs JSON/image files but never supplies a diagnostic target. Subtype/risk come from the audited metadata; ambiguous labels stay blank. Invalid geometry is explicitly recorded. pHash/canonical similarity never authorizes spatial reuse.

The second command revalidates human revision history and latest source artifacts against original geometry, exports `image_labels.csv`, and writes `logs/phase4_annotation_validation.json`. Reimport only after relevant evidence or converter changes; retain previous generations for provenance. Mechanical source validation is distinct from project review and clinical verification.

## Theory and limitations

Coordinates are native pixels: x right, y down. Polygon vertices round to nearest integer pixels; filled regions form a union. PNG masks contain 0/1 at original height/width. Rectangles produce xyxy boxes and never fabricated segmentation masks. Edge arithmetic noise within 1e-6 pixel is clipped; larger out-of-bounds coordinates fail. Later image/mask transforms must remain aligned.

Only identical bytes permit automatic duplicate reuse. Similar, resized or augmented images need independent alignment review. Phase 3 assignments and label gates remain unchanged; this workflow does not release new supervision into training. Later masked losses must ignore unavailable labels and preserve calibration/test isolation. Optional branches stay disabled until a deliberate validated release. The ambiguous BTXRD subtype and pHash-only source-label conflicts remain review items. Tool completion does not imply full-dataset human annotation or clinical verification.

## Libraries and important interfaces

Streamlit/AppTest provide controls and workflow tests; SQLite provides transactional history; LabelMe/PyQt5 provides drawing; Pillow/OpenCV/NumPy handle geometry; Pandas and PyYAML handle tables/configs. Core APIs are `AnnotationStore`, `validate_metadata`, `convert_labelme`, `stage_labelme`, `import_spatial_review`, and `import_source_annotations` under `src/bone_cancer_ai/annotations`. See that folder's README for contracts. Integration uses the installed LabelMe CLI/source and official Streamlit [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) and [session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) APIs.
