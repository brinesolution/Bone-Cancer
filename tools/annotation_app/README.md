# tools/annotation_app

## Purpose
Local Streamlit annotation review interface. See [the complete workflow](../../workspace/annotations/README.md).

## Files
app.py: review controls and autosave callbacks; storage.py and validators.py: public exports of reusable package APIs.

## Inputs
Verified schema and local radiographs.

## Outputs
Metadata history, CSV download, staged LabelMe copies, immutable spatial revisions and progress counts.

## Main Libraries
Streamlit, Pandas and the project annotation package; LabelMe/PyQt5 runs as a separate local process.

## Important Functions / Classes
AnnotationStore saves optimistic revisions. Session callbacks reload saved values and preserve unknown labels. Core logic lives in src/bone_cancer_ai/annotations.

## Theory
Spatial alignment and provenance must be validated before annotation-derived supervision.

## Execution
Run `.\.venv\Scripts\python.exe scripts/07_launch_annotation_tool.py` from the project root. The server binds to 127.0.0.1:8502. Use --config for an alternate workspace. AppTest covers editing, autosave and navigation on synthetic fixtures.

## Notes
Do not edit generated records manually. Keep original dataset files untouched. This annotation interface is separate from the later inference/demo app.
