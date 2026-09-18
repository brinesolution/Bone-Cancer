# app

## Purpose
Streamlit research demo for the current BTXRD subtype HOSP-BoneNet inference.

## Files
`streamlit_app.py` — compact image analysis and model-analytics tabs; `ui_helpers.py` — pure display and overlay helpers; `assets/` — static resources.

## Inputs
User-uploaded JPG/PNG, the checksum-registered BTXRD Phase B checkpoint, and sealed metric/review artifacts.

## Outputs
Binary/subtype probabilities, conformal sets, uncertainty, top-k ranking, compact image evidence, provenance and JSON metadata.

## Main Libraries
Streamlit, PyTorch, Pillow, NumPy and Matplotlib.

## Important Functions / Classes
`load_btxrd_predictor`, `_decode_upload`, `_show_btxrd_result`; model logic stays in `src/bone_cancer_ai/inference`.

## Theory
Inference must not require/expose ground-truth labels.

## Execution
`python launch_web_app.py` (or `python launch_web_app.py --port 8502`)

## Notes
No ground-truth labels are shown. Subtype predictions remain research-only. Keep `dataset/Main` untouched.
