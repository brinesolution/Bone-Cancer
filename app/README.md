# app

## Purpose
Streamlit research demo for frozen HOSP-BoneNet inference.

## Files
`streamlit_app.py` — image analysis and sealed-results tabs; `ui_helpers.py` — pure display and overlay helpers; `assets/` — static resources.

## Inputs
User-uploaded JPG/PNG, the checksum-registered Phase 8 HOSP checkpoint, and sealed Phase 5/9 metric artifacts.

## Outputs
Binary probability, confidence, entropy, conformal abstention, unverified spatial attention, Grad-CAM and JSON metadata.

## Main Libraries
Streamlit, PyTorch, Pillow, NumPy and Matplotlib.

## Important Functions / Classes
`load_predictor`, `_decode_upload`, `_show_result`; model logic stays in `src/bone_cancer_ai/inference`.

## Theory
Inference must not require/expose ground-truth labels.

## Execution
`python launch_web_app.py` (or `python launch_web_app.py --port 8502`)

## Notes
No ground-truth labels are shown. Subtype/concept/OOD outputs remain unavailable. Keep `dataset/Main` untouched.
