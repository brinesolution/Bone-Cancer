# app

## Purpose
Streamlit research demo for frozen HOSP-BoneNet inference.

## Files
`streamlit_app.py` — upload/predict/display/export flow; `ui_helpers.py` — pure display and overlay helpers; `assets/` — static resources.

## Inputs
User-uploaded JPG/PNG plus the checksum-registered Phase 8 HOSP checkpoint.

## Outputs
Binary probability, confidence, entropy, conformal abstention, unverified spatial attention, Grad-CAM and JSON metadata.

## Main Libraries
Streamlit, PyTorch, Pillow, NumPy and Matplotlib.

## Important Functions / Classes
`load_predictor`, `_decode_upload`, `_show_result`; model logic stays in `src/bone_cancer_ai/inference`.

## Theory
Inference must not require/expose ground-truth labels.

## Execution
`streamlit run app\streamlit_app.py`

## Notes
No ground-truth labels are shown. Subtype/concept/OOD outputs remain unavailable. Keep `dataset/Main` untouched.
