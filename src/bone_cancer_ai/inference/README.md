# src/bone_cancer_ai/inference

## Purpose
Frozen single-image inference and calibrated research output.

## Files
`predictor.py` — registry-bound preprocessing/model/explanation flow; `postprocess.py` — JSON-safe result contract; `conformal.py` — binary split-conformal sets.

## Inputs
uint8 RGB/grayscale radiograph; `configs/inference.yaml`; registry alias `hosp_bonenet.phase8.selected`.

## Outputs
`InferenceResult` with binary probability, confidence, entropy, conformal set, abstention and optional maps.

## Main Libraries
PyTorch, OpenCV, Pillow/NumPy, Albumentations-compatible preprocessing and Grad-CAM helpers.

## Important Functions / Classes
`FrozenPredictor.from_project_root`, `.predict_array`, `.predict_path`, `validate_image_array`, `result_to_json`.

## Theory
Inference works without labels and must abstain when appropriate.

## Execution
`.venv\Scripts\python.exe scripts\15_infer_image.py --image PATH --device cuda`

## Notes
Inference never reads ground-truth labels. Subtype, concept and OOD outputs stay unavailable until verified supervision is released.
