# Phase 9 Explainability

Purpose:
Store qualitative HOSP-BoneNet overlays and outcome galleries.

Files:
- `true_*`, `false_*`, `uncertain_rejected/` — category galleries.
- `failure_gallery.png` — combined FP/FN cases.
- `gallery_index.csv` — displayed probabilities and uncertainty.
- `visual_qa.json` — technical visual-review findings.

Main libraries:
PyTorch and Matplotlib.

Flow:
Frozen HOSP samples → unverified spatial attention + Grad-CAM → reviewed galleries.

Run:
`.venv\Scripts\python.exe scripts\12_evaluate.py --config configs\evaluation.yaml`

Record technical QA in `visual_qa.json`, then rerun with `--finalize-visual-qa`.

Do not interpret these maps as verified segmentation or clinical localization.
