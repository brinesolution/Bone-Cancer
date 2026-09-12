# Losses

Purpose: Task-aware losses that exclude missing or invalid labels.

Files:
- `classification.py` — masked binary and multiclass losses.
- `segmentation.py` — FP32-stable Dice plus BCE.
- `concepts.py`, `grounding.py`, `hierarchy.py` — optional supervised terms.
- `total.py` — weighted loss composition and configuration checks.

Main library: PyTorch.

Flow: predictions + targets + validity masks → finite weighted total.

Run: `pytest tests/test_hosp_losses.py -q`

Sentinel targets are ignored only where their validity mask is false.
