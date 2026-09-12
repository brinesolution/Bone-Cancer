# HOSP-BoneNet

Purpose: Configurable hierarchical, spatially grounded radiograph model.

Files:
- `encoder.py` — timm multistage features.
- `localization_head.py`, `pooling.py`, `fusion.py` — lesion map and global/local fusion.
- `concept_head.py`, `hierarchy.py` — gated optional predictions.
- `model.py`, `output.py` — validated assembly and typed output mapping.

Main libraries: PyTorch and timm.

Flow: image → ConvNeXt features → global/local fusion → configured heads.

Run: `python scripts/phase6_smoke_overfit.py`

Only binary supervision is currently verified; optional heads require schema gates.
