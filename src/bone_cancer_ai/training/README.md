# Training

Purpose: Shared AMP training, validation selection, checkpointing and deterministic resume.

Files:
- `trainer.py` — masked binary fit/predict loop.
- `amp.py`, `optimizer.py`, `scheduler.py` — numerical and optimization policy.
- `callbacks.py`, `checkpointing.py` — early stopping and safe best/last state.

Main libraries: PyTorch.

Flow: fit loader → weighted BCE/AdamW → validation AUROC → `best.pt`/`last.pt`.

Defaults: CUDA AMP, 40 epochs, patience 8, seed 42, effective batch 8.

Test data is evaluated only after the best validation checkpoint is selected.
