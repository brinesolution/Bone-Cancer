# Requirements

## Purpose and files

`base.txt` lists data/config dependencies. `train.txt` lists the ML stack, with the verified CUDA torch/vision pair pinned. `dev.txt` lists testing, development and application dependencies. `lock_windows_gpu.txt` records installed versions after imports, CUDA and dependency checks succeed. It is a machine-specific freeze, not a cross-platform or wheel-hash lock.

## Inputs and outputs

Inputs: Python 3.11 and a compatible NVIDIA driver. Output: project-local `.venv`. No dataset is used during installation. VS Code Python and Pylance extensions were already installed.

## Setup

The Windows launcher lists uv-managed Python 3.11 but `py -3.11` cannot resolve it on this machine. For initial environment creation, use uv:

```powershell
uv venv --python 3.11 --seed .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements/base.txt -r requirements/train.txt -r requirements/dev.txt
python -m pip install -e .
python -m pip check
python scripts/00_verify_environment.py --require-cuda
pytest -q
```

Do not recreate the verified environment simply to resume another phase. For version reproduction, install the exact torch/vision pair above, then `python -m pip install -r requirements/lock_windows_gpu.txt` and `python -m pip install -e . --no-deps`. Run verification again. The editable project is installed separately so the freeze contains no developer-specific source path.

## Libraries and theory

PyTorch wheels include their CUDA runtime; the driver capability from `nvidia-smi` differs from the wheel runtime version. Current wheel availability was checked against PyPI and the official CUDA 13.0 index. Driver 591.86 advertises CUDA 13.1 capability; actual tensor execution verifies compatibility. The maintained distribution for `pytorch_grad_cam` is `grad-cam`.

References: [PyTorch selector](https://pytorch.org/get-started/locally/), [version commands](https://pytorch.org/get-started/previous-versions/), [CUDA wheel index](https://download.pytorch.org/whl/cu130/torch/), [Grad-CAM installation](https://github.com/jacobgil/pytorch-grad-cam#advanced-ai-explainability-for-pytorch).

## Functions and notes

No Python functions live here. Requirements express dependency intent; the freeze records executed installation. Refresh it only after verification and document why. Never silently replace the CUDA build with CPU wheels.

OpenCV uses `opencv-python-headless` instead of `opencv-python`: Albumentations and Grad-CAM both require this distribution, and this project does not use cv2 GUI windows. The two distributions share the same `cv2` namespace and must not coexist. The API remains OpenCV; future annotation/UI work uses dedicated GUI tooling. See the [OpenCV package installation guidance](https://pypi.org/project/opencv-python-headless/).
