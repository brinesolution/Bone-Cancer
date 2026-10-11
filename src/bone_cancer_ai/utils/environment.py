"""Report installed dependencies and actual compute capability without loading datasets."""

import contextlib
import importlib
import importlib.metadata
import platform
import sys
from datetime import UTC, datetime

import torch

from bone_cancer_ai.utils.device import verify_tensor_device

PACKAGE_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "opencv-python-headless": "cv2",
    "pyyaml": "yaml",
    "omegaconf": "omegaconf",
    "hydra-core": "hydra",
    "tqdm": "tqdm",
    "rich": "rich",
    "psutil": "psutil",
    "joblib": "joblib",
    "imagehash": "imagehash",
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "monai": "monai",
    "albumentations": "albumentations",
    "torchmetrics": "torchmetrics",
    "segmentation-models-pytorch": "segmentation_models_pytorch",
    "pydicom": "pydicom",
    "SimpleITK": "SimpleITK",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "tensorboard": "tensorboard",
    "mlflow": "mlflow",
    "optuna": "optuna",
    "grad-cam": "pytorch_grad_cam",
    "shap": "shap",
    "pytest": "pytest",
    "pytest-cov": "pytest_cov",
    "ruff": "ruff",
    "black": "black",
    "mypy": "mypy",
    "jupyter": "jupyter",
    "ipykernel": "ipykernel",
    "streamlit": "streamlit",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}


def environment_report(
    require_cuda: bool = False, *, run_tensor_check: bool = True, device: str | None = None
) -> dict:
    """Import dependencies and optionally verify tensor compute on the actual device."""
    packages = {}
    for distribution, module in PACKAGE_IMPORTS.items():
        try:
            with contextlib.redirect_stdout(sys.stderr):
                importlib.import_module(module)
            packages[distribution] = {
                "version": importlib.metadata.version(distribution),
                "import_ok": True,
            }
        except Exception as exc:
            packages[distribution] = {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}
    available = torch.cuda.is_available()
    cuda = {"available": available, "build_version": torch.version.cuda, "devices": []}
    if available and device != "cpu":
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    if run_tensor_check:
        try:
            compute = verify_tensor_device(device or ("cuda" if available else "cpu"))
        except Exception as exc:
            compute = {"finite_gradient": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        compute = {"status": "Not run"}
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "tensor_check": compute,
        "ok": (
            sys.version_info[:2] == (3, 11)
            and all(value["import_ok"] for value in packages.values())
            and (not run_tensor_check or compute["finite_gradient"])
            and (available or not require_cuda)
        ),
        "model_training": "Not run",
    }
