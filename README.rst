HOSP-BoneNet: bone-tumour radiograph research
===========================================

Python research pipeline for binary bone-tumour classification and a nine-class
BTXRD subtype experiment. It includes data auditing, duplicate-aware splitting,
annotation review, baseline and HOSP-BoneNet training, calibration, evaluation,
explanations, and a local Streamlit interface. Predictions are research outputs
and are not a diagnosis.

The Streamlit app uses the selected BTXRD binary/subtype model. The older
``scripts/15_infer_image.py`` command uses the frozen binary-only model. Their
checkpoints, label schemas, and calibration artifacts are separate.

Automatic Windows setup and launch
----------------------------------

Clone into a Windows x64 folder, including paths containing spaces. Use
Windows PowerShell 5.1 or PowerShell 7. Python need not be installed beforehand.
From the clone's root::

    powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1

``setup.ps1`` finds the clone's actual path, downloads a checksum-verified uv
bootstrapper when needed, obtains Python 3.11, creates ``.venv``, installs missing
dependencies and the editable project, and runs dependency/import/tensor/model
checks. It uses the recorded Windows version constraints; CPU installations
substitute the matching PyTorch/torchvision CPU wheels for the CUDA wheel pins.
Auto mode preserves a usable existing CPU/CUDA environment, detects compatible
NVIDIA support for a new environment, and falls back to CPU if CUDA cannot
execute its verification tensors. System GPU drivers are not modified.

The installer downloads missing BTXRD data from the authors' public Figshare
archive, verifies its published MD5 checksum before extraction, and validates
workbook labels and all 3,746 expected image paths. It also restores missing
trained-model/result assets from the project's GitHub Release or an explicitly
provided URL. Model bundles and every declared asset are SHA-256 checked.
Existing source files with different bytes are preserved and reported.

``run.ps1`` checks trained-model readiness, runs portable tests, starts the
Streamlit service hidden in the background, waits for its HTTP health endpoint,
and opens the browser. Streamlit serves both the interface and inference backend;
there is no separate FastAPI service to start. If the preferred port is busy,
the launcher finds a free local port. A matching existing instance is reused.

Useful options::

    .\setup.ps1 -Device cpu
    .\setup.ps1 -CheckOnly
    .\setup.ps1 -RunTests
    .\run.ps1 -AutoSetup
    .\run.ps1 -Port 8502 -NoBrowser
    .\run.ps1 -Tests full
    .\run.ps1 -CheckOnly -Tests quick

``-Tests quick`` is the default portable unit/model-contract subset. ``-Tests
full`` runs the full research suite and requires the original Main dataset and
legacy frozen artifacts as well. ``logs/setup_report.json`` and
``logs/run_readiness.json`` record actual readiness; a missing or invalid model
causes a nonzero exit instead of a successful-looking launch.

The launchers support Windows. The reusable Python package remains separate from
these Windows entry points. ``requirements/lock_windows_gpu.txt`` is the recorded
Windows freeze, rather than a portable platform lock.

Local data and model setup
--------------------------

Git contains source code, configurations, tests, notebooks, a label schema, the
lightweight checkpoint registry, and a small frozen ConvNeXt memory record
required by its configuration and tests. Datasets, derived manifests, trained
weights, evaluation results, run directories, and local documentation are
excluded. A fresh clone needs those assets before training or inference.

BTXRD is downloaded automatically from the `authors' dataset record
<https://doi.org/10.6084/m9.figshare.27865398>`_. It is licensed CC BY 4.0; retain
attribution to the authors of *A Radiograph Dataset for the Classification,
Localization, and Segmentation of Primary Bone Tumors*. The archive contains
``BTXRD/images``, ``BTXRD/Annotations``, and ``BTXRD/dataset.xlsx``.

The larger original ``dataset/Main`` collection has 12,557 images and is not the
same download as BTXRD. It is optional for the selected web app, but needed for
the older Main training/evaluation protocol. Restore that original collection
separately or supply a checksum-bound ``Main/`` ZIP using ``-MainDatasetUrl`` and
``-MainDatasetSha256``. The installer does not invent the old official split
assignments from a different dataset.

Setup updates the two data configurations' absolute root fields when the clone
moves. If Main is unavailable, the subtype data configuration points at the
original BTXRD images for any fresh BTXRD preparation. Existing historical
manifests are not rewritten: their absolute paths, cohort assignments and run
fingerprints belong to the original experiment. Regenerate derived manifests
before a new training run in a relocated folder.

The selected BTXRD app requires its checksum-addressed checkpoint under
``artifacts/checkpoints/``, the matching registry entry, and its completed run
bundle under ``experiments/btxrd_subtype/``. The prepared
``dist/bone-cancer-runtime.zip`` contains the selected checkpoint, matching
calibration/run evidence, results and figures, excluding the dataset, environment
and private annotation workspace. **Upload this bundle once as a GitHub Release
asset named ``bone-cancer-runtime.zip`` before expecting automatic model download
in a fresh clone.** The default URL is inferred from the clone's GitHub
``origin`` remote. No GitHub remote or release existed when these launchers were
prepared.

For another host or a newly exported bundle::

    .\setup.ps1 -ModelAssetsUrl "https://HOST/bone-cancer-runtime.zip" -ModelAssetsSha256 "EXPECTED_SHA256"
    .\setup.ps1 -ExportAssets

The prepared bundle's checksum is pinned in ``setup.ps1`` and recorded in
``dist/bone-cancer-runtime.zip.json``. Re-exporting changes the archive checksum;
provide its new checksum or update the pin before publishing the new release.
The selected checkpoint must still match the checked-in registry. Temperature calibration and
robustness reporting are provided by scripts 21 and 22; script 19 checks the
selected run's integrity. Review each command's ``--help`` before a new run.

The original minimal launcher remains available as
``.venv/Scripts/python.exe launch_web_app.py``. The two-file workflow above adds
automatic setup, verification, health checks and browser opening.

The app provides image analysis and model-result tabs. The legacy binary CLI
requires its separate frozen Phase 9 artifacts::

    .\.venv\Scripts\python.exe scripts/15_infer_image.py --image "PATH_TO_XRAY" --device cuda

Development checks
------------------

From an installed environment::

    .\.venv\Scripts\python.exe -m ruff check src scripts app tools tests
    .\.venv\Scripts\python.exe -m compileall -q src scripts app tools
    .\.venv\Scripts\python.exe -m pytest -q

Some integration tests require local datasets and sealed evaluation artifacts.
Their absence in a fresh clone is not equivalent to a model verification pass.
CUDA checks use the ``cuda`` pytest marker.

Repository layout
-----------------

* ``src/bone_cancer_ai/``: reusable data, annotation, model, training, evaluation,
  inference, explanation, artifact, and safety modules.
* ``scripts/``: numbered pipeline entry points and phase helpers.
* ``configs/``: data policies, schemas, training, optimization, and inference
  configuration.
* ``app/`` and ``tools/``: Streamlit analysis and annotation tools.
* ``tests/``: unit, CLI, contract, integration, and synthetic model checks.
* ``requirements/``: dependency lists and the recorded Windows GPU freeze.
* ``notebooks/``: small exploratory placeholders; core logic lives in the package.
* ``dataset/``, ``workspace/``, ``experiments/``, ``artifacts/``, ``logs/``: local
  source and generated-data boundaries, retained with directory placeholders.

All project Markdown is kept locally under ``documentry/``, mirroring its former
relative paths. ``documentry/finalarch.md`` contains the detailed architecture
and file inventory. That directory is intentionally ignored by Git. Training
run evidence uses ``notes.txt``; archived Markdown originals remain in
``documentry/experiments/``.
