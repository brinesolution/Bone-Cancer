"""Streamlit research demo for checksum-bound HOSP-BoneNet inference."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app.ui_helpers import (
    RESEARCH_DISCLAIMER,
    display_state,
    overlay_images,
    subtype_display_state,
)
from bone_cancer_ai.evaluation.btxrd_review import review_btxrd_run
from bone_cancer_ai.inference.btxrd import (
    BtxrdSubtypePredictor,
    BtxrdSubtypeResult,
    btxrd_result_to_json,
)
from bone_cancer_ai.inference.postprocess import InferenceResult, result_to_json
from bone_cancer_ai.inference.predictor import FrozenPredictor


@st.cache_resource(show_spinner=False)
def load_predictor(project_root: str) -> FrozenPredictor:
    """Load the frozen predictor once per Streamlit process."""

    return FrozenPredictor.from_project_root(Path(project_root))


@st.cache_resource(show_spinner=False)
def load_btxrd_predictor(project_root: str) -> BtxrdSubtypePredictor:
    """Load the separately registered BTXRD subtype predictor once."""

    return BtxrdSubtypePredictor.from_project_root(Path(project_root))


@st.cache_data(show_spinner=False)
def load_btxrd_review(project_root: str) -> dict[str, object]:
    """Load the post-training review artifact without running model inference."""

    root = Path(project_root)
    path = root / "artifacts" / "metrics" / "btxrd_subtype_review.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    review = review_btxrd_run(root)
    if stored != review or review.get("status") != "pass":
        raise ValueError("BTXRD post-training review did not pass")
    return review


def _decode_upload(value: bytes) -> np.ndarray:
    with Image.open(BytesIO(value)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        return np.asarray(image, dtype=np.uint8).copy()


@st.cache_data(show_spinner=False)
def load_model_comparison(project_root: str) -> pd.DataFrame:
    """Load the sealed comparison table without running test inference."""

    path = Path(project_root) / "artifacts" / "metrics" / "final_model_comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"model", "test_samples", "AUROC", "AUPRC", "F1", "ECE", "latency_ms"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"comparison table is missing columns: {sorted(missing)}")
    if frame["model"].duplicated().any() or len(frame) != 5:
        raise ValueError("comparison table must contain five unique frozen models")
    return frame


def _show_result(result: InferenceResult, *, original_image: np.ndarray | None = None) -> None:
    state = display_state(result)
    if state["status"] == "abstain":
        st.warning("Abstain / review required: the conformal set is not a singleton.")
    else:
        st.success(f"Research result: {state['prediction']}")
    left, middle, right = st.columns(3)
    left.metric("Probability", f"{state['probability']:.3f}")
    middle.metric("Confidence", f"{state['confidence']:.3f}")
    right.metric("Uncertainty", f"{state['uncertainty']:.3f}")
    st.write({"conformal_set": state["conformal_set"], "open_set": state["open_set_status"]})

    if original_image is not None:
        st.image(
            original_image,
            caption="Uploaded radiograph (input seen by the app)",
            width="stretch",
        )
    images = overlay_images(result)
    columns = st.columns(len(images))
    labels = {
        "image": "Preprocessed image",
        "spatial_attention": "Unverified lesion-head spatial attention",
        "gradcam": "Grad-CAM (predicted class)",
    }
    for column, (key, image) in zip(columns, images.items(), strict=True):
        column.image(image, caption=labels[key], width="stretch")
    if result.subtype is None:
        st.info("Subtype output unavailable: verified subtype supervision is disabled.")
    else:
        st.write({"subtype": result.subtype})
    if result.concepts is None:
        st.info("Concept output unavailable: verified concept supervision is disabled.")
    else:
        st.write({"concepts": result.concepts})
    st.download_button(
        "Download JSON result",
        data=result_to_json(result),
        file_name="bone_cancer_inference.json",
        mime="application/json",
    )


def _show_btxrd_result(
    result: BtxrdSubtypeResult,
    *,
    original_image: np.ndarray | None = None,
) -> None:
    """Render binary and masked-subtype predictions with provenance and evidence."""

    state = subtype_display_state(result)
    if state["binary_rejected"] or state["subtype_rejected"]:
        st.warning("Review required: a conformal prediction set is ambiguous or empty.")
    else:
        st.success(f"Subtype-aware research result: {state['subtype_prediction']}")
    columns = st.columns(4)
    columns[0].metric("Tumour probability", f"{state['binary_probability']:.3f}")
    columns[1].metric("Binary confidence", f"{state['binary_confidence']:.3f}")
    columns[2].metric("Binary uncertainty", f"{state['binary_uncertainty']:.3f}")
    columns[3].metric("Subtype", state["subtype_prediction"])
    st.write(
        {
            "binary_prediction": state["binary_prediction"],
            "binary_conformal_set": state["binary_conformal_set"],
            "subtype_conformal_set": state["subtype_conformal_set"],
        }
    )
    st.caption(
        f"Run: {result.run_id} · alias: {result.model_alias} · "
        f"checkpoint SHA-256: {result.checkpoint_sha256}"
    )
    if original_image is not None:
        image_columns = st.columns(2)
        image_columns[0].image(
            original_image,
            caption="Uploaded radiograph (original pixels)",
            width="stretch",
        )
        image_columns[1].image(
            result.image,
            caption="Preprocessed image used by HOSP-BoneNet (384×384)",
            width="stretch",
        )
    st.subheader("Subtype probability ranking")
    ranking = pd.DataFrame(result.subtype_top_k)
    ranking["probability"] = ranking["probability"].astype(float).round(4)
    st.dataframe(ranking, hide_index=True, width="stretch")
    images = overlay_images(result)
    columns = st.columns(len(images))
    labels = {
        "image": "Preprocessed image",
        "spatial_attention": "Unverified lesion-head spatial attention",
        "gradcam": "Grad-CAM (binary tumour class)",
    }
    for column, (key, image) in zip(columns, images.items(), strict=True):
        column.image(image, caption=labels[key], width="stretch")
    st.download_button(
        "Download BTXRD subtype JSON result",
        data=btxrd_result_to_json(result),
        file_name="btxrd_subtype_inference.json",
        mime="application/json",
    )


def _show_model_results(project_root: Path) -> None:
    st.subheader("Frozen test-set model results")
    st.caption(
        "Read-only Phase 5/9 artifacts. Opening this tab does not retrain models "
        "or rerun test inference."
    )
    try:
        comparison = load_model_comparison(str(project_root))
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Sealed comparison results are unavailable: {type(error).__name__}.")
        return
    columns = ["model", "test_samples", "AUROC", "AUPRC", "F1", "ECE", "latency_ms"]
    display = comparison.loc[:, columns].copy()
    for column in ("AUROC", "AUPRC", "F1", "ECE", "latency_ms"):
        display[column] = display[column].astype(float).round(4)
    st.dataframe(display, hide_index=True, width="stretch")
    hosp = comparison.loc[comparison["model"] == "HOSP-BoneNet"]
    if not hosp.empty:
        row = hosp.iloc[0]
        left, middle, right = st.columns(3)
        left.metric("HOSP AUROC", f"{row['AUROC']:.4f}")
        middle.metric("HOSP AUPRC", f"{row['AUPRC']:.4f}")
        right.metric("HOSP F1", f"{row['F1']:.4f}")
    figure_root = project_root / "artifacts" / "figures"
    figures = [
        ("ROC curves", "roc_curves.png"),
        ("PR curves", "pr_curves.png"),
        ("Calibration", "calibration_curves.png"),
    ]
    available = [
        (caption, figure_root / filename)
        for caption, filename in figures
        if (figure_root / filename).is_file()
    ]
    if available:
        chart_columns = st.columns(len(available))
        for column, (caption, path) in zip(chart_columns, available, strict=True):
            column.image(str(path), caption=caption, width="stretch")
    with st.expander("BTXRD subtype-aware post-training review"):
        try:
            review = load_btxrd_review(str(project_root))
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(f"BTXRD review is unavailable: {type(error).__name__}.")
        else:
            st.json(review)
            st.caption(
                "Subtype metrics use fresh BTXRD calibration/test rows and are not reused "
                "from the frozen binary test set."
            )
    st.info("Metrics are from the sealed Phase 9 evaluation; no clinical validation is implied.")


def _show_single_image(project_root: Path) -> None:
    uploaded = st.file_uploader("Upload a radiograph", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG or PNG image to begin.")
        return
    uploaded_bytes = uploaded.getvalue()
    try:
        image = _decode_upload(uploaded_bytes)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"The uploaded file is not a readable radiograph: {type(error).__name__}.")
        return
    upload_key = hashlib.sha256(uploaded_bytes).hexdigest()
    if st.session_state.get("upload_key") != upload_key:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_btxrd_result", None)
        st.session_state["upload_key"] = upload_key
    st.image(image, caption="Uploaded radiograph", width="stretch")
    if st.button("Run frozen HOSP-BoneNet", type="primary"):
        try:
            with st.spinner("Running frozen inference and explanations..."):
                result = load_predictor(str(project_root)).predict_array(
                    image, sample_id=uploaded.name
                )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.error(f"Inference could not run: {type(error).__name__}.")
            return
        st.session_state["last_result"] = result
    result = st.session_state.get("last_result")
    if isinstance(result, InferenceResult):
        _show_result(result, original_image=image)
    if st.button("Run BTXRD subtype HOSP-BoneNet"):
        try:
            with st.spinner("Running subtype-aware inference and explanations..."):
                subtype_result = load_btxrd_predictor(str(project_root)).predict_array(
                    image, sample_id=uploaded.name
                )
        except (FileNotFoundError, RuntimeError, ValueError, KeyError, OSError) as error:
            st.error(f"Subtype inference could not run: {type(error).__name__}.")
            return
        st.session_state["last_btxrd_result"] = subtype_result
    subtype_result = st.session_state.get("last_btxrd_result")
    if isinstance(subtype_result, BtxrdSubtypeResult):
        _show_btxrd_result(subtype_result, original_image=image)


def main() -> None:
    st.set_page_config(page_title="HOSP-BoneNet Demo", page_icon="🩻", layout="wide")
    st.title("HOSP-BoneNet radiograph demo")
    st.caption(RESEARCH_DISCLAIMER)
    root = Path(__file__).resolve().parents[1]
    image_tab, results_tab = st.tabs(["Analyze an image", "Model test results"])
    with image_tab:
        _show_single_image(root)
    with results_tab:
        _show_model_results(root)


if __name__ == "__main__":
    main()
