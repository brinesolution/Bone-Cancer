"""Streamlit research demo for checksum-bound HOSP-BoneNet inference."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app.ui_helpers import RESEARCH_DISCLAIMER, display_state, overlay_images
from bone_cancer_ai.inference.postprocess import InferenceResult, result_to_json
from bone_cancer_ai.inference.predictor import FrozenPredictor


@st.cache_resource(show_spinner=False)
def load_predictor(project_root: str) -> FrozenPredictor:
    """Load the frozen predictor once per Streamlit process."""

    return FrozenPredictor.from_project_root(Path(project_root))


def _decode_upload(value: bytes) -> np.ndarray:
    with Image.open(BytesIO(value)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        return np.asarray(image, dtype=np.uint8).copy()


def _show_result(result: InferenceResult) -> None:
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

    images = overlay_images(result)
    columns = st.columns(len(images))
    labels = {
        "image": "Preprocessed image",
        "spatial_attention": "Unverified lesion-head spatial attention",
        "gradcam": "Grad-CAM (predicted class)",
    }
    for column, (key, image) in zip(columns, images.items(), strict=True):
        column.image(image, caption=labels[key], use_container_width=True)
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


def main() -> None:
    st.set_page_config(page_title="HOSP-BoneNet Demo", page_icon="🩻", layout="wide")
    st.title("HOSP-BoneNet radiograph demo")
    st.caption(RESEARCH_DISCLAIMER)
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
        st.session_state["upload_key"] = upload_key
    st.image(image, caption="Uploaded radiograph", use_container_width=True)
    if st.button("Run frozen HOSP-BoneNet", type="primary"):
        root = Path(__file__).resolve().parents[1]
        try:
            with st.spinner("Running frozen inference and explanations..."):
                result = load_predictor(str(root)).predict_array(image, sample_id=uploaded.name)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            st.error(f"Inference could not run: {type(error).__name__}.")
            return
        st.session_state["last_result"] = result
    result = st.session_state.get("last_result")
    if isinstance(result, InferenceResult):
        _show_result(result)


if __name__ == "__main__":
    main()
