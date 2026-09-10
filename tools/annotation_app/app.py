"""Local metadata review with unknown-aware autosave and immutable revisions."""

import os
from pathlib import Path

import streamlit as st

from bone_cancer_ai.annotations.context import load_annotation_context
from bone_cancer_ai.annotations.schema import STATUSES
from bone_cancer_ai.annotations.workflow import (
    export_annotations,
    import_spatial_review,
    launch_labelme,
    prepare_labelme,
    spatial_frame,
)

st.set_page_config(page_title="Bone tumour annotation review", layout="wide")
config_path = Path(os.environ.get("BONE_ANNOTATION_CONFIG", "configs/annotation.yaml"))


@st.cache_resource
def context(path: str):
    return load_annotation_context(Path(path))


ctx = context(str(config_path.resolve()))
state = st.session_state
state.setdefault("index", 0)
row = ctx.manifest.iloc[state.index].to_dict()


def load_record():
    record = ctx.store.latest(row["sample_id"]) or {}
    state["expected_revision"] = record.get("revision", 0)
    for field in ("binary_review", "subtype_label", "risk_label"):
        value = record.get(field)
        state[field] = "unknown" if value is None else value
    state["reviewer"] = record.get("reviewer", state.get("reviewer", ""))
    state["notes"] = record.get("notes", "")
    state["review_status"] = record.get("review_status", "draft")
    for name in ctx.config["concept_names"]:
        value = record.get("concepts", {}).get(name)
        state[f"known_{name}"] = value is not None
        state[f"present_{name}"] = value == 1
    state["loaded_sample"] = row["sample_id"]
    state["save_error"] = ""


def save():
    try:
        ctx.image_path(row)
        payload = {
            key: state[key]
            for key in (
                "binary_review",
                "subtype_label",
                "risk_label",
                "reviewer",
                "review_status",
                "notes",
            )
        }
        payload["concepts"] = {
            name: int(state[f"present_{name}"]) if state[f"known_{name}"] else None
            for name in ctx.config["concept_names"]
        }
        record = ctx.store.save(
            row["sample_id"],
            row["image_id"],
            row["sha256"],
            payload,
            expected_revision=state.expected_revision,
        )
        state.expected_revision = record["revision"]
        state.save_error = ""
    except (ValueError, OSError) as exc:
        state.save_error = str(exc)


def navigate(offset):
    if state.get("save_error"):
        return
    state.index = max(0, min(len(ctx.manifest) - 1, state.index + offset))


if state.get("loaded_sample") != row["sample_id"]:
    load_record()

st.title("Bone tumour annotation review")
st.caption("Research annotations · Source labels are shown separately · Unknown is a valid answer")
progress = ctx.store.progress(len(ctx.manifest))
st.write(
    f"Image {state.index + 1} of {len(ctx.manifest)} · "
    f"{progress['saved']} saved · {progress['reviewed']} marked reviewed"
)
left, right = st.columns([1.2, 1])
with left:
    st.write(f"Sample ID: {row['sample_id']}")
    st.caption(f"Image ID: {row['image_id']}")
    try:
        st.image(str(ctx.image_path(row)), width="stretch")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()
    st.caption(
        f"Original source binary label: {row.get('binary_label', 'Unavailable')}. "
        "This value is not copied into your review."
    )
with right:
    st.text_input("Reviewer", key="reviewer", on_change=save)
    st.selectbox("Binary review", ["unknown", 0, 1], key="binary_review", on_change=save)
    st.selectbox(
        "Subtype",
        ["unknown", *ctx.schema.get("subtype", {}).get("classes", [])],
        key="subtype_label",
        on_change=save,
    )
    st.selectbox(
        "Risk category",
        ["unknown", *ctx.schema.get("risk", {}).get("classes", [])],
        key="risk_label",
        on_change=save,
    )
    st.selectbox("Review status", STATUSES, key="review_status", on_change=save)
    st.caption(
        "For each concept, leave Assessed unchecked for unknown; check Present only if observed."
    )
    for name in ctx.config["concept_names"]:
        first, second = st.columns(2)
        first.checkbox(f"{name.replace('_', ' ')} — assessed", key=f"known_{name}", on_change=save)
        second.checkbox(
            "Present", key=f"present_{name}", on_change=save, disabled=not state[f"known_{name}"]
        )
    st.text_area("Notes", key="notes", on_change=save)
    if state.save_error:
        st.error(state.save_error)
        st.button("Reload saved revision / discard unsaved edits", on_click=load_record)
    elif state.expected_revision:
        st.success(f"Saved revision {state.expected_revision}")
    else:
        st.info("Enter your reviewer name to start autosaving. No review has been created.")
    previous, following = st.columns(2)
    previous.button(
        "Previous", key="previous", on_click=navigate, args=(-1,), disabled=state.index == 0
    )
    following.button(
        "Next",
        key="next",
        on_click=navigate,
        args=(1,),
        disabled=state.index == len(ctx.manifest) - 1,
    )

with st.expander("Lesion polygons and boxes — LabelMe"):
    st.write(
        "Stage a local copy, draw polygons or rectangles with the class lesion, save in LabelMe, "
        "then import a revision here. Boxes alone do not create segmentation masks."
    )
    if st.button("Prepare editable copy"):
        try:
            staged = prepare_labelme(ctx, row)
            st.info(f"Editable copy: {staged['image_path']}")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if st.button("Open LabelMe"):
        try:
            launch_labelme(ctx, row)
            st.info("LabelMe opened on this computer. Save your drawing before importing.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    if st.button("Import saved spatial revision"):
        try:
            record = import_spatial_review(ctx, row, state.reviewer, state.notes)
            st.success(f"Imported spatial revision {record['revision_id']}")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    spatial = spatial_frame(ctx.root)
    current_spatial = spatial.loc[spatial.sample_id.eq(row["sample_id"])]
    if not current_spatial.empty:
        st.image(current_spatial.iloc[0].overlay_path, caption="Latest imported spatial overlay")
    st.caption(f"{len(spatial)} images have imported project spatial reviews.")

st.download_button(
    "Export master annotation CSV",
    export_annotations(ctx).to_csv(index=False),
    file_name="image_labels.csv",
    mime="text/csv",
)
with st.expander("Revision history"):
    st.json(ctx.store.history(row["sample_id"]))
