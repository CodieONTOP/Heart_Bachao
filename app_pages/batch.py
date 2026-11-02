from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heartrisk import config
from heartrisk.app_support import artifacts_ready, get_bundle, pct
from heartrisk.batch import (
    BatchValidationError,
    prepare_batch,
    score_batch,
    template_frame,
)
from heartrisk.explain import batch_top_drivers

FEATURE_SET_LABELS = {
    "early": "Early screening · 11 factors",
    "full": "Full clinical · 17 factors",
}

if not artifacts_ready():
    st.title("Batch scoring")
    st.error("Model artifacts are not available yet.")
    st.code(
        r".venv\Scripts\python.exe scripts\train_models.py",
        language="powershell",
    )
    st.caption("Run the training command above, then reload this page.")
    st.stop()

st.title("Batch scoring")
st.caption(
    "Upload a CSV of survey-style profiles, score every row with the calibrated model, and "
    "download the results. Educational demonstration only."
)

feature_set = st.segmented_control(
    "Feature set",
    options=list(config.FEATURE_SETS),
    format_func=lambda item: FEATURE_SET_LABELS[item],
    default="early",
    key="batch_feature_set",
    help="Scores must use the same feature set as the uploaded columns.",
)
if feature_set is None:
    feature_set = "early"

try:
    bundle = get_bundle(feature_set)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

template = template_frame(bundle)
st.download_button(
    "Download CSV template",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name=f"heart_risk_template_{feature_set}.csv",
    mime="text/csv",
    icon=":material/download:",
    help="A one-row example with readable survey labels. Yes/No and 1/0 are both accepted.",
)

uploaded = st.file_uploader(
    "Upload respondent CSV",
    type=["csv"],
    help=(
        "Required columns: " + ", ".join(bundle["features"]) + ". Extra columns are ignored."
    ),
)

if uploaded is None:
    st.info(
        "Upload a CSV to begin, or download the template and replace the example row. "
        "Target labels are optional and ignored if present.",
        icon=":material/upload_file:",
    )
    st.divider()
    st.caption(config.DISCLAIMER)
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:  # pragma: no cover - depends on uploaded file
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

try:
    prepared = prepare_batch(raw, feature_set)
except BatchValidationError as exc:
    st.error("The uploaded file could not be scored.")
    st.code(str(exc), language="text")
    st.caption("Fix the rows above and upload the file again.")
    st.divider()
    st.caption(config.DISCLAIMER)
    st.stop()

for warning in prepared["warnings"]:
    st.warning(warning, icon=":material/warning:")

signature = (
    f"{uploaded.name}:{prepared['rows']}:"
    f"{int(pd.util.hash_pandas_object(prepared['frame'], index=False).sum())}"
)

st.caption(
    f"Validated {prepared['rows']:,} row(s) from a file with {prepared['source_rows']:,} row(s)."
)
with st.expander("Preview validated input", expanded=False):
    st.dataframe(prepared["frame"].head(20), hide_index=True, width="stretch")

if st.button(
    "Score batch",
    type="primary",
    icon=":material/play_arrow:",
    width="stretch",
):
    with st.spinner("Scoring every row and computing drivers..."):
        scored = score_batch(bundle, prepared["frame"])
        drivers = batch_top_drivers(bundle, prepared["frame"])
        scored = pd.concat([scored, drivers], axis=1)
    st.session_state["batch_result"] = scored
    st.session_state["batch_result_fs"] = feature_set
    st.session_state["batch_signature"] = signature
    st.session_state["batch_threshold"] = float(bundle["threshold"])

scored = st.session_state.get("batch_result")
if (
    scored is None
    or st.session_state.get("batch_result_fs") != feature_set
    or st.session_state.get("batch_signature") != signature
):
    st.info("Select **Score batch** to calculate probabilities and drivers.", icon=":material/arrow_upward:")
    st.divider()
    st.caption(config.DISCLAIMER)
    st.stop()

threshold = float(st.session_state.get("batch_threshold", bundle["threshold"]))
high_label = config.RISK_BAND_LABELS["high"]
high_share = float((scored["risk_band"] == high_label).mean())

metric_cols = st.columns(4, gap="medium")
metric_cols[0].metric("Rows scored", f"{len(scored):,}", border=True)
metric_cols[1].metric("Mean probability", pct(float(scored["risk_probability"].mean())), border=True)
metric_cols[2].metric(
    "At or above threshold",
    f"{int((scored['risk_band'] == high_label).sum()):,}",
    delta=pct(high_share),
    delta_color="off",
    help=f"Threshold {pct(threshold)}.",
    border=True,
)
metric_cols[3].metric("Decision threshold", pct(threshold), border=True)

histogram = go.Figure(
    go.Histogram(
        x=scored["risk_probability"],
        nbinsx=40,
        marker_color="#0F766E",
        hovertemplate="Probability %{x:.3f}<br>Rows %{y}<extra></extra>",
    )
)
histogram.add_vline(
    x=threshold,
    line_dash="dash",
    line_color="#B91C1C",
    annotation_text="Screening threshold",
    annotation_position="top right",
)
histogram.update_layout(
    height=320,
    margin={"l": 12, "r": 12, "t": 24, "b": 12},
    xaxis_title="Calibrated probability",
    yaxis_title="Rows",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#0F172A"},
    bargap=0.04,
)
st.plotly_chart(histogram, width="stretch", config={"displayModeBar": False})

band_counts = (
    scored["risk_band"].value_counts().rename_axis("band").reset_index(name="rows")
)
band_counts["share"] = band_counts["rows"] / len(scored) * 100
st.dataframe(
    band_counts,
    hide_index=True,
    width="stretch",
    column_config={
        "band": "Screening band",
        "rows": st.column_config.NumberColumn("Rows", format="%d"),
        "share": st.column_config.NumberColumn("Share (%)", format="%.1f"),
    },
)

st.subheader("Scored results")
display_columns = [*bundle["features"], "risk_probability", "raw_probability", "risk_band"]
if "top_driver" in scored.columns:
    display_columns += ["top_driver", "top_driver_effect", "top_driver_shap"]
preview = scored[display_columns].head(200).copy()
for column in ("risk_probability", "raw_probability"):
    preview[column] = preview[column] * 100
st.dataframe(
    preview,
    hide_index=True,
    width="stretch",
    column_config={
        "risk_probability": st.column_config.NumberColumn("Probability (%)", format="%.2f"),
        "raw_probability": st.column_config.NumberColumn("Raw score (%)", format="%.2f"),
        "risk_band": "Screening band",
        "top_driver": "Top driver",
        "top_driver_effect": "Effect",
        "top_driver_shap": st.column_config.NumberColumn("Driver SHAP", format="%+.3f"),
    },
)
if len(scored) > 200:
    st.caption("Preview shows the first 200 rows. The download includes every row.")

st.download_button(
    "Download scored CSV",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name=f"heart_risk_scored_{feature_set}.csv",
    mime="text/csv",
    icon=":material/download:",
    type="primary",
    width="stretch",
)

st.divider()
st.caption(config.DISCLAIMER)
