from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from heartrisk import config
from heartrisk.app_support import artifacts_ready, get_bundle, pct, signed_pct_points
from heartrisk.explain import find_lower_risk_profile, local_explanation, what_if
from heartrisk.predict import default_patient, example_patients, predict_patient
from heartrisk.report import build_patient_report

FEATURE_SET_LABELS = {
    "early": "Early screening · 11 factors",
    "full": "Full clinical · 17 factors",
}

BAND_HELP = {
    "low": "Below half of the screening threshold.",
    "moderate": "Between half of the threshold and the threshold.",
    "high": "At or above the screening threshold.",
}


def _key(feature_set: str, feature: str) -> str:
    return f"risk_{feature_set}_{feature}"


def _coerce_default(feature: str, value):
    if feature in config.BINARY_COLUMNS:
        return bool(int(value))
    if feature in ("SleepTime", "PhysicalHealth", "MentalHealth"):
        return int(round(float(value)))
    if feature == "BMI":
        return float(value)
    return str(value)


def _init_state(bundle: dict) -> None:
    feature_set = bundle["feature_set"]
    defaults = default_patient(bundle)
    for feature, value in defaults.items():
        key = _key(feature_set, feature)
        if key not in st.session_state:
            st.session_state[key] = _coerce_default(feature, value)


def _clear_result() -> None:
    for key in (
        "risk_result",
        "risk_explanation",
        "risk_patient",
        "risk_result_fs",
        "whatif_result",
        "whatif_result_fs",
        "lower_profile",
        "lower_profile_fs",
    ):
        st.session_state.pop(key, None)


def _apply_example(bundle: dict, name: str) -> None:
    feature_set = bundle["feature_set"]
    examples = example_patients(bundle)
    if name not in examples:
        return
    for feature, value in examples[name].items():
        st.session_state[_key(feature_set, feature)] = _coerce_default(feature, value)
    _clear_result()


def _build_patient(bundle: dict) -> dict:
    feature_set = bundle["feature_set"]
    patient = {
        "AgeCategory": st.session_state[_key(feature_set, "AgeCategory")],
        "Sex": st.session_state[_key(feature_set, "Sex")],
        "Race": st.session_state[_key(feature_set, "Race")],
        "BMI": float(st.session_state[_key(feature_set, "BMI")]),
        "Smoking": int(st.session_state[_key(feature_set, "Smoking")]),
        "AlcoholDrinking": int(st.session_state[_key(feature_set, "AlcoholDrinking")]),
        "PhysicalActivity": int(st.session_state[_key(feature_set, "PhysicalActivity")]),
        "SleepTime": int(st.session_state[_key(feature_set, "SleepTime")]),
        "PhysicalHealth": int(st.session_state[_key(feature_set, "PhysicalHealth")]),
        "MentalHealth": int(st.session_state[_key(feature_set, "MentalHealth")]),
        "GenHealth": st.session_state[_key(feature_set, "GenHealth")],
    }
    if feature_set == "full":
        for feature in (
            "Stroke",
            "Diabetic",
            "DiffWalking",
            "Asthma",
            "KidneyDisease",
            "SkinCancer",
        ):
            value = st.session_state[_key(feature_set, feature)]
            patient[feature] = value if feature == "Diabetic" else int(value)
    return patient


def _gauge(probability: float, threshold: float) -> go.Figure:
    low_cut = threshold * 0.5
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11}},
                "bar": {"color": "#0F766E", "thickness": 0.32},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, low_cut * 100], "color": "#D1FAE5"},
                    {"range": [low_cut * 100, threshold * 100], "color": "#FEF3C7"},
                    {"range": [threshold * 100, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(
        height=280,
        margin={"l": 24, "r": 24, "t": 24, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0F172A"},
    )
    return figure


def _format_value(feature: str, value) -> str:
    if feature in config.BINARY_COLUMNS:
        return "Yes" if int(value) >= 0.5 else "No"
    if feature in config.NUMERIC_COLUMNS:
        return f"{float(value):g}"
    return str(value)


def _format_changes(changes: dict) -> str:
    parts = []
    for feature, value in changes.items():
        parts.append(f"{config.FEATURE_LABELS.get(feature, feature)} → {_format_value(feature, value)}")
    return "; ".join(parts)


def _render_result(bundle: dict, result: dict, explanation: dict, patient: dict) -> None:
    st.divider()
    st.subheader("Estimated risk")
    st.caption(
        "The calibrated probability is the model output. The screening band compares it "
        "with the threshold selected on the validation set."
    )

    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        st.plotly_chart(
            _gauge(result["probability"], result["threshold"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            f"Model probability {pct(result['probability'])} · "
            f"threshold {pct(result['threshold'])} · "
            f"raw score {pct(result['raw_probability'])}"
        )
    with right:
        st.metric(
            "Screening band",
            result["risk_band_label"],
            help=BAND_HELP[result["risk_band"]],
            border=True,
        )
        st.metric(
            "Calibrated probability",
            pct(result["probability"]),
            help="Isotonic calibration maps the raw model score to an observed event rate on held-out validation data.",
            border=True,
        )
        st.metric(
            "Decision threshold",
            pct(result["threshold"]),
            help="The threshold was chosen on validation data to favour recall for a screening-style use case.",
            border=True,
        )

    if result["risk_band"] == "low":
        st.success(
            "The model places this profile below its screening threshold. This does not rule out disease.",
            icon=":material/check_circle:",
        )
    elif result["risk_band"] == "moderate":
        st.warning(
            "The model places this profile near its screening threshold. Consider discussing risk factors with a clinician.",
            icon=":material/warning:",
        )
    else:
        st.error(
            "The model places this profile above its screening threshold. This is a prompt for clinical follow-up, not a diagnosis.",
            icon=":material/priority_high:",
        )

    st.divider()
    st.subheader("Why this estimate?")
    st.caption(
        "SHAP contributions show how each factor moved this individual's raw model output "
        "relative to the model's baseline. Positive values push risk up; negative values push it down. "
        "Sex, race, and diabetes indicators are grouped for readability."
    )

    contributions = explanation["contributions"][:10][::-1]
    labels = [f"{item['label']} ({item['value']})" for item in contributions]
    values = [item["shap"] for item in contributions]
    colors = ["#B91C1C" if value > 0 else "#0F766E" for value in values]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.3f}" for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>SHAP %{x:+.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(360, 38 * len(contributions)),
        margin={"l": 12, "r": 32, "t": 12, "b": 12},
        xaxis_title="SHAP contribution (log-odds)",
        yaxis_title="",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0F172A"},
    )
    figure.update_xaxes(zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.caption(
        f"Baseline raw output: {explanation['base_value']:+.3f} log-odds. "
        "The grouped contributions and baseline sum to this profile's raw model output."
    )

    increase_col, decrease_col = st.columns(2, gap="large")
    with increase_col:
        with st.container(border=True):
            st.markdown("**Factors increasing risk**")
            if explanation["top_increasing"]:
                for item in explanation["top_increasing"][:5]:
                    st.markdown(
                        f"- {item['label']}: {item['value']} ({item['shap']:+.3f})"
                    )
            else:
                st.caption("No factor is pushing this profile above the baseline.")
    with decrease_col:
        with st.container(border=True):
            st.markdown("**Factors decreasing risk**")
            if explanation["top_decreasing"]:
                for item in explanation["top_decreasing"][:5]:
                    st.markdown(
                        f"- {item['label']}: {item['value']} ({item['shap']:+.3f})"
                    )
            else:
                st.caption("No factor is pulling this profile below the baseline.")

    st.divider()
    _render_what_if(bundle, result, patient)
    _render_lower_profile(bundle, result, patient)

    what_if_result = (
        st.session_state.get("whatif_result")
        if st.session_state.get("whatif_result_fs") == bundle["feature_set"]
        else None
    )
    lower_profile = (
        st.session_state.get("lower_profile")
        if st.session_state.get("lower_profile_fs") == bundle["feature_set"]
        else None
    )
    st.divider()
    st.subheader("Printable summary")
    st.caption(
        "Download a self-contained HTML snapshot with the profile, drivers, and any "
        "explorations above. Open it in a browser and print to PDF if needed."
    )
    st.download_button(
        "Download summary report",
        data=build_patient_report(
            result,
            explanation,
            patient,
            what_if=what_if_result,
            lower_profile=lower_profile,
        ),
        file_name="heart-risk-summary.html",
        mime="text/html",
        icon=":material/download:",
        width="stretch",
    )


def _whatif_input(feature: str, patient: dict) -> object:
    feature_set = patient.get("_feature_set", "early")
    key = f"whatif_value_{feature_set}_{feature}"
    current = patient.get(feature)

    if feature in config.BINARY_COLUMNS:
        if key not in st.session_state:
            st.session_state[key] = bool(int(current))
        return int(st.toggle(config.FEATURE_LABELS[feature], key=key))

    if feature == "BMI":
        if key not in st.session_state:
            st.session_state[key] = float(current)
        return float(
            st.number_input(
                "New body mass index",
                min_value=10.0,
                max_value=100.0,
                step=0.1,
                format="%.1f",
                key=key,
            )
        )

    if feature == "SleepTime":
        if key not in st.session_state:
            st.session_state[key] = int(current)
        return int(st.slider("New average sleep (hours)", 1, 24, step=1, key=key))

    if feature in ("PhysicalHealth", "MentalHealth"):
        if key not in st.session_state:
            st.session_state[key] = int(current)
        return int(
            st.slider("New number of days (past 30)", 0, 30, step=1, key=key)
        )

    if feature == "GenHealth":
        if key not in st.session_state or st.session_state[key] not in config.GENHEALTH_CATEGORIES:
            st.session_state[key] = current if current in config.GENHEALTH_CATEGORIES else "Good"
        return st.selectbox("New general health", config.GENHEALTH_CATEGORIES, key=key)

    return current


def _render_what_if(bundle: dict, result: dict, patient: dict) -> None:
    st.subheader("What-if simulator")
    st.caption(
        "Change one modifiable factor and compare the model's calibrated probability. "
        "These are model-based associations, not causal effects or treatment recommendations."
    )

    with st.form("whatif_form", border=True):
        feature = st.selectbox(
            "Factor to change",
            config.MODIFIABLE_FEATURES,
            format_func=lambda item: config.FEATURE_LABELS.get(item, item),
            key=f"whatif_feature_{bundle['feature_set']}",
        )
        new_value = _whatif_input(feature, {**patient, "_feature_set": bundle["feature_set"]})
        submitted = st.form_submit_button(
            "Compare with current profile",
            type="secondary",
            icon=":material/compare_arrows:",
            width="stretch",
        )

    if submitted:
        if new_value == patient.get(feature):
            st.info("Choose a different value to compare.")
            st.session_state.pop("whatif_result", None)
            st.session_state.pop("whatif_result_fs", None)
        else:
            with st.spinner("Recalculating the model..."):
                comparison = what_if(bundle, patient, {feature: new_value})
            st.session_state["whatif_result"] = comparison
            st.session_state["whatif_result_fs"] = bundle["feature_set"]
            st.session_state["whatif_base_probability"] = result["probability"]

    comparison = st.session_state.get("whatif_result")
    if comparison and st.session_state.get("whatif_result_fs") == bundle["feature_set"]:
        base_probability = st.session_state.get("whatif_base_probability", result["probability"])
        st.metric(
            "Updated probability",
            pct(comparison["probability"]),
            delta=signed_pct_points(comparison["delta"]),
            delta_color="inverse",
            help=f"Compared with {pct(base_probability)} for the current profile.",
            border=True,
        )
        st.caption(
            f"Change in probability: {signed_pct_points(comparison['delta'])}. "
            "Negative values mean lower estimated risk."
        )
        movers = pd.DataFrame(comparison["movers"])
        if not movers.empty:
            movers = movers.sort_values("delta")
            figure = go.Figure(
                go.Bar(
                    x=movers["delta"],
                    y=movers["label"],
                    orientation="h",
                    marker_color=[
                        "#B91C1C" if value > 0 else "#0F766E"
                        for value in movers["delta"]
                    ],
                    hovertemplate="%{y}<br>Change %{x:+.3f}<extra></extra>",
                )
            )
            figure.update_layout(
                height=max(260, 34 * len(movers)),
                margin={"l": 12, "r": 24, "t": 12, "b": 12},
                xaxis_title="Change in SHAP contribution",
                yaxis_title="",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#0F172A"},
            )
            figure.update_xaxes(zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
            st.caption(
                "Movers include interaction effects, so factors other than the edited one can shift."
            )


def _render_lower_profile(bundle: dict, result: dict, patient: dict) -> None:
    st.divider()
    st.subheader("Explore a lower-risk profile")
    st.caption(
        "A greedy search over realistic changes to modifiable factors. It is a model-based "
        "what-if exploration, not medical advice or a guaranteed risk reduction."
    )

    if st.button(
        "Search for a lower-risk combination",
        type="secondary",
        icon=":material/trending_down:",
        width="stretch",
    ):
        with st.spinner("Testing realistic combinations..."):
            profile = find_lower_risk_profile(bundle, patient, max_steps=4)
        st.session_state["lower_profile"] = profile
        st.session_state["lower_profile_fs"] = bundle["feature_set"]
        st.session_state["lower_profile_base"] = result["probability"]

    profile = st.session_state.get("lower_profile")
    if not profile or st.session_state.get("lower_profile_fs") != bundle["feature_set"]:
        return

    base_probability = st.session_state.get("lower_profile_base", result["probability"])
    st.metric(
        "Explored profile",
        pct(profile["probability"]),
        delta=signed_pct_points(profile["probability"] - base_probability),
        delta_color="inverse",
        help=f"Started at {pct(base_probability)}.",
        border=True,
    )
    if not profile["steps"]:
        st.info("No further realistic improvements were found within the search limits.")
        return

    for index, step in enumerate(profile["steps"], start=1):
        st.markdown(
            f"**{index}.** {_format_changes(step['changes'])}  \n"
            f"Estimated probability: {pct(step['probability'])} "
            f"({signed_pct_points(step['delta'])})"
        )
    if profile["probability"] > profile["target"]:
        st.caption(
            f"The search did not reach its target of {pct(profile['target'])} within four steps."
        )


if not artifacts_ready():
    st.title("Risk assessment")
    st.error("Model artifacts are not available yet.")
    st.code(
        r".venv\Scripts\python.exe scripts\train_models.py",
        language="powershell",
    )
    st.caption("Run the training command above, then reload this page.")
    st.stop()

st.title("Risk assessment")
st.caption(
    "Estimate heart-disease risk from survey-style factors, then inspect the drivers and "
    "test what-if changes. Educational demonstration only."
)

feature_set = st.segmented_control(
    "Feature set",
    options=list(config.FEATURE_SETS),
    format_func=lambda item: FEATURE_SET_LABELS[item],
    default="early",
    key="risk_feature_set",
    help="The early set is designed for a first-contact screening scenario. The full set adds diagnosed conditions and functional limitations.",
)

if feature_set is None:
    feature_set = "early"

if st.session_state.get("risk_result_fs") not in (None, feature_set):
    _clear_result()

try:
    bundle = get_bundle(feature_set)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

_init_state(bundle)

example_cols = st.columns(2, gap="small")
with example_cols[0]:
    if st.button(
        "Load lower-risk example",
        icon=":material/person:",
        width="stretch",
    ):
        _apply_example(bundle, "low_risk")
        st.rerun()
with example_cols[1]:
    if st.button(
        "Load higher-risk example",
        icon=":material/person_alert:",
        width="stretch",
    ):
        _apply_example(bundle, "high_risk")
        st.rerun()

st.caption(
    "Use the examples for a quick tour, or enter a profile below. The model is not a diagnostic test."
)

with st.form("risk_form", border=True):
    st.subheader("Profile")
    first, second, third = st.columns(3, gap="large")
    with first:
        st.selectbox(
            "Age group",
            config.AGE_CATEGORIES,
            key=_key(feature_set, "AgeCategory"),
        )
        st.selectbox(
            "Sex",
            config.SEX_CATEGORIES,
            key=_key(feature_set, "Sex"),
        )
        st.selectbox(
            "Race / ethnicity",
            config.RACE_CATEGORIES,
            key=_key(feature_set, "Race"),
        )
        st.number_input(
            "Body mass index",
            min_value=10.0,
            max_value=100.0,
            step=0.1,
            format="%.1f",
            key=_key(feature_set, "BMI"),
            help="Weight in kilograms divided by height in metres squared.",
        )
    with second:
        st.toggle(
            "Current or former smoker",
            key=_key(feature_set, "Smoking"),
        )
        st.toggle(
            "Heavy alcohol use",
            key=_key(feature_set, "AlcoholDrinking"),
        )
        st.toggle(
            "Physically active in the past 30 days",
            key=_key(feature_set, "PhysicalActivity"),
        )
        st.slider(
            "Average sleep (hours)",
            1,
            24,
            step=1,
            key=_key(feature_set, "SleepTime"),
        )
    with third:
        st.slider(
            "Poor physical-health days (past 30)",
            0,
            30,
            step=1,
            key=_key(feature_set, "PhysicalHealth"),
        )
        st.slider(
            "Poor mental-health days (past 30)",
            0,
            30,
            step=1,
            key=_key(feature_set, "MentalHealth"),
        )
        st.selectbox(
            "General health",
            config.GENHEALTH_CATEGORIES,
            key=_key(feature_set, "GenHealth"),
        )

    if feature_set == "full":
        st.markdown("**Clinical history**")
        clinical_first, clinical_second, clinical_third = st.columns(3, gap="large")
        with clinical_first:
            st.toggle("Stroke history", key=_key(feature_set, "Stroke"))
            st.toggle("Difficulty walking", key=_key(feature_set, "DiffWalking"))
        with clinical_second:
            st.selectbox(
                "Diabetes status",
                config.DIABETIC_CATEGORIES,
                key=_key(feature_set, "Diabetic"),
            )
            st.toggle("Asthma", key=_key(feature_set, "Asthma"))
        with clinical_third:
            st.toggle("Kidney disease", key=_key(feature_set, "KidneyDisease"))
            st.toggle("Skin cancer", key=_key(feature_set, "SkinCancer"))

    submitted = st.form_submit_button(
        "Estimate risk",
        type="primary",
        icon=":material/monitor_heart:",
        width="stretch",
    )

if submitted:
    patient = _build_patient(bundle)
    with st.spinner("Scoring the profile and computing SHAP contributions..."):
        result = predict_patient(bundle, patient)
        explanation = local_explanation(bundle, patient)
    st.session_state["risk_result"] = result
    st.session_state["risk_explanation"] = explanation
    st.session_state["risk_patient"] = patient
    st.session_state["risk_result_fs"] = feature_set
    st.session_state.pop("whatif_result", None)
    st.session_state.pop("lower_profile", None)

result = st.session_state.get("risk_result")
explanation = st.session_state.get("risk_explanation")
patient = st.session_state.get("risk_patient")
if result and explanation and patient and st.session_state.get("risk_result_fs") == feature_set:
    _render_result(bundle, result, explanation, patient)
else:
    st.info("Enter a profile and select **Estimate risk** to see the model output.", icon=":material/arrow_upward:")

st.divider()
st.caption(config.DISCLAIMER)
