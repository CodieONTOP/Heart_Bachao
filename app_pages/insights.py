from __future__ import annotations

import pandas as pd
import streamlit as st

from heartrisk import config
from heartrisk.app_support import artifacts_ready, ci_text, get_metrics, pct, show_figure

MODEL_LABELS = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}

FEATURE_SET_LABELS = {"early": "Early screening", "full": "Full clinical"}

SUBGROUP_LABELS = {"sex": "Sex", "race": "Race / ethnicity", "age": "Age group"}

st.title("Model insights")
st.caption(
    "Held-out test performance, calibration, global explanations, and subgroup checks for the "
    "early-screening and full-clinical models."
)

if not artifacts_ready():
    st.error("Model artifacts are not available yet.")
    st.code(
        r".venv\Scripts\python.exe scripts\train_models.py",
        language="powershell",
    )
    st.stop()

metrics = get_metrics()
early = metrics["models"]["early"]
full = metrics["models"]["full"]
early_test = early["test"]
full_test = full["test"]
pr_delta = metrics["early_vs_full"]["pr_auc"]
roc_delta = metrics["early_vs_full"]["roc_auc"]

overview = st.columns(4, gap="medium")
overview[0].metric(
    "Early PR-AUC",
    f"{early_test['pr_auc']:.3f}",
    help="Precision-recall AUC is more informative than accuracy when positives are rare.",
    border=True,
)
overview[1].metric(
    "Full PR-AUC",
    f"{full_test['pr_auc']:.3f}",
    delta=f"{pr_delta['delta']:+.3f}",
    help=f"Full-model gain: {ci_text(pr_delta)} (95% bootstrap interval).",
    border=True,
)
overview[2].metric(
    "Early sensitivity",
    pct(early_test["sensitivity"]),
    help=f"At the selected threshold of {pct(early_test['threshold'])}.",
    border=True,
)
overview[3].metric(
    "Full specificity",
    pct(full_test["specificity"]),
    help=f"At the selected threshold of {pct(full_test['threshold'])}.",
    border=True,
)

st.caption(
    f"Test set: {early_test['n']:,} respondents, {early_test['positives']:,} positive cases "
    f"({pct(early_test['prevalence'])} prevalence). Full-model gains: "
    f"ROC-AUC {roc_delta['delta']:+.3f} ({ci_text(roc_delta)}), "
    f"PR-AUC {pr_delta['delta']:+.3f} ({ci_text(pr_delta)})."
)

overview_tab, calibration_tab, explain_tab, fairness_tab, data_tab = st.tabs(
    ["Performance", "Calibration", "SHAP explanations", "Subgroup checks", "Dataset"]
)

with overview_tab:
    st.subheader("Model comparison")
    st.caption(
        "All models use the same preprocessing and validation-selected threshold. "
        "The full set adds diagnosed conditions and functional limitations."
    )
    comparison = pd.DataFrame(metrics["model_comparison"]).copy()
    comparison["feature_set"] = comparison["feature_set"].map(FEATURE_SET_LABELS)
    comparison["model"] = comparison["model"].map(MODEL_LABELS)
    comparison = comparison[
        [
            "feature_set",
            "model",
            "roc_auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "precision",
            "f2",
        ]
    ]
    comparison_display = comparison.copy()
    for column in ("sensitivity", "specificity", "precision"):
        comparison_display[column] = comparison_display[column] * 100
    st.dataframe(
        comparison_display,
        hide_index=True,
        width="stretch",
        column_config={
            "feature_set": "Feature set",
            "model": "Model",
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
            "sensitivity": st.column_config.NumberColumn("Sensitivity (%)", format="%.1f"),
            "specificity": st.column_config.NumberColumn("Specificity (%)", format="%.1f"),
            "precision": st.column_config.NumberColumn("Precision (%)", format="%.1f"),
            "f2": st.column_config.NumberColumn("F2", format="%.3f"),
        },
    )
    show_figure(
        "model_comparison.png",
        "Validation performance across candidate models and feature sets.",
    )
    roc_col, pr_col = st.columns(2, gap="large")
    with roc_col:
        show_figure("roc_pr_early.png", "Early-screening ROC and precision-recall curves.")
    with pr_col:
        show_figure("roc_pr_full.png", "Full-clinical ROC and precision-recall curves.")

with calibration_tab:
    st.subheader("Calibration and threshold")
    st.caption(
        "Isotonic calibration maps raw scores to observed event rates on validation data. "
        "The app reports calibrated probabilities; raw scores are shown for transparency."
    )
    calibration_cols = st.columns(2, gap="large")
    with calibration_cols[0]:
        show_figure("calibration_early.png", "Early-screening calibration curve.")
    with calibration_cols[1]:
        show_figure("calibration_full.png", "Full-clinical calibration curve.")
    st.caption(
        f"Test ECE: early {early_test['ece']:.4f}, full {full_test['ece']:.4f}. "
        "Lower is better; ECE is not a substitute for external validation."
    )
    threshold_cols = st.columns(2, gap="large")
    with threshold_cols[0]:
        show_figure("threshold_early.png", "Validation threshold selection for the early model.")
    with threshold_cols[1]:
        show_figure("threshold_full.png", "Validation threshold selection for the full model.")
    confusion_cols = st.columns(2, gap="large")
    with confusion_cols[0]:
        show_figure("confusion_early.png", "Early-screening confusion matrix at the selected threshold.")
    with confusion_cols[1]:
        show_figure("confusion_full.png", "Full-clinical confusion matrix at the selected threshold.")

with explain_tab:
    st.subheader("What drives the model?")
    st.caption(
        "Global SHAP importance is computed on a sample of held-out test respondents. "
        "One-hot encoded Sex, Race, and Diabetic indicators can appear separately in global plots; "
        "the risk-assessment page groups them for readability."
    )
    shap_feature_set = st.segmented_control(
        "Feature set",
        options=["early", "full"],
        format_func=lambda item: FEATURE_SET_LABELS[item],
        default="early",
        key="insights_shap_feature_set",
    )
    if shap_feature_set is None:
        shap_feature_set = "early"

    importance = pd.DataFrame(
        metrics["models"][shap_feature_set]["shap"]["importance"]
    ).sort_values("mean_abs_shap", ascending=True)
    chart_data = importance.set_index("label")["mean_abs_shap"]
    st.bar_chart(
        chart_data,
        horizontal=True,
        color="#0F766E",
        height=max(320, 30 * len(chart_data)),
        x_label="Mean absolute SHAP value",
        y_label="",
    )
    st.caption(
        "Mean absolute SHAP ranks factors by average influence, but does not show direction."
    )

    shap_cols = st.columns(2, gap="large")
    with shap_cols[0]:
        show_figure(
            f"shap_beeswarm_{shap_feature_set}.png",
            "SHAP summary: direction and spread of feature effects.",
        )
    with shap_cols[1]:
        show_figure(
            f"shap_bar_{shap_feature_set}.png",
            "Global mean absolute SHAP importance.",
        )

    st.subheader("Local examples")
    local_cols = st.columns(2, gap="large")
    with local_cols[0]:
        show_figure(
            f"shap_waterfall_{shap_feature_set}_low.png",
            "Lowest-risk held-out example.",
        )
    with local_cols[1]:
        show_figure(
            f"shap_waterfall_{shap_feature_set}_high.png",
            "Highest-risk held-out example.",
        )

    dependence_paths = sorted(
        config.FIGURES_DIR.glob(f"shap_dependence_{shap_feature_set}_*.png")
    )
    if dependence_paths:
        labels = [
            path.stem.replace(f"shap_dependence_{shap_feature_set}_", "").replace("_", " ")
            for path in dependence_paths
        ]
        selected = st.selectbox("Dependence plot", labels, key=f"dependence_{shap_feature_set}")
        selected_path = dependence_paths[labels.index(selected)]
        st.image(
            str(selected_path),
            caption=f"SHAP dependence for {selected}.",
            width="stretch",
        )

with fairness_tab:
    st.subheader("Subgroup performance")
    st.caption(
        "Descriptive test-set metrics by subgroup. Small groups have wide uncertainty, and "
        "these checks do not establish fairness or clinical safety."
    )
    dimension = st.segmented_control(
        "Subgroup",
        options=["sex", "race", "age"],
        format_func=lambda item: SUBGROUP_LABELS[item],
        default="sex",
        key="fairness_dimension",
    )
    if dimension is None:
        dimension = "sex"

    subgroup_frame = pd.DataFrame(early["subgroups"][dimension]).copy()
    subgroup_frame = subgroup_frame[
        [
            "group",
            "n",
            "positives",
            "prevalence",
            "roc_auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "precision",
        ]
    ]
    subgroup_display = subgroup_frame.copy()
    for column in ("prevalence", "sensitivity", "specificity", "precision"):
        subgroup_display[column] = subgroup_display[column] * 100
    st.dataframe(
        subgroup_display,
        hide_index=True,
        width="stretch",
        column_config={
            "group": SUBGROUP_LABELS[dimension],
            "n": st.column_config.NumberColumn("n", format="%d"),
            "positives": st.column_config.NumberColumn("Positives", format="%d"),
            "prevalence": st.column_config.NumberColumn("Prevalence (%)", format="%.1f"),
            "roc_auc": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
            "pr_auc": st.column_config.NumberColumn("PR-AUC", format="%.3f"),
            "sensitivity": st.column_config.NumberColumn("Sensitivity (%)", format="%.1f"),
            "specificity": st.column_config.NumberColumn("Specificity (%)", format="%.1f"),
            "precision": st.column_config.NumberColumn("Precision (%)", format="%.1f"),
        },
    )
    chart = (
        subgroup_frame.sort_values("sensitivity", ascending=True)
        .set_index("group")[["sensitivity", "specificity"]]
    )
    st.bar_chart(
        chart,
        horizontal=True,
        color=["#0F766E", "#94A3B8"],
        height=max(320, 46 * len(chart)),
        x_label="Rate",
        y_label="",
    )
    st.warning(
        "The early model is not equally sensitive across groups. In the held-out test set, "
        "sensitivity was about 66% for women and 84% for men, and lower for Hispanic and Asian "
        "respondents. Treat subgroup estimates with caution because some groups are small.",
        icon=":material/warning:",
    )

with data_tab:
    st.subheader("Dataset and split")
    dataset = metrics["dataset"]
    split = metrics["split"]
    data_cols = st.columns(4, gap="medium")
    data_cols[0].metric("Rows", f"{dataset['rows']:,}", border=True)
    data_cols[1].metric("Positive cases", f"{dataset['positives']:,}", border=True)
    data_cols[2].metric("Prevalence", pct(dataset["prevalence"]), border=True)
    data_cols[3].metric("Duplicate rows", f"{dataset['duplicate_rows']:,}", border=True)
    st.caption(
        f"Source: CDC BRFSS 2020 cleaned extract via "
        f"[Zenodo record 15364962](https://doi.org/10.5281/zenodo.15364962), CC BY 4.0. "
        f"Missing cells: {dataset['missing_cells']:,}. Duplicate rows were retained because "
        "identical survey answers can still represent different respondents."
    )
    split_frame = pd.DataFrame(
        [
            {
                "split": name.title(),
                "n": values["n"],
                "prevalence": values["prevalence"] * 100,
            }
            for name, values in split.items()
        ]
    )
    st.dataframe(
        split_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "split": "Split",
            "n": st.column_config.NumberColumn("Respondents", format="%d"),
            "prevalence": st.column_config.NumberColumn("Prevalence (%)", format="%.2f"),
        },
    )
    st.caption(
        "Model metrics are reported on the test split only. The validation split was used for "
        "early stopping, calibration, and threshold selection."
    )
    st.caption(
        "The target is self-reported coronary heart disease or myocardial infarction. "
        "It is not a future-event label, so this is a cross-sectional risk-association model."
    )
