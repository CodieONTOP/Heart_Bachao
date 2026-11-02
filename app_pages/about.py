from __future__ import annotations

import streamlit as st

from heartrisk import config
from heartrisk.app_support import artifacts_ready, get_metrics, pct

st.title("About & methods")
st.caption(
    "An educational, explainable machine-learning project on heart-disease risk associations "
    "in a large U.S. health survey."
)

st.warning(
    "This is not a medical device or a diagnostic test. It describes associations in survey "
    "data, not causal effects or guaranteed individual outcomes. Do not use it to make "
    "clinical decisions.",
    icon=":material/health_and_safety:",
)

st.subheader("What the project does")
st.markdown(
    "The project compares an **early-screening** profile with a **full-clinical** profile for "
    "estimating self-reported heart-disease risk. It explains individual predictions with SHAP, "
    "lets users explore model-based what-if changes, scores uploaded CSV batches, and exports a "
    "printable HTML summary. The goal is to show how a calibrated, explainable model can be "
    "built and audited, not to replace a clinician."
)

feature_col, target_col = st.columns(2, gap="large")
with feature_col:
    with st.container(border=True):
        st.markdown("**Early-screening features (11)**")
        st.markdown(
            "\n".join(f"- {config.FEATURE_LABELS[item]}" for item in config.EARLY_FEATURES)
        )
        st.caption(
            "Diagnosed conditions and functional limitations are excluded to reduce reverse "
            "causation and match a first-contact setting."
        )
with target_col:
    with st.container(border=True):
        st.markdown("**Full-clinical features (17)**")
        st.markdown(
            "\n".join(f"- {config.FEATURE_LABELS[item]}" for item in config.FULL_FEATURES)
        )
        st.caption(
            "The extra six factors are diagnosed conditions or functional limitations. They may "
            "improve discrimination but can leak information about existing disease."
        )

st.subheader("Dataset")
st.markdown(
    "The data are a cleaned extract of the CDC **Behavioral Risk Factor Surveillance System "
    "(BRFSS) 2020** survey, distributed through "
    "[Zenodo record 15364962](https://doi.org/10.5281/zenodo.15364962) under CC BY 4.0."
)

if artifacts_ready():
    metrics = get_metrics()
    dataset = metrics["dataset"]
    data_cols = st.columns(4, gap="medium")
    data_cols[0].metric("Respondents", f"{dataset['rows']:,}", border=True)
    data_cols[1].metric("Heart-disease cases", f"{dataset['positives']:,}", border=True)
    data_cols[2].metric("Prevalence", pct(dataset["prevalence"]), border=True)
    data_cols[3].metric("Features", str(len(config.ALL_FEATURES)), border=True)
    st.caption(
        "The target is self-reported coronary heart disease or myocardial infarction. "
        "The survey is cross-sectional, so the model does not predict a future event."
    )
else:
    st.caption("Train the model to populate the dataset and evaluation summary.")

st.subheader("Modeling workflow")
st.markdown(
    """
1. **Validation first.** The cleaned data are split 70/15/15 with stratification. The test
   split is untouched until final reporting.
2. **Preprocessing.** Numeric factors are median-imputed and scaled; age and general health
   are ordinal encoded; sex, race, and diabetes status are one-hot encoded; binary survey
   answers are passed through.
3. **Candidates.** Logistic regression is the baseline, random forest is a benchmark, and
   histogram gradient boosting is the primary model.
4. **Class imbalance.** Class weighting is used instead of resampling. PR-AUC, sensitivity,
   and calibration are emphasized over accuracy.
5. **Calibration.** The selected model is calibrated with isotonic regression on the
   validation split. The app reports calibrated probabilities.
6. **Threshold.** A validation-selected threshold favours recall for a screening-style use
   case. Sensitivity and specificity are reported together.
7. **Explanations.** SHAP values are computed from the tree model. The app aggregates one-hot
   indicators into readable factors and shows both global and local explanations.
8. **Audit.** Bootstrap confidence intervals and sex, race, and age subgroup metrics expose
   uncertainty and performance disparities.
"""
)

if artifacts_ready():
    metrics = get_metrics()
    early = metrics["models"]["early"]["test"]
    full = metrics["models"]["full"]["test"]
    result_cols = st.columns(2, gap="large")
    with result_cols[0]:
        st.metric("Early-screening PR-AUC", f"{early['pr_auc']:.3f}", border=True)
        st.caption(
            f"Sensitivity {pct(early['sensitivity'])} at specificity "
            f"{pct(early['specificity'])}."
        )
    with result_cols[1]:
        st.metric("Full-clinical PR-AUC", f"{full['pr_auc']:.3f}", border=True)
        st.caption(
            f"Sensitivity {pct(full['sensitivity'])} at specificity "
            f"{pct(full['specificity'])}."
        )
    st.caption(
        "The full model improves ranking metrics, but the early model is the more defensible "
        "screening-style comparison because it excludes already-diagnosed conditions."
    )

st.subheader("Limitations")
st.markdown(
    """
- **Cross-sectional data.** The target is prevalent self-reported disease, not a future
  event. Associations are not causal.
- **Self-report and measurement error.** Survey answers can be inaccurate or incomplete.
- **No external validation.** Metrics come from one random holdout split of one survey year.
  Performance and calibration may differ in another population or care setting.
- **Subgroup disparities.** Sensitivity is lower for women and for some racial/ethnic groups
  in the held-out test set. Small subgroup estimates are uncertain.
- **Duplicate rows.** Identical survey responses were retained because they can represent
  different respondents. Deduplication is a sensitivity analysis, not a default.
- **No clinical utility study.** A risk score does not establish benefit, harm, or cost-
  effectiveness. Prospective evaluation would be required.
- **Threshold choice.** The threshold is selected on validation data for a screening-style
  trade-off. Different settings need different thresholds.
"""
)

st.subheader("Reproduce")
st.markdown(
    "From the project root in PowerShell:"
)
st.code(
    r""".venv\Scripts\python.exe scripts\download_data.py
.venv\Scripts\python.exe scripts\train_models.py
.venv\Scripts\streamlit.exe run app.py""",
    language="powershell",
)
st.caption(
    "The download script verifies the MD5 checksum before use. Training writes model bundles, "
    "metrics, and figures under `models/` and `reports/`."
)

st.subheader("Citation and license")
st.markdown(
    "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2020, cleaned extract: "
    "[Zenodo 15364962](https://doi.org/10.5281/zenodo.15364962), CC BY 4.0. "
    "The project code is provided for educational use; review the dataset license before "
    "redistribution."
)

st.divider()
st.caption(config.DISCLAIMER)
