# Early Heart Disease Risk Prediction with Explainable ML

An educational, end-to-end machine-learning project that estimates heart-disease risk from
survey-style health factors, compares an early-screening profile with a fuller clinical profile,
and explains individual predictions with SHAP.

The app is a demonstration, **not a medical device**. It describes associations in cross-sectional
survey data, not causal effects or future clinical events.

## Results

Held-out test set: 47,970 respondents, 4,106 positive cases (8.56% prevalence).

| Feature set | Predictors | PR-AUC | ROC-AUC | Sensitivity | Specificity | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early screening | 11 | 0.300 | 0.830 | 76.9% | 74.0% | 10.5% |
| Full clinical | 17 | 0.337 | 0.841 | 77.1% | 75.5% | 10.5% |

Adding diagnosed conditions and functional limitations improves ranking metrics:

- PR-AUC gain: **+0.037** (95% bootstrap interval 0.027 to 0.045)
- ROC-AUC gain: **+0.012** (95% bootstrap interval 0.009 to 0.014)

The early model is the more defensible screening-style comparison because it excludes factors
that may already reflect diagnosed disease. The full model is useful as an upper-bound benchmark,
not as a deployable early-warning model.

### Subgroup signal

Performance is not uniform. In the early-model test results, sensitivity was about **66.4% for
women** and **84.4% for men**. Sensitivity was **58.5% for Hispanic** and **50.0% for Asian**
respondents, although those groups are smaller (4,109 and 1,227 test respondents). These are
descriptive estimates with real uncertainty, not proof of fairness or clinical safety.

## Dataset

- CDC **Behavioral Risk Factor Surveillance System (BRFSS) 2020**, cleaned extract.
- Source: [Zenodo record 15364962](https://doi.org/10.5281/zenodo.15364962), CC BY 4.0.
- 319,795 rows, 18 columns, 27,373 positive cases.
- Target: self-reported coronary heart disease or myocardial infarction (`HeartDisease`).
- 18,078 duplicate rows are retained because identical survey answers can still represent
  different respondents.
- No missing cells in the cleaned extract.

The raw CSV and cleaned parquet are intentionally excluded from version control. Download them
with the provided script.

## Feature sets

**Early screening (11):** age group, sex, race/ethnicity, BMI, smoking, heavy alcohol use,
physical activity, sleep hours, poor physical-health days, poor mental-health days, general health.

**Full clinical (17):** the early set plus stroke history, diabetes status, difficulty walking,
asthma, kidney disease, and skin cancer.

The early set deliberately excludes diagnosed conditions and functional limitations to reduce
reverse causation and to match a first-contact screening scenario.

## Modeling

- Stratified 70/15/15 train/validation/test split, seed 42.
- Preprocessing inside a scikit-learn pipeline:
  - median imputation and scaling for numeric factors;
  - ordinal encoding for age and general health;
  - one-hot encoding for sex, race, and diabetes status;
  - pass-through for binary survey answers.
- Candidate models: logistic regression, random forest, and histogram gradient boosting.
- Class imbalance handled with class weights; no SMOTE.
- Histogram gradient boosting is the selected model, chosen on validation PR-AUC.
- Isotonic calibration on the validation split.
- Validation-selected screening threshold; the delivered app reports the calibration-adjusted
  probability and the threshold together.
- Metrics include PR-AUC, ROC-AUC, Brier score, sensitivity, specificity, precision, F2, and
  bootstrap confidence intervals.
- SHAP `TreeExplainer` for global and local explanations.
- A greedy, model-based what-if search over realistic modifiable-factor changes.

## App

The Streamlit app has four pages:

1. **Risk assessment** — enter a profile, get a calibrated probability and risk band, inspect
   grouped SHAP contributions, test one-factor what-if changes, explore a lower-risk
   combination, and download a printable HTML summary.
2. **Batch scoring** — upload a CSV of survey-style profiles, validate and score every row,
   review the probability distribution and top driver, and download the scored CSV.
3. **Model insights** — compare models, inspect calibration and thresholds, view global SHAP
   plots, and review subgroup performance.
4. **About & methods** — dataset provenance, workflow, limitations, citation, and reproduction
   commands.

Batch scoring accepts either survey labels (`Yes`/`No`, `White`, `Very good`, …) or the encoded
values in the cleaned dataset. A one-row template is available for download on the page. The
scored download adds calibrated probability, raw score, screening band, and the largest grouped
SHAP driver for each row.

## Quickstart

Windows PowerShell, from the project root:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_data.py
python scripts\train_models.py
streamlit run app.py
```

If script execution is restricted, call the executables directly:

```powershell
.venv\Scripts\python.exe scripts\download_data.py
.venv\Scripts\python.exe scripts\train_models.py
.venv\Scripts\streamlit.exe run app.py
```

The download script verifies the 23,264,546-byte file against MD5
`f5a885ff39a113e797af67367b6bc2cf` before use. Training writes model bundles and metadata under
`models/`, and metrics/figures under `reports/`.

## Deployment

The container image runs the app as a non-root user and exposes a Streamlit health check on
`/_stcore/health`. Build it after generating `models/`, `reports/`, and `data/` locally, because
those artifacts are not downloaded during the image build.

```powershell
docker build -t project-heart .
docker run --rm -p 8501:8501 project-heart
```

Or with Compose:

```powershell
docker compose up --build
```

GitHub Actions (`.github/workflows/ci.yml`) runs Ruff, the pytest suite, and a Docker image build
on pushes and pull requests. Tests that need trained artifacts skip automatically when the
`models/` directory is absent.

## Tests and lint

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\ruff.exe check .
```

The test suite covers feature-set definitions, cleaned-data schema, stratified splits, model
bundles, prediction ranges, SHAP additivity/grouping, what-if behavior, metrics invariants, and
Streamlit page rendering. Tests that need trained artifacts skip with a clear message when those
artifacts are absent.

## Project structure

```text
app.py                      Streamlit entry point
app_pages/
  risk.py                   Risk assessment, SHAP, what-if, report download
  batch.py                  CSV batch validation, scoring, and export
  insights.py               Performance, calibration, fairness, SHAP
  about.py                  Methods, limitations, citation
heartrisk/
  config.py                 Paths, feature sets, labels, categories
  data.py                   Download, validation, cleaning, splits
  features.py               Preprocessing and label helpers
  evaluate.py               Metrics, bootstrap intervals, plots
  explain.py                SHAP, what-if, and batch driver logic
  batch.py                  Batch validation, scoring, and CSV templates
  report.py                 Printable HTML summary
  predict.py                Bundle loading and patient scoring
  train.py                  Training, calibration, evaluation orchestration
  app_support.py            Cached app data access and formatting
scripts/
  download_data.py          Checksum-verified download and cleaning
  train_models.py           Training CLI
tests/                      Pytest suite
notebooks/                  Result exploration notebook
models/                     Generated bundles and metadata
reports/                    Generated metrics and figures
Dockerfile                  Non-root app image with health check
docker-compose.yml          Local container deployment
.github/workflows/ci.yml    Lint, test, and Docker build CI
LICENSE                     MIT license for the project code
```

## Reproducibility notes

- Seed: `42`.
- Python: `3.14.5`.
- Key libraries: scikit-learn `1.9.0`, pandas `3.0.5`, NumPy `2.5.2`, SHAP `0.54.1`,
  Streamlit `1.62.0`.
- Model selection and thresholding use the validation split; the test split is used only for
  final reporting.
- The notebook reads the generated metrics and figures rather than retraining by default.

## Limitations

- **Cross-sectional target.** The outcome is prevalent self-reported disease, not a future event.
- **Self-report and measurement error.** Survey responses can be inaccurate.
- **No external validation.** One random holdout from one survey year is not evidence of
  transportability or clinical utility.
- **Subgroup disparities.** Sensitivity differs by sex and race/ethnicity; small groups have wide
  uncertainty.
- **Duplicate rows.** Retained by default; deduplication should be a sensitivity analysis.
- **No causal claims.** What-if changes are model-based associations, not treatment effects.
- **Not a medical device.** No diagnosis, treatment, or prevention decision should rely on it.

## Citation and license

**Code:** MIT License, copyright 2026 Project Heart contributors. See [`LICENSE`](LICENSE).

**Data:** CDC BRFSS 2020, cleaned extract,
[Zenodo 15364962](https://doi.org/10.5281/zenodo.15364962), CC BY 4.0. The dataset license is
separate from the code license; review it before redistributing the data or derived artifacts.
