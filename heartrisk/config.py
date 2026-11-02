"""Project-wide configuration and feature definitions."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def resolve_models_dir() -> Path:
    """Allow tests/CI to point the app at a different artifact directory."""
    return Path(os.environ.get("HEARTRISK_MODEL_DIR", PROJECT_ROOT / "models"))


MODELS_DIR = PROJECT_ROOT / "models"
RAW_CSV = RAW_DIR / "heart_2020_cleaned.csv"
CLEAN_PARQUET = PROCESSED_DIR / "heart_2020_clean.parquet"
METRICS_JSON = REPORTS_DIR / "metrics.json"
METADATA_JSON = MODELS_DIR / "metadata.json"

DATA_URL = "https://zenodo.org/records/15364962/files/heart_2020_cleaned.csv"
DATA_MD5 = "f5a885ff39a113e797af67367b6bc2cf"
DATA_SIZE_BYTES = 23_264_546

TARGET = "HeartDisease"
SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15

BINARY_COLUMNS = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "PhysicalActivity",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

NUMERIC_COLUMNS = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

ONE_HOT_COLUMNS = ["Sex", "Race", "Diabetic"]
ORDINAL_COLUMNS = ["AgeCategory", "GenHealth"]

AGE_CATEGORIES = [
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80 or older",
]

GENHEALTH_CATEGORIES = ["Excellent", "Very good", "Good", "Fair", "Poor"]

RACE_CATEGORIES = [
    "White",
    "Black",
    "Asian",
    "American Indian/Alaskan Native",
    "Hispanic",
    "Other",
]

SEX_CATEGORIES = ["Female", "Male"]

DIABETIC_CATEGORIES = [
    "No",
    "No, borderline diabetes",
    "Yes",
    "Yes (during pregnancy)",
]

ALL_FEATURES = [
    "BMI",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "PhysicalHealth",
    "MentalHealth",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "SleepTime",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

EXPECTED_COLUMNS = [TARGET, *ALL_FEATURES]

# Early screening deliberately excludes diagnosed conditions and functional
# limitations to reduce reverse causation and match a first-contact setting.
EARLY_FEATURES = [
    "AgeCategory",
    "Sex",
    "Race",
    "BMI",
    "Smoking",
    "AlcoholDrinking",
    "PhysicalActivity",
    "SleepTime",
    "PhysicalHealth",
    "MentalHealth",
    "GenHealth",
]

FULL_FEATURES = [
    *EARLY_FEATURES,
    "Stroke",
    "Diabetic",
    "DiffWalking",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

FEATURE_SETS = {
    "early": EARLY_FEATURES,
    "full": FULL_FEATURES,
}

FEATURE_LABELS = {
    "BMI": "Body mass index",
    "Smoking": "Smoking history",
    "AlcoholDrinking": "Heavy alcohol use",
    "Stroke": "Stroke history",
    "PhysicalHealth": "Poor physical-health days",
    "MentalHealth": "Poor mental-health days",
    "DiffWalking": "Difficulty walking",
    "Sex": "Sex",
    "AgeCategory": "Age group",
    "Race": "Race",
    "Diabetic": "Diabetes status",
    "PhysicalActivity": "Physical activity",
    "GenHealth": "General health",
    "SleepTime": "Sleep hours",
    "Asthma": "Asthma",
    "KidneyDisease": "Kidney disease",
    "SkinCancer": "Skin cancer",
}

MODIFIABLE_FEATURES = [
    "BMI",
    "Smoking",
    "AlcoholDrinking",
    "PhysicalActivity",
    "SleepTime",
    "PhysicalHealth",
    "MentalHealth",
    "GenHealth",
]

INPUT_RANGES = {
    "BMI": (10.0, 100.0),
    "PhysicalHealth": (0, 30),
    "MentalHealth": (0, 30),
    "SleepTime": (0, 24),
}

RISK_BAND_LABELS = {
    "low": "Lower risk",
    "moderate": "Moderate risk",
    "high": "Higher risk",
}

DISCLAIMER = (
    "This tool is for education and portfolio demonstration only. It is not a "
    "medical device, does not provide a diagnosis, and must not replace advice "
    "from a qualified clinician. Predictions describe associations in survey "
    "data, not causal effects or individual future outcomes."
)
