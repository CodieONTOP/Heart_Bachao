"""Validate and score batches of survey-style respondent rows."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from heartrisk import config
from heartrisk.predict import default_patient, risk_band

BATCH_OUTPUT_COLUMNS = [
    "risk_probability",
    "raw_probability",
    "risk_band",
    "top_driver",
    "top_driver_effect",
    "top_driver_shap",
]

_BINARY_TRUE = {"yes", "y", "true", "t", "1"}
_BINARY_FALSE = {"no", "n", "false", "f", "0"}
_MAX_MESSAGES = 20


class BatchValidationError(ValueError):
    """Raised when an uploaded batch cannot be scored as requested."""


def _parse_binary(value) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if float(value) == 1.0:
            return 1
        if float(value) == 0.0:
            return 0
        return None
    text = str(value).strip().casefold()
    if text in _BINARY_TRUE:
        return 1
    if text in _BINARY_FALSE:
        return 0
    return None


def _parse_category(value, categories: list[str]) -> str | None:
    if pd.isna(value):
        return None
    lookup = {category.casefold(): category for category in categories}
    return lookup.get(str(value).strip().casefold())


def _summary(messages: list[str]) -> str:
    if len(messages) <= _MAX_MESSAGES:
        return "\n".join(messages)
    remaining = len(messages) - _MAX_MESSAGES
    return "\n".join([*messages[:_MAX_MESSAGES], f"... and {remaining} more problem(s)."])


def prepare_batch(df: pd.DataFrame, feature_set: str) -> dict:
    """Normalise an uploaded frame and report validation problems.

    Accepts survey labels (``Yes``/``No`` and category strings) or the encoded
    values used by the cleaned dataset. Extra columns are ignored so a raw
    BRFSS extract can be scored without preprocessing.
    """
    if feature_set not in config.FEATURE_SETS:
        raise BatchValidationError(f"Unknown feature set: {feature_set}")
    features = config.FEATURE_SETS[feature_set]

    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise BatchValidationError(
            "Uploaded file is missing required columns: " + ", ".join(missing)
        )

    frame = frame[features]
    blank = frame.isna().all(axis=1)
    frame = frame.loc[~blank]
    if frame.empty:
        raise BatchValidationError("Uploaded file has no data rows to score.")

    errors: list[str] = []
    warnings: list[str] = []
    parsed: dict[str, pd.Series] = {}

    for column in features:
        series = frame[column]
        if column in config.BINARY_COLUMNS:
            values = series.map(_parse_binary)
            bad = values.isna()
            for index in series.index[bad]:
                errors.append(
                    f"Row {index + 2}: {column} must be Yes/No or 1/0 "
                    f"(got {series.loc[index]!r})."
                )
            parsed[column] = values.astype("float64")

        elif column in config.NUMERIC_COLUMNS:
            values = pd.to_numeric(series, errors="coerce")
            bad = values.isna()
            for index in series.index[bad]:
                errors.append(
                    f"Row {index + 2}: {column} must be numeric "
                    f"(got {series.loc[index]!r})."
                )
            low, high = config.INPUT_RANGES[column]
            clipped = int(((values < low) | (values > high)).sum())
            if clipped:
                warnings.append(
                    f"{clipped:,} {column} value(s) were clipped to the supported "
                    f"range [{low:g}, {high:g}]."
                )
            parsed[column] = values.clip(low, high)

        else:
            categories = {
                "AgeCategory": config.AGE_CATEGORIES,
                "GenHealth": config.GENHEALTH_CATEGORIES,
                "Sex": config.SEX_CATEGORIES,
                "Race": config.RACE_CATEGORIES,
                "Diabetic": config.DIABETIC_CATEGORIES,
            }[column]
            values = series.map(partial(_parse_category, categories=categories))
            bad = values.isna()
            for index in series.index[bad]:
                errors.append(
                    f"Row {index + 2}: {column} must be one of {categories} "
                    f"(got {series.loc[index]!r})."
                )
            parsed[column] = values.astype("string")

    if errors:
        raise BatchValidationError(_summary(errors))

    normalised = pd.DataFrame(parsed, index=frame.index)[features].reset_index(drop=True)
    return {
        "frame": normalised,
        "rows": int(len(normalised)),
        "source_rows": int(len(df)),
        "warnings": warnings,
    }


def score_batch(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with calibrated probabilities and bands."""
    features = bundle["features"]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise BatchValidationError(
            "Batch frame is missing required features: " + ", ".join(missing)
        )

    rows = frame[features].reset_index(drop=True)
    transformed = bundle["preprocessor"].transform(rows)
    calibrated = bundle["calibrated_model"].predict_proba(transformed)[:, 1]
    raw = bundle["base_model"].predict_proba(transformed)[:, 1]
    threshold = float(bundle["threshold"])

    scored = rows.copy()
    scored["risk_probability"] = np.round(calibrated, 4)
    scored["raw_probability"] = np.round(raw, 4)
    scored["risk_band"] = [
        config.RISK_BAND_LABELS[risk_band(float(probability), threshold)]
        for probability in calibrated
    ]
    return scored


def template_frame(bundle: dict) -> pd.DataFrame:
    """Build a one-row CSV template using readable survey labels."""
    defaults = default_patient(bundle)
    row = {}
    for feature in bundle["features"]:
        value = defaults[feature]
        if feature in config.BINARY_COLUMNS:
            row[feature] = "Yes" if int(value) >= 0.5 else "No"
        elif feature in ("PhysicalHealth", "MentalHealth", "SleepTime"):
            row[feature] = int(round(float(value)))
        else:
            row[feature] = value
    return pd.DataFrame([row], columns=bundle["features"])
