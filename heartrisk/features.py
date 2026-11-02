"""Preprocessing pipelines and feature-name helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from heartrisk import config

CONTINUOUS_FEATURES = list(config.NUMERIC_COLUMNS)
CATEGORICAL_FEATURES = list(config.ONE_HOT_COLUMNS)
ORDINAL_FEATURES = list(config.ORDINAL_COLUMNS)


def select_features(X: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_list if column not in X.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return X[feature_list].copy()


def patient_frame(patient: dict, feature_list: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_list if column not in patient]
    if missing:
        raise ValueError(f"Patient record is missing required features: {missing}")
    return pd.DataFrame(
        [{column: patient[column] for column in feature_list}],
        columns=feature_list,
    )


def build_preprocessor(feature_list: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    """Build a ColumnTransformer for a feature set.

    ``scale_numeric=True`` is used for the logistic-regression benchmark only.
    Tree models keep original units so SHAP values are directly interpretable.
    """
    one_hot = [c for c in CATEGORICAL_FEATURES if c in feature_list]
    ordinal = [c for c in ORDINAL_FEATURES if c in feature_list]
    continuous = [c for c in CONTINUOUS_FEATURES if c in feature_list]
    binary = [c for c in config.BINARY_COLUMNS if c in feature_list]

    transformers: list[tuple] = []
    if one_hot:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                one_hot,
            )
        )
    if ordinal:
        encoder = OrdinalEncoder(
            categories=[config.AGE_CATEGORIES if c == "AgeCategory" else config.GENHEALTH_CATEGORIES for c in ordinal],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )
        if scale_numeric:
            transformers.append(
                ("ord", Pipeline([("encode", encoder), ("scale", StandardScaler())]), ordinal)
            )
        else:
            transformers.append(("ord", encoder, ordinal))
    if continuous:
        transformers.append(
            ("num", StandardScaler() if scale_numeric else "passthrough", continuous)
        )
    if binary:
        transformers.append(("bin", "passthrough", binary))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(name) for name in preprocessor.get_feature_names_out()]


def pretty_feature_name(name: str) -> str:
    if name in config.FEATURE_LABELS:
        return config.FEATURE_LABELS[name]
    for prefix, label in (
        ("Sex_", "Sex"),
        ("Race_", "Race"),
        ("Diabetic_", "Diabetes"),
    ):
        if name.startswith(prefix):
            return f"{label}: {name[len(prefix):]}"
    return name


def age_label(index: int | float) -> str:
    try:
        idx = int(round(float(index)))
    except (TypeError, ValueError):
        return str(index)
    if 1 <= idx <= len(config.AGE_CATEGORIES):
        return config.AGE_CATEGORIES[idx - 1]
    return str(index)


def genhealth_label(index: int | float) -> str:
    try:
        idx = int(round(float(index)))
    except (TypeError, ValueError):
        return str(index)
    if 1 <= idx <= len(config.GENHEALTH_CATEGORIES):
        return config.GENHEALTH_CATEGORIES[idx - 1]
    return str(index)


def pretty_feature_value(feature: str, value: float | int | str) -> str:
    """Turn an encoded value into something a patient can read."""
    if feature.startswith("Sex_"):
        return feature.split("_", 1)[1]
    if feature.startswith("Race_"):
        return feature.split("_", 1)[1]
    if feature.startswith("Diabetic_"):
        return feature.split("_", 1)[1]
    if feature == "AgeCategory":
        return age_label(value)
    if feature == "GenHealth":
        return genhealth_label(value)
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def active_category(feature: str, encoded_value: float) -> str | None:
    """Return the category for a one-hot feature if it is active."""
    if any(feature.startswith(prefix) for prefix in ("Sex_", "Race_", "Diabetic_")):
        return feature.split("_", 1)[1] if float(encoded_value) >= 0.5 else None
    return None
