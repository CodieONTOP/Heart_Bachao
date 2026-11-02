"""Data acquisition, validation, cleaning, and splitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heartrisk import config


class DataValidationError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(
    dest: Path = config.RAW_CSV,
    url: str = config.DATA_URL,
    expected_md5: str = config.DATA_MD5,
    expected_size: int = config.DATA_SIZE_BYTES,
    force: bool = False,
) -> Path:
    """Download the BRFSS 2020 extract and verify its checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if dest.stat().st_size == expected_size and md5sum(dest) == expected_md5:
            print(f"Dataset already present and verified: {dest}")
            return dest
        print("Existing file failed verification; downloading a fresh copy.")

    import requests

    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", expected_size))
        written = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {written / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct:5.1f}%)", end="")
    print()

    actual_size = partial.stat().st_size
    actual_md5 = md5sum(partial)
    if actual_size != expected_size or actual_md5 != expected_md5:
        partial.unlink(missing_ok=True)
        raise DataValidationError(
            "Downloaded file failed verification. "
            f"Expected {expected_size} bytes / {expected_md5}, "
            f"got {actual_size} bytes / {actual_md5}."
        )

    partial.replace(dest)
    print(f"Verified and saved: {dest}")
    return dest


def load_raw(path: Path = config.RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    validate_raw_columns(df)
    return df


def validate_raw_columns(df: pd.DataFrame) -> None:
    missing = [col for col in config.EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Raw dataset is missing required columns: {missing}")
    if df.empty:
        raise DataValidationError("Raw dataset is empty.")


def _map_yes_no(series: pd.Series, column: str) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    mapped = cleaned.map({"Yes": 1, "No": 0})
    unknown = cleaned[mapped.isna()].dropna().unique().tolist()
    if unknown:
        raise DataValidationError(f"Unexpected values in {column}: {unknown[:10]}")
    return mapped.astype("int64")


def _validate_categories(series: pd.Series, column: str, allowed: list[str]) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    unknown = sorted(set(cleaned.dropna().unique()) - set(allowed))
    if unknown:
        raise DataValidationError(f"Unexpected values in {column}: {unknown[:10]}")
    if cleaned.isna().any():
        raise DataValidationError(f"{column} contains missing values.")
    return cleaned


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model-ready predictors and a binary target.

    Categorical labels are retained in ``X``; encoding happens inside the
    scikit-learn preprocessing pipeline so the app can accept human-readable
    survey answers.
    """
    validate_raw_columns(df)

    target = df[config.TARGET].astype("string").str.strip()
    if not target.isin(["Yes", "No"]).all():
        raise DataValidationError(
            f"Unexpected target values: {sorted(target.unique().tolist())}"
        )
    y = target.map({"Yes": 1, "No": 0}).astype("int64")

    X = df.drop(columns=[config.TARGET]).copy()

    for column in config.BINARY_COLUMNS:
        X[column] = _map_yes_no(X[column], column)

    for column in config.NUMERIC_COLUMNS:
        X[column] = pd.to_numeric(X[column], errors="raise")
        if X[column].isna().any():
            raise DataValidationError(f"{column} contains missing or non-numeric values.")

    X["AgeCategory"] = _validate_categories(
        X["AgeCategory"], "AgeCategory", config.AGE_CATEGORIES
    )
    X["GenHealth"] = _validate_categories(
        X["GenHealth"], "GenHealth", config.GENHEALTH_CATEGORIES
    )
    X["Sex"] = _validate_categories(X["Sex"], "Sex", config.SEX_CATEGORIES)
    X["Race"] = _validate_categories(X["Race"], "Race", config.RACE_CATEGORIES)
    X["Diabetic"] = _validate_categories(
        X["Diabetic"], "Diabetic", config.DIABETIC_CATEGORIES
    )

    for column, (low, high) in config.INPUT_RANGES.items():
        out_of_range = ((X[column] < low) | (X[column] > high)).sum()
        if out_of_range:
            print(
                f"Warning: clipping {out_of_range} out-of-range values in {column} "
                f"to [{low}, {high}]."
            )
            X[column] = X[column].clip(low, high)

    X = X[config.ALL_FEATURES]
    return X, y


def save_clean(X: pd.DataFrame, y: pd.Series, path: Path = config.CLEAN_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = X.copy()
    frame[config.TARGET] = y.to_numpy()
    frame.to_parquet(path, index=False)
    return path


def load_clean(
    path: Path = config.CLEAN_PARQUET, force: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    if path.exists() and not force:
        frame = pd.read_parquet(path)
        return frame[config.ALL_FEATURES], frame[config.TARGET].astype("int64")
    raw = load_raw()
    X, y = clean_data(raw)
    save_clean(X, y, path)
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = config.TEST_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 70/15/15 train/validation/test split."""
    holdout_fraction = test_fraction + val_fraction
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=holdout_fraction,
        stratify=y,
        random_state=seed,
    )
    relative_test = test_fraction / holdout_fraction
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def dataset_summary(df: pd.DataFrame) -> dict:
    target = df[config.TARGET].astype("string").str.strip()
    positives = int((target == "Yes").sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "positives": positives,
        "prevalence": positives / len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in config.NUMERIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "min": float(np.nanmin(values)),
                "median": float(np.nanmedian(values)),
                "max": float(np.nanmax(values)),
                "missing": int(values.isna().sum()),
            }
        )
    return pd.DataFrame(rows)
