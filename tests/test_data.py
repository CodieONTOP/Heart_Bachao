from __future__ import annotations

import pandas as pd
import pytest

from heartrisk import config
from heartrisk.data import load_clean, split_data


def test_cleaned_dataset_schema() -> None:
    if not config.CLEAN_PARQUET.exists():
        pytest.skip("Run scripts/download_data.py to create the cleaned dataset")
    X, y = load_clean()
    assert len(X) == len(y) > 300_000
    assert list(X.columns) == config.ALL_FEATURES
    assert set(y.unique()) <= {0, 1}
    assert int(y.sum()) == 27_373
    assert X.isna().sum().sum() == 0
    assert str(X["AgeCategory"].dtype) == "string"
    assert str(X["GenHealth"].dtype) == "string"
    assert pd.api.types.is_integer_dtype(X["Smoking"])
    assert pd.api.types.is_numeric_dtype(X["BMI"])


def test_stratified_split_sizes_and_prevalence() -> None:
    if not config.CLEAN_PARQUET.exists():
        pytest.skip("Run scripts/download_data.py to create the cleaned dataset")
    X, y = load_clean()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    total = len(y)
    assert len(y_train) + len(y_val) + len(y_test) == total
    assert abs(len(y_test) / total - config.TEST_FRACTION) < 0.002
    assert abs(len(y_val) / total - config.VAL_FRACTION) < 0.002
    assert abs(y_test.mean() - y.mean()) < 0.002
    assert X_train.index.intersection(X_test.index).empty
