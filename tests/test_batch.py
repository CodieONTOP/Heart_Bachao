from __future__ import annotations

import pandas as pd
import pytest

from heartrisk import config
from heartrisk.batch import (
    BatchValidationError,
    prepare_batch,
    score_batch,
    template_frame,
)
from heartrisk.explain import batch_top_drivers


def _row(**overrides) -> dict:
    row = {
        "AgeCategory": "50-54",
        "Sex": "Female",
        "Race": "White",
        "BMI": 27.5,
        "Smoking": "No",
        "AlcoholDrinking": "No",
        "PhysicalActivity": "Yes",
        "SleepTime": 7,
        "PhysicalHealth": 0,
        "MentalHealth": 0,
        "GenHealth": "Good",
    }
    row.update(overrides)
    return row


def test_prepare_batch_accepts_labels_and_encoded_values() -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(Smoking=1, PhysicalActivity=0, SleepTime="6", GenHealth="very good"),
        ]
    )
    prepared = prepare_batch(frame, "early")
    result = prepared["frame"]
    assert prepared["rows"] == 2
    assert list(result.columns) == config.EARLY_FEATURES
    assert result.loc[0, "Smoking"] == 0
    assert result.loc[1, "Smoking"] == 1
    assert result.loc[1, "PhysicalActivity"] == 0
    assert result.loc[1, "GenHealth"] == "Very good"
    assert not prepared["warnings"]


def test_prepare_batch_ignores_extra_columns_and_blank_rows() -> None:
    frame = pd.DataFrame([_row(HeartDisease="No", note="keep"), {}])
    prepared = prepare_batch(frame, "early")
    assert prepared["rows"] == 1
    assert "HeartDisease" not in prepared["frame"].columns
    assert "note" not in prepared["frame"].columns


def test_prepare_batch_reports_row_errors() -> None:
    frame = pd.DataFrame([_row(), _row(Smoking="maybe")])
    with pytest.raises(BatchValidationError, match="Row 3"):
        prepare_batch(frame, "early")


def test_prepare_batch_requires_all_feature_columns() -> None:
    frame = pd.DataFrame([_row()]).drop(columns=["BMI"])
    with pytest.raises(BatchValidationError, match="BMI"):
        prepare_batch(frame, "early")


def test_prepare_batch_warns_when_clipping() -> None:
    frame = pd.DataFrame([_row(BMI=250.0)])
    prepared = prepare_batch(frame, "early")
    assert prepared["frame"].loc[0, "BMI"] == 100.0
    assert any("clipped" in warning for warning in prepared["warnings"])


def test_template_uses_readable_labels(early_bundle: dict) -> None:
    template = template_frame(early_bundle)
    assert list(template.columns) == early_bundle["features"]
    for column in config.BINARY_COLUMNS:
        if column in template.columns:
            assert template.loc[0, column] in {"Yes", "No"}


def test_score_batch_and_drivers(early_bundle: dict) -> None:
    frame = pd.DataFrame(
        [
            _row(),
            _row(Smoking="Yes", BMI=41.0, PhysicalHealth=20, GenHealth="Poor"),
        ]
    )
    prepared = prepare_batch(frame, "early")
    scored = score_batch(early_bundle, prepared["frame"])
    assert len(scored) == 2
    assert {"risk_probability", "raw_probability", "risk_band"} <= set(scored.columns)
    assert scored["risk_probability"].between(0, 1).all()
    assert scored["raw_probability"].between(0, 1).all()
    assert set(scored["risk_band"]) <= set(config.RISK_BAND_LABELS.values())

    drivers = batch_top_drivers(early_bundle, prepared["frame"])
    assert len(drivers) == 2
    assert set(drivers["top_driver_effect"]) <= {"increases", "decreases"}
    assert drivers["top_driver"].notna().all()
