from __future__ import annotations

from datetime import UTC, datetime

from heartrisk import config
from heartrisk.report import build_patient_report


def _patient(**overrides) -> dict:
    patient = {
        "AgeCategory": "60-64",
        "Sex": "Female",
        "Race": "White",
        "BMI": 31.2,
        "Smoking": 1,
        "AlcoholDrinking": 0,
        "PhysicalActivity": 1,
        "SleepTime": 6,
        "PhysicalHealth": 4,
        "MentalHealth": 1,
        "GenHealth": "Fair",
    }
    patient.update(overrides)
    return patient


def _result(patient: dict) -> dict:
    return {
        "probability": 0.173,
        "raw_probability": 0.201,
        "threshold": 0.105,
        "risk_band": "high",
        "risk_band_label": "Higher risk",
        "feature_set": "early",
        "features": config.EARLY_FEATURES,
    }


def _explanation() -> dict:
    return {
        "base_value": -2.35,
        "top_increasing": [
            {"label": "Age group", "value": "60-64", "shap": 0.42},
            {"label": "Smoking history", "value": "Yes", "shap": 0.21},
        ],
        "top_decreasing": [
            {"label": "Physical activity", "value": "Yes", "shap": -0.18},
        ],
    }


def test_report_contains_core_sections() -> None:
    patient = _patient()
    report = build_patient_report(
        _result(patient),
        _explanation(),
        patient,
        generated_at=datetime(2026, 11, 2, 12, 0, tzinfo=UTC),
    )
    assert "<!doctype html>" in report
    assert "Heart risk summary" in report
    assert "17.3%" in report
    assert "Higher risk" in report
    assert "Factors increasing risk" in report
    assert "Factors decreasing risk" in report
    assert "Important limitations" in report
    assert "not a medical device" in report
    assert "2026-11-02 12:00 UTC" in report


def test_report_includes_explorations() -> None:
    patient = _patient()
    what_if = {
        "probability": 0.151,
        "delta": -0.022,
        "changes": {"Smoking": 0},
        "movers": [{"label": "Smoking history", "delta": -0.11}],
    }
    lower_profile = {
        "probability": 0.09,
        "delta": -0.083,
        "target": 0.0525,
        "steps": [
            {
                "changes": {"Smoking": 0},
                "probability": 0.151,
                "delta": -0.022,
            }
        ],
    }
    report = build_patient_report(
        _result(patient),
        _explanation(),
        patient,
        what_if=what_if,
        lower_profile=lower_profile,
    )
    assert "What-if comparison" in report
    assert "Explored lower-risk profile" in report
    assert "-2.2 pts" in report


def test_report_escapes_untrusted_values() -> None:
    patient = _patient(Race="<script>alert(1)</script>")
    report = build_patient_report(_result(patient), _explanation(), patient)
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
