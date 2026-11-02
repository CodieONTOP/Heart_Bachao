from __future__ import annotations

from heartrisk import config
from heartrisk.predict import (
    default_patient,
    example_patients,
    predict_patient,
    risk_band,
)


def test_bundles_have_expected_shape(early_bundle: dict, full_bundle: dict) -> None:
    for feature_set, bundle in (("early", early_bundle), ("full", full_bundle)):
        assert bundle["feature_set"] == feature_set
        assert bundle["features"] == config.FEATURE_SETS[feature_set]
        assert 0 < bundle["threshold"] < 1
        assert bundle["feature_names"]
        assert set(default_patient(bundle)) == set(bundle["features"])


def test_examples_score_in_expected_order(early_bundle: dict) -> None:
    examples = example_patients(early_bundle)
    low = predict_patient(early_bundle, examples["low_risk"])
    high = predict_patient(early_bundle, examples["high_risk"])
    assert 0 <= low["probability"] <= 1
    assert 0 <= high["probability"] <= 1
    assert high["probability"] > low["probability"]
    assert high["risk_band"] == risk_band(high["probability"], high["threshold"])
    assert low["risk_band"] == "low"


def test_full_model_scores_full_features(full_bundle: dict) -> None:
    patient = full_bundle["examples"]["high_risk"]
    result = predict_patient(full_bundle, patient)
    assert result["feature_set"] == "full"
    assert result["features"] == config.FULL_FEATURES
    assert 0 <= result["probability"] <= 1
    assert 0 <= result["raw_probability"] <= 1
