from __future__ import annotations

from heartrisk import config
from heartrisk.predict import risk_band


def test_feature_sets_are_nested() -> None:
    assert len(config.EARLY_FEATURES) == 11
    assert len(config.FULL_FEATURES) == 17
    assert set(config.EARLY_FEATURES) < set(config.FULL_FEATURES)
    assert set(config.FULL_FEATURES) - set(config.EARLY_FEATURES) == {
        "Stroke",
        "Diabetic",
        "DiffWalking",
        "Asthma",
        "KidneyDisease",
        "SkinCancer",
    }


def test_expected_columns_are_unique() -> None:
    assert len(config.EXPECTED_COLUMNS) == len(set(config.EXPECTED_COLUMNS))
    assert config.TARGET not in config.ALL_FEATURES
    assert set(config.ALL_FEATURES) == set(config.FULL_FEATURES)


def test_all_categories_have_labels() -> None:
    for feature in config.ALL_FEATURES:
        assert feature in config.FEATURE_LABELS
    for feature in config.MODIFIABLE_FEATURES:
        assert feature in config.EARLY_FEATURES


def test_risk_band_boundaries() -> None:
    threshold = 0.10
    assert risk_band(0.0, threshold) == "low"
    assert risk_band(0.049, threshold) == "low"
    assert risk_band(0.05, threshold) == "moderate"
    assert risk_band(0.099, threshold) == "moderate"
    assert risk_band(0.10, threshold) == "high"
    assert risk_band(0.50, threshold) == "high"
