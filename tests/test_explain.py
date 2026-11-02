from __future__ import annotations

import numpy as np

from heartrisk.explain import (
    find_lower_risk_profile,
    get_explainer,
    local_explanation,
    shap_values_for,
    waterfall_figure,
    what_if,
)
from heartrisk.features import patient_frame
from heartrisk.predict import predict_patient


def test_shap_values_are_well_formed(early_bundle: dict) -> None:
    patient = early_bundle["examples"]["high_risk"]
    frame = patient_frame(patient, early_bundle["features"])
    transformed = early_bundle["preprocessor"].transform(frame)
    explainer = get_explainer(early_bundle["base_model"])
    values, expected = shap_values_for(explainer, transformed)
    assert values.shape == (1, len(early_bundle["feature_names"]))
    assert np.isfinite(values).all()
    assert isinstance(expected, float)
    assert np.isfinite(expected)


def test_local_explanation_groups_and_sorts(early_bundle: dict) -> None:
    patient = early_bundle["examples"]["high_risk"]
    explanation = local_explanation(early_bundle, patient)
    assert np.isfinite(explanation["probability"])
    assert explanation["feature_names"] == early_bundle["feature_names"]
    assert len(explanation["contributions"]) == 11
    features = [item["feature"] for item in explanation["contributions"]]
    assert len(features) == len(set(features))
    assert all(item["label"] for item in explanation["contributions"])
    assert all(item["value"] for item in explanation["contributions"])
    abs_values = [abs(item["shap"]) for item in explanation["contributions"]]
    assert abs_values == sorted(abs_values, reverse=True)
    assert {item["direction"] for item in explanation["contributions"]} <= {"increases", "decreases"}


def test_what_if_produces_movers(early_bundle: dict) -> None:
    patient = early_bundle["examples"]["high_risk"]
    comparison = what_if(early_bundle, patient, {"Smoking": 0})
    assert 0 <= comparison["probability"] <= 1
    assert "delta" in comparison
    assert isinstance(comparison["movers"], list)
    if comparison["movers"]:
        assert "feature" in comparison["movers"][0]
        assert "label" in comparison["movers"][0]
        assert "delta" in comparison["movers"][0]


def test_greedy_lower_risk_finds_improvement(early_bundle: dict) -> None:
    patient = early_bundle["examples"]["high_risk"].copy()
    base = predict_patient(early_bundle, patient)
    result = find_lower_risk_profile(early_bundle, patient, max_steps=3)
    assert 0 <= result["probability"] <= 1
    assert result["probability"] <= base["probability"] + 1e-9
    assert result["delta"] <= 0.0
    assert isinstance(result["patient"], dict)
    assert isinstance(result["steps"], list)
    if result["steps"]:
        assert "changes" in result["steps"][0]
        assert 0 <= result["steps"][0]["probability"] <= 1


def test_waterfall_figure_returns_figure(early_bundle: dict) -> None:
    import matplotlib.figure

    patient = early_bundle["examples"]["high_risk"]
    figure = waterfall_figure(early_bundle, patient)
    assert isinstance(figure, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt

    plt.close(figure)
