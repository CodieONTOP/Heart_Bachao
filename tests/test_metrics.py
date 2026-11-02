from __future__ import annotations


def test_metrics_cover_both_feature_sets(metrics: dict) -> None:
    assert set(metrics["models"]) == {"early", "full"}
    assert metrics["dataset"]["rows"] == 319_795
    assert metrics["dataset"]["positives"] == 27_373
    assert metrics["split"]["test"]["n"] == 47_970


def test_test_metrics_are_bounded(metrics: dict) -> None:
    for feature_set in ("early", "full"):
        test = metrics["models"][feature_set]["test"]
        assert 0.5 < test["roc_auc"] < 1
        assert 0 < test["pr_auc"] < 1
        assert 0 <= test["sensitivity"] <= 1
        assert 0 <= test["specificity"] <= 1
        assert 0 <= test["ece"] < 0.1
        for metric in (
            "roc_auc",
            "pr_auc",
            "brier",
            "sensitivity",
            "specificity",
        ):
            interval = test["ci"][metric]
            assert interval["lower"] <= interval["upper"]


def test_full_model_improves_ranking_metrics(metrics: dict) -> None:
    for metric in ("roc_auc", "pr_auc"):
        delta = metrics["early_vs_full"][metric]
        assert delta["delta"] > 0
        assert delta["lower"] > 0
        assert delta["lower"] <= delta["delta"] <= delta["upper"]


def test_subgroups_and_shap_are_present(metrics: dict) -> None:
    for feature_set in ("early", "full"):
        model = metrics["models"][feature_set]
        assert set(model["subgroups"]) == {"sex", "race", "age"}
        assert model["subgroups"]["sex"]
        assert model["subgroups"]["race"]
        assert model["subgroups"]["age"]
        assert model["shap"]["importance"]
        assert model["shap"]["sample_size"] > 0
