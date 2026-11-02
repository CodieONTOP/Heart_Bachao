from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from heartrisk.app_support import artifacts_ready

ROOT = Path(__file__).resolve().parents[1]

requires_artifacts = pytest.mark.skipif(
    not artifacts_ready(),
    reason="Run scripts/train_models.py to generate model artifacts.",
)


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


@requires_artifacts
def test_risk_page_loads() -> None:
    app = AppTest.from_file(ROOT / "app.py", default_timeout=120).run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Risk assessment"]
    assert any(button.label == "Estimate risk" for button in app.button)
    assert not any(metric.label == "Screening band" for metric in app.metric)


@requires_artifacts
def test_risk_example_scores() -> None:
    app = AppTest.from_file(ROOT / "app.py", default_timeout=120).run()
    _button(app, "Load higher-risk example").click().run()
    _button(app, "Estimate risk").click().run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Screening band"] == "Higher risk"
    assert metrics["Calibrated probability"].endswith("%")
    assert metrics["Decision threshold"].endswith("%")
    assert any(subheader.value == "Why this estimate?" for subheader in app.subheader)


@requires_artifacts
def test_full_feature_set_scores() -> None:
    app = AppTest.from_file(ROOT / "app.py", default_timeout=120).run()
    app.segmented_control[0].set_value("full").run()
    assert not app.exception
    _button(app, "Estimate risk").click().run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert "Calibrated probability" in metrics
    assert any(toggle.label == "Stroke history" for toggle in app.toggle)


def test_insights_and_about_pages_render() -> None:
    insights = AppTest.from_file(ROOT / "app_pages/insights.py", default_timeout=120).run()
    assert not insights.exception
    assert [title.value for title in insights.title] == ["Model insights"]

    about = AppTest.from_file(ROOT / "app_pages/about.py", default_timeout=120).run()
    assert not about.exception
    assert [title.value for title in about.title] == ["About & methods"]


@requires_artifacts
def test_batch_page_loads() -> None:
    app = AppTest.from_file(ROOT / "app_pages/batch.py", default_timeout=120).run()
    assert not app.exception
    assert [title.value for title in app.title] == ["Batch scoring"]
    assert any(uploader.label == "Upload respondent CSV" for uploader in app.file_uploader)

