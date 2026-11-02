from __future__ import annotations

import json

import pytest

from heartrisk import config
from heartrisk.predict import load_bundle


@pytest.fixture(scope="session")
def early_bundle() -> dict:
    try:
        return load_bundle("early")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def full_bundle() -> dict:
    try:
        return load_bundle("full")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def metrics() -> dict:
    if not config.METRICS_JSON.exists():
        pytest.skip("Run scripts/train_models.py to generate reports/metrics.json")
    return json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))


@pytest.fixture
def high_risk_patient(early_bundle: dict) -> dict:
    return early_bundle["examples"]["high_risk"]


@pytest.fixture
def low_risk_patient(early_bundle: dict) -> dict:
    return early_bundle["examples"]["low_risk"]
