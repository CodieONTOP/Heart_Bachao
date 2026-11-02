"""Load saved model bundles and score individual patients."""

from __future__ import annotations

from pathlib import Path

import joblib

from heartrisk import config
from heartrisk.features import patient_frame


def bundle_path(feature_set: str, models_dir: Path | None = None) -> Path:
    root = models_dir or config.resolve_models_dir()
    return root / f"{feature_set}_bundle.joblib"


def load_bundle(feature_set: str, models_dir: Path | None = None) -> dict:
    path = bundle_path(feature_set, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `python scripts/train_models.py` first."
        )
    return joblib.load(path)


def load_metadata(models_dir: Path | None = None) -> dict:
    import json

    root = models_dir or config.resolve_models_dir()
    path = root / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {path}. Run `python scripts/train_models.py` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def risk_band(probability: float, threshold: float) -> str:
    low_cut = threshold * 0.5
    if probability < low_cut:
        return "low"
    if probability < threshold:
        return "moderate"
    return "high"


def predict_patient(bundle: dict, patient: dict) -> dict:
    frame = patient_frame(patient, bundle["features"])
    transformed = bundle["preprocessor"].transform(frame)
    calibrated = float(bundle["calibrated_model"].predict_proba(transformed)[0, 1])
    raw = float(bundle["base_model"].predict_proba(transformed)[0, 1])
    threshold = float(bundle["threshold"])
    band = risk_band(calibrated, threshold)
    return {
        "probability": calibrated,
        "raw_probability": raw,
        "threshold": threshold,
        "risk_band": band,
        "risk_band_label": config.RISK_BAND_LABELS[band],
        "feature_set": bundle["feature_set"],
        "features": bundle["features"],
    }


def default_patient(bundle: dict) -> dict:
    defaults = bundle.get("defaults")
    if not defaults:
        raise ValueError("Bundle does not contain default patient values.")
    return {key: defaults[key] for key in bundle["features"]}


def example_patients(bundle: dict) -> dict[str, dict]:
    examples = bundle.get("examples", {})
    return {
        name: {key: values[key] for key in bundle["features"]}
        for name, values in examples.items()
    }
