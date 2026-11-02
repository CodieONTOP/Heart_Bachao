"""SHAP explanations and model-based what-if analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from heartrisk import config
from heartrisk.features import age_label, genhealth_label, pretty_feature_name
from heartrisk.predict import risk_band

ONE_HOT_PREFIXES = ("Sex_", "Race_", "Diabetic_")


def _group_name(feature_name: str) -> str:
    return next(
        (prefix[:-1] for prefix in ONE_HOT_PREFIXES if feature_name.startswith(prefix)),
        feature_name,
    )


def get_explainer(base_model) -> shap.TreeExplainer:
    return shap.TreeExplainer(base_model)


def shap_values_for(
    explainer: shap.TreeExplainer, X_transformed: np.ndarray
) -> tuple[np.ndarray, float]:
    values = explainer.shap_values(X_transformed)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 1]

    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected_array = np.asarray(expected, dtype=float).ravel()
        expected = float(expected_array[1] if expected_array.size > 1 else expected_array[0])
    else:
        expected = float(expected)
    return values, expected


def _predict(
    bundle: dict, patient: dict
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    frame = pd.DataFrame([patient], columns=bundle["features"])
    transformed = bundle["preprocessor"].transform(frame)
    calibrated = float(bundle["calibrated_model"].predict_proba(transformed)[0, 1])
    raw = float(bundle["base_model"].predict_proba(transformed)[0, 1])
    explainer = get_explainer(bundle["base_model"])
    values, expected = shap_values_for(explainer, transformed)
    return calibrated, raw, values[0], np.asarray(transformed[0]), expected


def local_explanation(bundle: dict, patient: dict, top_k: int = 10) -> dict:
    probability, raw_probability, values, transformed, expected = _predict(bundle, patient)
    feature_names = bundle["feature_names"]
    threshold = float(bundle["threshold"])
    band = risk_band(probability, threshold)

    groups: dict[str, dict] = {}
    for name, _value, shap_value in zip(feature_names, transformed, values, strict=False):
        group_name = _group_name(name)
        if group_name not in groups:
            groups[group_name] = {
                "feature": group_name,
                "label": pretty_feature_name(group_name),
                "value": patient.get(group_name, ""),
                "shap": 0.0,
            }
        groups[group_name]["shap"] += float(shap_value)

    contributions = []
    for group_name, item in groups.items():
        raw_value = patient.get(group_name, item["value"])
        if group_name == "AgeCategory":
            value_label = age_label(_ordinal_index(raw_value, config.AGE_CATEGORIES))
        elif group_name == "GenHealth":
            value_label = genhealth_label(_ordinal_index(raw_value, config.GENHEALTH_CATEGORIES))
        elif group_name in ("Smoking", "AlcoholDrinking", "PhysicalActivity", "Stroke", "DiffWalking", "Asthma", "KidneyDisease", "SkinCancer"):
            value_label = "Yes" if float(raw_value) >= 0.5 else "No"
        elif group_name in config.NUMERIC_COLUMNS:
            value_label = f"{float(raw_value):.1f}"
        else:
            value_label = str(raw_value)

        contributions.append(
            {
                "feature": group_name,
                "label": item["label"],
                "value": value_label,
                "shap": float(item["shap"]),
                "direction": "increases" if item["shap"] >= 0 else "decreases",
            }
        )

    contributions.sort(key=lambda item: abs(item["shap"]), reverse=True)
    increasing = [c for c in contributions if c["shap"] > 0][:top_k]
    decreasing = [c for c in contributions if c["shap"] < 0][:top_k]

    return {
        "probability": probability,
        "raw_probability": raw_probability,
        "threshold": threshold,
        "risk_band": band,
        "base_value": float(expected),
        "contributions": contributions,
        "top_increasing": increasing,
        "top_decreasing": decreasing,
        "feature_names": feature_names,
        "shap_values": values,
        "transformed": transformed,
    }


def _ordinal_index(value, categories: list[str]) -> int:
    if isinstance(value, str):
        try:
            return categories.index(value) + 1
        except ValueError:
            return -1
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return -1


def what_if(bundle: dict, patient: dict, changes: dict) -> dict:
    updated = patient.copy()
    for key, value in changes.items():
        if key not in bundle["features"]:
            continue
        updated[key] = value

    updated = _clip_patient(updated)
    base_probability, _, base_values, _, _ = _predict(bundle, patient)
    new_probability, _, new_values, _, _ = _predict(bundle, updated)
    feature_names = bundle["feature_names"]

    deltas = []
    for name, before, after in zip(feature_names, base_values, new_values, strict=False):
        deltas.append((_group_name(name), float(after - before)))
    grouped: dict[str, float] = {}
    for group, delta in deltas:
        grouped[group] = grouped.get(group, 0.0) + delta

    movers = [
        {
            "feature": group,
            "label": pretty_feature_name(group),
            "delta": float(delta),
        }
        for group, delta in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
    ][:8]

    return {
        "probability": new_probability,
        "delta": new_probability - base_probability,
        "changes": changes,
        "movers": movers,
    }


def _clip_patient(patient: dict) -> dict:
    updated = patient.copy()
    for column, (low, high) in config.INPUT_RANGES.items():
        if column in updated:
            updated[column] = float(np.clip(float(updated[column]), low, high))
    return updated


def find_lower_risk_profile(
    bundle: dict,
    patient: dict,
    target_probability: float | None = None,
    max_steps: int = 5,
) -> dict:
    """Greedy search over realistic, model-based changes.

    This is a what-if analysis, not a causal recommendation.
    """
    current = patient.copy()
    current_probability = _predict(bundle, current)[0]
    target = target_probability if target_probability is not None else bundle["threshold"] * 0.5
    steps = []

    for _ in range(max_steps):
        if current_probability <= target:
            break
        candidates = list(_candidate_changes(current))
        best = None
        for changes in candidates:
            trial = current.copy()
            trial.update(changes)
            trial = _clip_patient(trial)
            trial_probability = _predict(bundle, trial)[0]
            improvement = current_probability - trial_probability
            if improvement > 1e-6 and (best is None or improvement > best[0]):
                best = (improvement, changes, trial, trial_probability)
        if best is None:
            break
        improvement, changes, current, current_probability = best
        steps.append(
            {
                "changes": changes,
                "probability": current_probability,
                "delta": -improvement,
            }
        )

    return {
        "steps": steps,
        "probability": current_probability,
        "delta": current_probability - _predict(bundle, patient)[0],
        "target": target,
        "patient": current,
    }


def _candidate_changes(patient: dict):
    if float(patient.get("Smoking", 0)) >= 0.5:
        yield {"Smoking": 0}
    if float(patient.get("AlcoholDrinking", 0)) >= 0.5:
        yield {"AlcoholDrinking": 0}
    if float(patient.get("PhysicalActivity", 1)) < 0.5:
        yield {"PhysicalActivity": 1}

    bmi = float(patient.get("BMI", 25))
    for reduction in (1, 2, 3, 5):
        if bmi - reduction >= 18.5:
            yield {"BMI": round(bmi - reduction, 1)}

    sleep = float(patient.get("SleepTime", 7))
    if sleep < 7:
        yield {"SleepTime": 7}
    elif sleep > 9:
        yield {"SleepTime": 9}

    for column in ("PhysicalHealth", "MentalHealth"):
        days = float(patient.get(column, 0))
        for reduction in (5, 10):
            if days - reduction >= 0:
                yield {column: int(days - reduction)}

    genhealth = patient.get("GenHealth")
    if genhealth in config.GENHEALTH_CATEGORIES:
        idx = config.GENHEALTH_CATEGORIES.index(genhealth)
        if idx > 0:
            yield {"GenHealth": config.GENHEALTH_CATEGORIES[idx - 1]}


def global_shap(
    bundle: dict,
    X_sample: pd.DataFrame,
    seed: int = config.SEED,
) -> dict:
    frame = X_sample[bundle["features"]].reset_index(drop=True)
    transformed = bundle["preprocessor"].transform(frame)
    explainer = get_explainer(bundle["base_model"])
    values, expected = shap_values_for(explainer, transformed)
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    feature_names = bundle["feature_names"]

    return {
        "values": values,
        "expected_value": expected,
        "transformed": np.asarray(transformed),
        "feature_names": feature_names,
        "importance": [
            {
                "feature": feature_names[i],
                "label": pretty_feature_name(feature_names[i]),
                "mean_abs_shap": float(importance[i]),
            }
            for i in order
        ],
        "frame": frame,
    }


def save_global_explanations(
    bundle: dict,
    X_test: pd.DataFrame,
    feature_set: str,
    out_dir: Path = config.FIGURES_DIR,
    sample_n: int = 2000,
    seed: int = config.SEED,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_n = min(sample_n, len(X_test))
    sample = X_test.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    result = global_shap(bundle, sample)
    values = result["values"]
    transformed = result["transformed"]
    feature_names = result["feature_names"]

    plt.figure()
    shap.summary_plot(
        values,
        transformed,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP summary — {feature_set} model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_beeswarm_{feature_set}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.summary_plot(
        values,
        transformed,
        feature_names=feature_names,
        max_display=12,
        show=False,
        plot_type="bar",
    )
    plt.title(f"Mean absolute SHAP — {feature_set} model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_bar_{feature_set}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    importance_frame = pd.DataFrame(result["importance"])
    importance_frame.to_csv(out_dir / f"shap_importance_{feature_set}.csv", index=False)

    interesting = [
        item["feature"]
        for item in result["importance"]
        if not item["feature"].startswith(ONE_HOT_PREFIXES)
    ][:6]
    for feature in interesting:
        plt.figure()
        shap.dependence_plot(
            feature,
            values,
            transformed,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f"SHAP dependence — {pretty_feature_name(feature)}", fontsize=11)
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace(" ", "_")
        plt.savefig(
            out_dir / f"shap_dependence_{feature_set}_{safe_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")

    probabilities = bundle["calibrated_model"].predict_proba(
        bundle["preprocessor"].transform(X_test[bundle["features"]])
    )[:, 1]
    low_idx = int(np.argmin(probabilities))
    high_idx = int(np.argmax(probabilities))
    local_indices = {"low": low_idx, "high": high_idx}
    for label, idx in local_indices.items():
        _save_waterfall(
            bundle,
            X_test.iloc[[idx]][bundle["features"]],
            out_dir / f"shap_waterfall_{feature_set}_{label}.png",
        )

    return {
        "importance": result["importance"][:20],
        "local_example_indices": local_indices,
        "sample_size": int(sample_n),
        "expected_value": result["expected_value"],
    }


def _save_waterfall(bundle: dict, row: pd.DataFrame, path: Path) -> None:
    transformed = bundle["preprocessor"].transform(row)
    explainer = get_explainer(bundle["base_model"])
    values, expected = shap_values_for(explainer, transformed)
    explanation = shap.Explanation(
        values=values[0],
        base_values=expected,
        data=np.asarray(transformed[0]),
        feature_names=bundle["feature_names"],
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def waterfall_figure(bundle: dict, patient: dict):
    """Return a matplotlib figure for Streamlit."""
    _, _, values, transformed, expected = _predict(bundle, patient)
    explanation = shap.Explanation(
        values=values,
        base_values=expected,
        data=transformed,
        feature_names=bundle["feature_names"],
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.tight_layout()
    return plt.gcf()


def grouped_shap_matrix(bundle: dict, frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return grouped SHAP contributions as ``(group_names, rows_by_groups)``."""
    rows = frame[bundle["features"]].reset_index(drop=True)
    transformed = bundle["preprocessor"].transform(rows)
    explainer = get_explainer(bundle["base_model"])
    values, _ = shap_values_for(explainer, transformed)

    column_groups: dict[str, list[int]] = {}
    for index, name in enumerate(bundle["feature_names"]):
        column_groups.setdefault(_group_name(name), []).append(index)

    group_names = list(column_groups)
    matrix = np.column_stack(
        [values[:, indices].sum(axis=1) for indices in column_groups.values()]
    )
    return group_names, matrix


def batch_top_drivers(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the largest grouped SHAP driver for every row in a batch."""
    group_names, matrix = grouped_shap_matrix(bundle, frame)
    if not len(matrix):
        return pd.DataFrame(
            columns=["top_driver", "top_driver_effect", "top_driver_shap"]
        )

    top_index = np.abs(matrix).argmax(axis=1)
    top_values = matrix[np.arange(len(matrix)), top_index]
    return pd.DataFrame(
        {
            "top_driver": [pretty_feature_name(group_names[index]) for index in top_index],
            "top_driver_effect": [
                "increases" if value >= 0 else "decreases" for value in top_values
            ],
            "top_driver_shap": np.round(top_values, 4),
        }
    )
