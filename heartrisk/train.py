"""Train, tune, calibrate, and evaluate the early/full risk models."""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from heartrisk import config
from heartrisk.data import (
    clean_data,
    dataset_summary,
    load_raw,
    md5sum,
    save_clean,
    split_data,
)
from heartrisk.evaluate import (
    bootstrap_metrics,
    choose_threshold,
    compute_metrics,
    expected_calibration_error,
    metrics_row,
    paired_bootstrap_delta,
    plot_calibration,
    plot_confusion,
    plot_model_comparison,
    plot_roc_pr,
    plot_subgroups,
    plot_threshold_curve,
    save_json,
    subgroup_metrics,
)
from heartrisk.explain import save_global_explanations
from heartrisk.features import (
    build_preprocessor,
    select_features,
    transformed_feature_names,
)


def _fit_logistic(X_train, y_train, features, seed=config.SEED):
    preprocessor = build_preprocessor(features, scale_numeric=True)
    transformed = preprocessor.fit_transform(select_features(X_train, features))
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    start = time.perf_counter()
    model.fit(transformed, y_train)
    return preprocessor, model, time.perf_counter() - start


def _fit_random_forest(X_train, y_train, features, seed=config.SEED):
    preprocessor = build_preprocessor(features, scale_numeric=False)
    transformed = preprocessor.fit_transform(select_features(X_train, features))
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=25,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    start = time.perf_counter()
    model.fit(transformed, y_train)
    return preprocessor, model, time.perf_counter() - start


def hgb_configs(quick: bool = False) -> list[dict]:
    base = {
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "class_weight": "balanced",
        "random_state": config.SEED,
    }
    if quick:
        return [
            {
                **base,
                "learning_rate": 0.08,
                "max_iter": 120,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 50,
                "l2_regularization": 1.0,
            }
        ]
    return [
        {
            **base,
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        },
        {
            **base,
            "learning_rate": 0.05,
            "max_iter": 400,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
            "l2_regularization": 0.0,
        },
        {
            **base,
            "learning_rate": 0.03,
            "max_iter": 350,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 100,
            "l2_regularization": 5.0,
        },
    ]


def _fit_hgb(X_train, y_train, X_val, y_val, features, quick=False):
    preprocessor = build_preprocessor(features, scale_numeric=False)
    train_t = preprocessor.fit_transform(select_features(X_train, features))
    val_t = preprocessor.transform(select_features(X_val, features))

    best = None
    trials = []
    for index, params in enumerate(hgb_configs(quick=quick), start=1):
        model = HistGradientBoostingClassifier(**params)
        start = time.perf_counter()
        model.fit(train_t, y_train)
        elapsed = time.perf_counter() - start
        val_prob = model.predict_proba(val_t)[:, 1]
        score = float(average_precision_score(y_val, val_prob))
        trials.append(
            {
                "trial": index,
                "params": params,
                "val_pr_auc": score,
                "val_roc_auc": float(roc_auc_score(y_val, val_prob)),
                "fit_seconds": elapsed,
            }
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "model": model,
                "params": params,
                "val_prob": val_prob,
                "trial": index,
            }

    return preprocessor, best["model"], best["params"], best["val_prob"], trials


def _defaults(X_train: pd.DataFrame, features: list[str]) -> dict:
    result: dict = {}
    age_map = {label: index for index, label in enumerate(config.AGE_CATEGORIES)}
    health_map = {label: index for index, label in enumerate(config.GENHEALTH_CATEGORIES)}

    for column in features:
        if column in config.NUMERIC_COLUMNS:
            result[column] = float(X_train[column].median())
        elif column == "AgeCategory":
            index = int(X_train[column].map(age_map).median())
            result[column] = config.AGE_CATEGORIES[index]
        elif column == "GenHealth":
            index = int(X_train[column].map(health_map).median())
            result[column] = config.GENHEALTH_CATEGORIES[index]
        elif column in config.BINARY_COLUMNS:
            result[column] = int(X_train[column].mode().iloc[0])
        else:
            result[column] = str(X_train[column].mode().iloc[0])
    return result


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _example_patients(bundle: dict, X_test: pd.DataFrame, probabilities: np.ndarray) -> dict:
    order = np.argsort(probabilities)
    low_index = int(order[max(0, int(0.02 * len(order)))])
    high_index = int(order[min(len(order) - 1, int(0.98 * len(order)))])
    examples = {}
    for name, index in (("low_risk", low_index), ("high_risk", high_index)):
        row = X_test.iloc[index]
        examples[name] = {
            feature: _python_scalar(row[feature]) for feature in bundle["features"]
        }
    return examples


def _split_summary(y_train, y_val, y_test) -> dict:
    return {
        "train": {"n": int(len(y_train)), "prevalence": float(y_train.mean())},
        "validation": {"n": int(len(y_val)), "prevalence": float(y_val.mean())},
        "test": {"n": int(len(y_test)), "prevalence": float(y_test.mean())},
    }


def _save_bundle(bundle: dict, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{bundle['feature_set']}_bundle.joblib"
    joblib.dump(bundle, path, compress=3)
    return path


def train_and_evaluate(
    models_dir: Path = config.MODELS_DIR,
    figures_dir: Path = config.FIGURES_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    quick: bool = False,
    seed: int = config.SEED,
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and validating data...")
    raw = load_raw()
    data_summary = dataset_summary(raw)
    X, y = clean_data(raw)
    save_clean(X, y)
    print(
        f"  {data_summary['rows']:,} rows, {data_summary['positives']:,} positives, "
        f"prevalence {data_summary['prevalence']:.2%}"
    )

    if quick:
        sample_n = min(30_000, len(X))
        X = X.sample(n=sample_n, random_state=seed)
        y = y.loc[X.index]
        print(f"  Quick mode: using {sample_n:,} rows")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed=seed)

    comparison_rows: list[dict] = []
    test_predictions: dict[str, np.ndarray] = {}
    bundles: dict[str, dict] = {}
    metrics_payload: dict = {
        "dataset": data_summary,
        "split": _split_summary(y_train, y_val, y_test),
        "feature_sets": {k: len(v) for k, v in config.FEATURE_SETS.items()},
        "models": {},
    }

    for feature_set, features in config.FEATURE_SETS.items():
        print(f"\n=== {feature_set.upper()} feature set ({len(features)} predictors) ===")
        selected_models: dict[str, dict] = {}

        print("  Training logistic regression baseline...")
        lr_pre, lr_model, lr_time = _fit_logistic(X_train, y_train, features, seed)
        lr_val_prob = lr_model.predict_proba(lr_pre.transform(select_features(X_val, features)))[:, 1]
        comparison_rows.append(
            metrics_row(feature_set, "logistic", compute_metrics(y_val, lr_val_prob, 0.5))
        )
        selected_models["logistic"] = {"fit_seconds": lr_time, "params": {"class_weight": "balanced"}}

        print("  Training random forest benchmark...")
        rf_pre, rf_model, rf_time = _fit_random_forest(X_train, y_train, features, seed)
        rf_val_prob = rf_model.predict_proba(rf_pre.transform(select_features(X_val, features)))[:, 1]
        comparison_rows.append(
            metrics_row(feature_set, "random_forest", compute_metrics(y_val, rf_val_prob, 0.5))
        )
        selected_models["random_forest"] = {
            "fit_seconds": rf_time,
            "params": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 25},
        }

        print("  Training histogram gradient boosting (primary)...")
        hgb_pre, hgb_model, hgb_params, hgb_val_prob, trials = _fit_hgb(
            X_train, y_train, X_val, y_val, features, quick=quick
        )
        comparison_rows.append(
            metrics_row(feature_set, "hist_gradient_boosting", compute_metrics(y_val, hgb_val_prob, 0.5))
        )
        selected_models["hist_gradient_boosting"] = {
            "params": hgb_params,
            "trials": trials,
        }

        print("  Calibrating and selecting screening threshold...")
        val_t = hgb_pre.transform(select_features(X_val, features))
        test_t = hgb_pre.transform(select_features(X_test, features))

        calibrated = CalibratedClassifierCV(FrozenEstimator(hgb_model), method="isotonic")
        calibrated.fit(val_t, y_val)
        val_prob = calibrated.predict_proba(val_t)[:, 1]
        threshold_info = choose_threshold(y_val, val_prob, target_sensitivity=0.80)
        threshold_info["threshold"] = float(np.clip(threshold_info["threshold"], 0.01, 0.99))
        threshold_info["threshold_sensitivity"] = float(
            np.clip(threshold_info["threshold_sensitivity"], 0.01, 0.99)
        )
        threshold = threshold_info["threshold"]

        test_prob = calibrated.predict_proba(test_t)[:, 1]
        test_prob_raw = hgb_model.predict_proba(test_t)[:, 1]
        test_predictions[feature_set] = test_prob

        test_metrics = compute_metrics(y_test, test_prob, threshold)
        test_metrics["ece"] = expected_calibration_error(y_test, test_prob)
        test_metrics["ci"] = bootstrap_metrics(y_test, test_prob, threshold, seed=seed)
        raw_metrics = compute_metrics(y_test, test_prob_raw, 0.5)
        raw_metrics["ece"] = expected_calibration_error(y_test, test_prob_raw)

        print(
            f"  Test PR-AUC {test_metrics['pr_auc']:.3f} | ROC-AUC {test_metrics['roc_auc']:.3f} "
            f"| sensitivity {test_metrics['sensitivity']:.3f} | specificity "
            f"{test_metrics['specificity']:.3f}"
        )

        feature_names = transformed_feature_names(hgb_pre)
        bundle = {
            "feature_set": feature_set,
            "features": features,
            "preprocessor": hgb_pre,
            "base_model": hgb_model,
            "calibrated_model": calibrated,
            "threshold": threshold,
            "threshold_info": threshold_info,
            "feature_names": feature_names,
            "metrics": test_metrics,
            "raw_metrics_at_0_5": raw_metrics,
            "defaults": _defaults(X_train, features),
            "examples": {},
            "calibration_method": "isotonic",
            "split_sizes": {
                "train": int(len(y_train)),
                "validation": int(len(y_val)),
                "test": int(len(y_test)),
            },
        }
        bundle["examples"] = _example_patients(bundle, X_test, test_prob)
        bundle_path = _save_bundle(bundle, models_dir)
        print(f"  Saved {bundle_path}")
        bundles[feature_set] = bundle

        subgroup_payload = {}
        for group_name, column in (("sex", "Sex"), ("race", "Race"), ("age", "AgeCategory")):
            rows = subgroup_metrics(X_test[column], y_test, test_prob, threshold)
            subgroup_payload[group_name] = rows
            if group_name in ("sex", "race"):
                plot_subgroups(rows, column, figures_dir / f"subgroups_{group_name}_{feature_set}.png")

        plot_roc_pr(
            y_test,
            test_prob,
            figures_dir / f"roc_pr_{feature_set}.png",
            f"{feature_set.title()} model — test set",
        )
        ece = plot_calibration(
            y_test,
            test_prob,
            figures_dir / f"calibration_{feature_set}.png",
            f"Calibration — {feature_set} model",
        )
        plot_confusion(
            y_test,
            test_prob,
            threshold,
            figures_dir / f"confusion_{feature_set}.png",
            f"Confusion matrix — {feature_set} @ {threshold:.3f}",
        )
        plot_threshold_curve(
            y_test,
            test_prob,
            threshold,
            figures_dir / f"threshold_{feature_set}.png",
            f"Threshold trade-off — {feature_set}",
        )

        print("  Computing global SHAP explanations...")
        shap_summary = save_global_explanations(
            bundle,
            X_test,
            feature_set,
            out_dir=figures_dir,
            sample_n=2000,
            seed=seed,
        )

        metrics_payload["models"][feature_set] = {
            "features": features,
            "selected_model": "hist_gradient_boosting",
            "selected_params": hgb_params,
            "trials": trials,
            "threshold": threshold_info,
            "calibration": "isotonic (validation set)",
            "test": test_metrics,
            "test_raw_at_0_5": raw_metrics,
            "calibration_ece": ece,
            "subgroups": subgroup_payload,
            "shap": shap_summary,
            "candidates": selected_models,
        }

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    plot_model_comparison(comparison, figures_dir / "model_comparison.png")
    metrics_payload["model_comparison"] = comparison.to_dict(orient="records")

    delta_roc = paired_bootstrap_delta(
        y_test,
        test_predictions["early"],
        test_predictions["full"],
        metric="roc_auc",
        seed=seed,
    )
    delta_pr = paired_bootstrap_delta(
        y_test,
        test_predictions["early"],
        test_predictions["full"],
        metric="pr_auc",
        seed=seed,
    )
    metrics_payload["early_vs_full"] = {"roc_auc": delta_roc, "pr_auc": delta_pr}

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset": {
            **data_summary,
            "source": config.DATA_URL,
            "checksum_md5": md5sum(config.RAW_CSV),
            "target": config.TARGET,
        },
        "split": _split_summary(y_train, y_val, y_test),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "feature_sets": {k: len(v) for k, v in config.FEATURE_SETS.items()},
        "models": {
            feature_set: {
                "threshold": bundles[feature_set]["threshold"],
                "selected_params": metrics_payload["models"][feature_set]["selected_params"],
                "test_pr_auc": metrics_payload["models"][feature_set]["test"]["pr_auc"],
                "test_roc_auc": metrics_payload["models"][feature_set]["test"]["roc_auc"],
                "test_sensitivity": metrics_payload["models"][feature_set]["test"]["sensitivity"],
                "test_specificity": metrics_payload["models"][feature_set]["test"]["specificity"],
            }
            for feature_set in bundles
        },
        "early_vs_full": metrics_payload["early_vs_full"],
        "shap_importance": {
            feature_set: metrics_payload["models"][feature_set]["shap"]["importance"][:10]
            for feature_set in bundles
        },
    }

    save_json(metrics_payload, reports_dir / "metrics.json")
    save_json(metadata, models_dir / "metadata.json")
    print(f"\nSaved metrics to {reports_dir / 'metrics.json'}")
    print(f"Saved metadata to {models_dir / 'metadata.json'}")
    return metrics_payload
