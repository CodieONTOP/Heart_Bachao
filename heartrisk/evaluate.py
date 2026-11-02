"""Metrics, confidence intervals, subgroup diagnostics, and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"early": "#0F766E", "full": "#7C3AED", "baseline": "#64748B"}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(tp + fn),
        "prevalence": float((tp + fn) / len(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        gap = abs(y_true[mask].mean() - y_prob[mask].mean())
        ece += gap * mask.sum() / len(y_true)
    return float(ece)


def bootstrap_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    records: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "brier": [],
        "sensitivity": [],
        "specificity": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        records["roc_auc"].append(roc_auc_score(yt, yp))
        records["pr_auc"].append(average_precision_score(yt, yp))
        records["brier"].append(brier_score_loss(yt, yp))
        pred = (yp >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        records["sensitivity"].append(_safe_div(tp, tp + fn))
        records["specificity"].append(_safe_div(tn, tn + fp))

    return {
        metric: {
            "lower": float(np.nanpercentile(values, 2.5)),
            "upper": float(np.nanpercentile(values, 97.5)),
        }
        for metric, values in records.items()
        if values
    }


def paired_bootstrap_delta(
    y_true: pd.Series | np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric: str = "roc_auc",
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """Bootstrap the difference between model B and model A (B - A)."""
    y_true = np.asarray(y_true, dtype=int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    scorer = roc_auc_score if metric == "roc_auc" else average_precision_score
    point = float(scorer(y_true, prob_b) - scorer(y_true, prob_a))
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        deltas.append(float(scorer(yt, prob_b[idx]) - scorer(yt, prob_a[idx])))
    return {
        "metric": metric,
        "delta": point,
        "lower": float(np.percentile(deltas, 2.5)),
        "upper": float(np.percentile(deltas, 97.5)),
    }


def subgroup_metrics(
    groups: pd.Series,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    frame = pd.DataFrame({"group": groups.to_numpy(), "y": y_true, "p": y_prob})
    rows: list[dict] = []
    for group, part in frame.groupby("group", dropna=False):
        yt = part["y"].to_numpy()
        yp = part["p"].to_numpy()
        n = len(part)
        positives = int(yt.sum())
        row = {
            "group": str(group),
            "n": int(n),
            "positives": positives,
            "prevalence": float(positives / n),
            "roc_auc": float(roc_auc_score(yt, yp)) if positives and positives < n else None,
            "pr_auc": float(average_precision_score(yt, yp)) if positives else None,
        }
        pred = (yp >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        row["sensitivity"] = _safe_div(tp, tp + fn)
        row["specificity"] = _safe_div(tn, tn + fp)
        row["precision"] = _safe_div(tp, tp + fp)
        rows.append(row)
    return sorted(rows, key=lambda item: item["n"], reverse=True)


def choose_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float = 0.80,
) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f2[:-1]))
    threshold_f2 = float(thresholds[best_idx])

    candidates = np.where(recall[:-1] >= target_sensitivity)[0]
    if len(candidates):
        threshold_sens = float(thresholds[candidates].max())
    else:
        threshold_sens = threshold_f2

    return {
        "threshold": threshold_f2,
        "threshold_f2": threshold_f2,
        "threshold_sensitivity": threshold_sens,
        "target_sensitivity": target_sensitivity,
        "validation_f2": float(f2[best_idx]),
        "validation_recall_at_f2": float(recall[best_idx]),
        "validation_precision_at_f2": float(precision[best_idx]),
    }


def plot_roc_pr(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    axes[0].plot(fpr, tpr, color="#0F766E", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#94A3B8", ls="--", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")

    baseline = float(np.mean(y_true))
    axes[1].plot(recall, precision, color="#7C3AED", lw=2, label=f"AP = {pr_auc:.3f}")
    axes[1].axhline(baseline, color="#94A3B8", ls="--", lw=1, label=f"Prevalence = {baseline:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend(loc="upper right")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    path: Path,
    title: str,
) -> float:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    ece = expected_calibration_error(y_true, y_prob)
    ax.plot([0, 1], [0, 1], color="#94A3B8", ls="--", lw=1, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color="#0F766E", lw=2, label=f"Model (ECE={ece:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title=title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ece


def plot_confusion(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    path: Path,
    title: str,
) -> None:
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["No disease", "Disease"],
        yticklabels=["No disease", "Disease"],
        ax=ax,
    )
    ax.set(xlabel="Predicted", ylabel="Observed", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curve(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    chosen_threshold: float,
    path: Path,
    title: str,
) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(thresholds, precision[:-1], label="Precision", color="#7C3AED", lw=2)
    ax.plot(thresholds, recall[:-1], label="Recall / sensitivity", color="#0F766E", lw=2)
    ax.axvline(chosen_threshold, color="#F59E0B", ls="--", lw=1.5, label="Chosen threshold")
    ax.set(xlabel="Threshold", ylabel="Score", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, metric, title in zip(
        axes,
        ("roc_auc", "pr_auc"),
        ("ROC-AUC", "Precision-recall AUC"),
        strict=False,
    ):
        sns.barplot(
            data=results,
            x="feature_set",
            y=metric,
            hue="model",
            palette="Set2",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set(xlabel="Feature set", ylabel=title, title=title)
        ax.legend(title="Model", loc="lower right", fontsize=8)
    fig.suptitle("Validation model comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subgroups(rows: list[dict], group_name: str, path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame = frame[frame["positives"] >= 20].sort_values("n", ascending=True)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.42 * len(frame) + 1.2)))
    y = np.arange(len(frame))
    ax.barh(y - 0.18, frame["sensitivity"], height=0.36, color="#0F766E", label="Sensitivity")
    ax.barh(y + 0.18, frame["specificity"], height=0.36, color="#7C3AED", label="Specificity")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["group"])
    ax.set_xlim(0, 1)
    ax.set(xlabel="Score", title=f"Subgroup performance: {group_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def metrics_row(feature_set: str, model: str, metrics: dict) -> dict:
    return {
        "feature_set": feature_set,
        "model": model,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier": metrics["brier"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "f2": metrics["f2"],
    }
