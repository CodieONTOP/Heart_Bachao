from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from heartrisk import config
from heartrisk.predict import load_bundle


@dataclass(frozen=True)
class ChartPalette:
    text: str
    grid: str
    accent: str
    positive: str
    negative: str
    threshold: str
    gauge_bg: str
    band_low: str
    band_moderate: str
    band_high: str
    baseline: str


def chart_palette() -> ChartPalette:
    if st.context.theme.type == "dark":
        return ChartPalette(
            text="#E2E8F0",
            grid="#475569",
            accent="#2DD4BF",
            positive="#F87171",
            negative="#2DD4BF",
            threshold="#F8FAFC",
            gauge_bg="#0B1220",
            band_low="#064E3B",
            band_moderate="#78350F",
            band_high="#7F1D1D",
            baseline="#64748B",
        )
    return ChartPalette(
        text="#0F172A",
        grid="#94A3B8",
        accent="#0F766E",
        positive="#B91C1C",
        negative="#0F766E",
        threshold="#111827",
        gauge_bg="#FFFFFF",
        band_low="#D1FAE5",
        band_moderate="#FEF3C7",
        band_high="#FEE2E2",
        baseline="#94A3B8",
    )


@st.cache_resource(show_spinner=False)
def get_bundle(feature_set: str) -> dict:
    return load_bundle(feature_set)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    return json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_metadata() -> dict:
    path = config.resolve_models_dir() / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def artifacts_ready() -> bool:
    model_dir = config.resolve_models_dir()
    return all(
        (model_dir / f"{feature_set}_bundle.joblib").exists()
        for feature_set in config.FEATURE_SETS
    ) and config.METRICS_JSON.exists()


def figure_path(name: str) -> Path:
    return config.FIGURES_DIR / name


def show_figure(name: str, caption: str | None = None) -> bool:
    path = figure_path(name)
    if not path.exists():
        st.caption(f"Figure `{name}` is not available. Run the training script to generate it.")
        return False
    st.image(str(path), caption=caption, width="stretch")
    return True


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def signed_pct_points(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f} pts"


def ci_text(interval: dict, digits: int = 3) -> str:
    return f"{interval['lower']:.{digits}f} to {interval['upper']:.{digits}f}"
