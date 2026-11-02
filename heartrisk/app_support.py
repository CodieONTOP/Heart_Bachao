from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from heartrisk import config
from heartrisk.predict import load_bundle


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
