"""Cached loaders shared by every dashboard section.

Nothing here trains a model: it only reads the data, the saved models and the
reports in reports/. Streamlit re-runs the whole script on every click, so:
    st.cache_data      keeps tables and results (copied for each caller)
    st.cache_resource  keeps one shared copy of models and explainers
"""

import json
import os

import httpx
import numpy as np
import pandas as pd
import streamlit as st

from src.api.scoring import Scorer
from src.config.settings import PROJECT_ROOT, get_settings
from src.data.loader import load_data, split_data
from src.explain.lime_explainer import LimeExplainer
from src.models.artifacts import ModelArtifact, latest_version_name, load_artifact

settings = get_settings()
REPORTS = PROJECT_ROOT / "reports"

# Set by docker-compose so the dashboard scores through the API. When it is
# empty, the dashboard loads the saved model itself.
API_URL = os.getenv("API_URL", "").rstrip("/")

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MAIN_MODEL = "purchase_intent"
COMPARISON_MODEL = "purchase_intent_with_page_values"


@st.cache_data
def sessions() -> pd.DataFrame:
    return load_data()


@st.cache_data
def test_split() -> tuple[pd.DataFrame, pd.Series]:
    _, X_test, _, y_test = split_data(sessions())
    return X_test, y_test


@st.cache_data
def train_features() -> pd.DataFrame:
    X_train, _, _, _ = split_data(sessions())
    return X_train


@st.cache_data
def report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


@st.cache_data
def final_models() -> dict:
    return json.loads((REPORTS / "final_models.json").read_text(encoding="utf-8"))


@st.cache_resource
def scorer() -> Scorer:
    """The real model and its SHAP explainer, loaded once."""
    return Scorer.load(settings)


@st.cache_resource
def comparison_model() -> ModelArtifact:
    """The model with PageValues. For comparison only, never for decisions."""
    return load_artifact(
        latest_version_name(settings.models_dir, COMPARISON_MODEL), settings.models_dir
    )


@st.cache_resource
def lime_explainer() -> LimeExplainer:
    return LimeExplainer(scorer().artifact.pipeline, train_features())


@st.cache_data
def test_probabilities() -> dict[str, np.ndarray]:
    """Predicted probabilities on the test split, for both models."""
    X_test, _ = test_split()
    probabilities = {}
    for name, artifact in [(MAIN_MODEL, scorer().artifact), (COMPARISON_MODEL, comparison_model())]:
        columns = X_test[artifact.feature_columns]
        probabilities[name] = artifact.pipeline.predict_proba(columns)[:, 1]
    return probabilities


def score_session(session: dict, threshold: float) -> dict:
    """Score one session through the API if API_URL is set, otherwise locally."""
    if API_URL:
        response = httpx.post(
            f"{API_URL}/predict", json=session, params={"threshold": threshold}, timeout=10
        )
        response.raise_for_status()
        return response.json()
    return scorer().score(session, threshold)


def api_health() -> dict | None:
    """The API's /health answer, or None if no API is configured or it is down."""
    if not API_URL:
        return None
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).json()
    except httpx.HTTPError:
        return None
