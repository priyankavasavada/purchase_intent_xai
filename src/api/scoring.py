"""Scoring logic used by the API: probability, decision and explanation."""

import pandas as pd

from src.config.settings import Settings
from src.explain.shap_explainer import ShapExplainer
from src.models.artifacts import ModelArtifact, latest_version_name, load_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_PREFIX = "purchase_intent"


def confidence_label(
    probability: float, threshold: float, low_below: float, medium_below: float
) -> str:
    """Label how clear-cut a decision is by its distance from the threshold.

    A session just above or below the threshold could easily have gone the
    other way, so it gets "low". This is a simple rule of thumb, not a
    statistical measure of certainty. The bands come from config.yaml.
    """
    distance = abs(probability - threshold)
    if distance < low_below:
        return "low"
    if distance < medium_below:
        return "medium"
    return "high"


class Scorer:
    """Holds the loaded model and explainer, so both are built only once."""

    def __init__(self, artifact: ModelArtifact, settings: Settings) -> None:
        self.artifact = artifact
        self.settings = settings
        self.explainer = ShapExplainer(artifact.pipeline, settings)

    @classmethod
    def load(cls, settings: Settings) -> "Scorer":
        """Load the newest saved purchase_intent model."""
        name = latest_version_name(settings.models_dir, MODEL_PREFIX)
        return cls(load_artifact(name, settings.models_dir), settings)

    @property
    def model_version(self) -> str:
        return self.artifact.metadata["name"]

    def score(self, session: dict, threshold: float | None = None) -> dict:
        """Score one session and explain the result."""
        threshold = self.artifact.threshold if threshold is None else threshold
        frame = pd.DataFrame([session])[self.artifact.feature_columns]

        probability = float(self.artifact.pipeline.predict_proba(frame)[0, 1])
        return {
            "model_version": self.model_version,
            "conversion_probability": round(probability, 4),
            "will_convert": probability >= threshold,
            "threshold": threshold,
            "confidence": confidence_label(
                probability,
                threshold,
                self.settings.confidence_low_below,
                self.settings.confidence_medium_below,
            ),
            "top_contributors": self.explainer.top_contributors(
                frame, n=self.settings.api_top_contributors
            ),
        }
