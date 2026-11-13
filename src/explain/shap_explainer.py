"""SHAP explanations: which behaviours pushed a session's score up or down.

For tree models, SHAP's TreeExplainer gives exact contributions and is fast,
so the explainer is built once and reused for every session.

Contributions are in log-odds, the model's internal score before it becomes a
probability. Positive values push towards a purchase, negative values away
from it, and base_value plus all contributions equals the model's score.
"""

import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.config.settings import Settings, get_settings
from src.features.engineering import add_behavioural_features
from src.preprocessing.pipeline import original_column


class ShapExplainer:
    """Explain a fitted pipeline in terms of the original session columns."""

    def __init__(
        self,
        pipeline: Pipeline,
        settings: Settings | None = None,
        background: pd.DataFrame | None = None,
    ) -> None:
        settings = settings or get_settings()
        model = pipeline.named_steps["model"]
        # Everything before the model: feature engineering and preprocessing.
        self.transform = pipeline[:-1]

        encoded_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        self.columns = [
            original_column(name, settings.categorical_features) for name in encoded_names
        ]

        if isinstance(model, LogisticRegression):
            # A linear model needs sample data to know what "average" looks like.
            if background is None:
                raise ValueError("Logistic regression needs background data to explain")
            self.explainer = shap.LinearExplainer(model, self.transform.transform(background))
        else:
            self.explainer = shap.TreeExplainer(model)

    @property
    def base_value(self) -> float:
        """The model's average score, before any feature pushes it up or down."""
        value = np.ravel(self.explainer.expected_value)
        return float(value[-1])

    def contributions(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """One row per session, one column per original feature.

        The one-hot columns of a categorical feature (Month_Nov, Month_Dec, ...)
        are added together, so each feature appears once.
        """
        values = self.explainer.shap_values(self.transform.transform(sessions))
        # Some models return one array per class; keep the buyer class.
        if isinstance(values, list):
            values = values[-1]
        if np.ndim(values) == 3:
            values = values[:, :, -1]

        per_encoded_column = pd.DataFrame(values, columns=self.columns, index=sessions.index)
        return per_encoded_column.T.groupby(level=0, sort=False).sum().T

    def top_contributors(self, session: pd.DataFrame, n: int = 5) -> list[dict]:
        """The n features that moved this session's score the most."""
        if len(session) != 1:
            raise ValueError("top_contributors explains one session at a time")

        contributions = self.contributions(session).iloc[0]
        top = contributions.reindex(contributions.abs().sort_values(ascending=False).index)[:n]
        values = add_behavioural_features(session).iloc[0]

        return [
            {
                "feature": feature,
                "value": _plain(values[feature]),
                "contribution": round(float(contribution), 4),
                "effect": "increases" if contribution > 0 else "decreases",
            }
            for feature, contribution in top.items()
        ]

    def global_importance(self, sessions: pd.DataFrame) -> pd.Series:
        """Average absolute contribution per feature: how much it matters overall."""
        return self.contributions(sessions).abs().mean().sort_values(ascending=False)


def _plain(value):
    """Turn numpy numbers into plain Python ones so they can be sent as JSON."""
    return value.item() if hasattr(value, "item") else value
