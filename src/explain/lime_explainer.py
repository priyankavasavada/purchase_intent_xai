"""LIME explanations: a second, independent view of a single prediction.

LIME makes thousands of small changes to one session, watches how the model's
probability reacts, and fits a simple linear model to those reactions. Its
weights say which conditions (for example "ExitRates > 0.07") moved the
probability of a purchase, and by roughly how much.
"""

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer

from src.config.settings import Settings, get_settings
from src.features.engineering import SOURCE_COLUMNS


class LimeExplainer:
    """Explain one session at a time in terms of its raw input columns."""

    def __init__(
        self,
        pipeline: Pipeline,
        training_data: pd.DataFrame,
        settings: Settings | None = None,
        num_samples: int = 2000,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline
        self.num_samples = num_samples

        # Only explain columns the model actually uses, directly or through the
        # engineered features. Dropped columns would always get a weight of ~0.
        kept = set(self.settings.model_log_scaled + self.settings.model_scaled)
        kept |= set(self.settings.model_categorical) | set(SOURCE_COLUMNS)
        self.columns = [col for col in self.settings.input_columns if col in kept]

        self.categorical = [
            col for col in self.columns if col in self.settings.categorical_features
        ]
        self.categories = {col: sorted(training_data[col].unique()) for col in self.categorical}
        categorical_index = [self.columns.index(col) for col in self.categorical]

        self.explainer = LimeTabularExplainer(
            self._encode(training_data).to_numpy(dtype=float),
            feature_names=self.columns,
            class_names=["no purchase", "purchase"],
            categorical_features=categorical_index,
            categorical_names={
                self.columns.index(col): [str(value) for value in self.categories[col]]
                for col in self.categorical
            },
            discretize_continuous=True,
            random_state=self.settings.random_seed,
        )

    def _encode(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Replace each category with its position in the category list."""
        encoded = sessions[self.columns].copy()
        for col in self.categorical:
            lookup = {value: code for code, value in enumerate(self.categories[col])}
            encoded[col] = encoded[col].map(lookup)
        return encoded

    def _decode(self, rows: np.ndarray, session: pd.DataFrame) -> pd.DataFrame:
        """Turn LIME's numeric rows back into sessions the pipeline understands."""
        decoded = pd.DataFrame(rows, columns=self.columns)
        for col in self.categorical:
            codes = decoded[col].round().astype(int)
            decoded[col] = [self.categories[col][code] for code in codes]

        # Columns LIME does not vary keep the explained session's own values.
        for col in self.settings.input_columns:
            if col not in self.columns:
                decoded[col] = session[col].iloc[0]
        return decoded[self.settings.input_columns]

    def predict_function(self, session: pd.DataFrame):
        """The function LIME calls: numeric rows in, class probabilities out."""

        def predict(rows: np.ndarray) -> np.ndarray:
            return self.pipeline.predict_proba(self._decode(rows, session))

        return predict

    def explain(self, session: pd.DataFrame, n: int = 5) -> dict:
        """Return the n strongest conditions and how well the simple model fits.

        local_fit_r2 close to 1 means the linear model describes the real model
        well around this session; a low value means the explanation is rough.
        """
        if len(session) != 1:
            raise ValueError("LIME explains one session at a time")

        # Re-seed LIME's random generator so the same session always gets the same
        # answer. It is shared by LIME's internal parts, so seed it in place.
        self.explainer.random_state.seed(self.settings.random_seed)
        explanation = self.explainer.explain_instance(
            self._encode(session).iloc[0].to_numpy(dtype=float),
            self.predict_function(session),
            num_features=n,
            num_samples=self.num_samples,
        )

        return {
            "conditions": [
                {
                    "condition": condition,
                    "weight": round(float(weight), 4),
                    "effect": "increases" if weight > 0 else "decreases",
                }
                for condition, weight in explanation.as_list(label=1)
            ],
            "local_fit_r2": round(float(explanation.score), 4),
        }
