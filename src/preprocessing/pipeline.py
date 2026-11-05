"""Preprocessing (ColumnTransformer) and the full training pipeline.

The preprocessing lives inside the same Pipeline as the model. That way,
cross-validation fits the scaler, encoder and resampler on each training
fold only, and the API can reuse the exact same fitted transforms.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config.settings import Settings, get_settings
from src.features.engineering import add_behavioural_features

IMBALANCE_STRATEGIES = ("none", "class_weight", "smote")


def build_preprocessor(settings: Settings | None = None) -> ColumnTransformer:
    """Create the (unfitted) ColumnTransformer for numeric and categorical columns."""
    settings = settings or get_settings()

    log_then_scale = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )

    # handle_unknown="ignore": a category never seen in training (e.g. a new
    # Browser ID) becomes all zeros instead of crashing at prediction time.
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("log_scaled", log_then_scale, settings.model_log_scaled),
            ("scaled", StandardScaler(), settings.model_scaled),
            ("categorical", one_hot, settings.model_categorical),
        ],
        remainder="drop",
    )


def build_pipeline(
    model: BaseEstimator,
    strategy: str | None = None,
    settings: Settings | None = None,
) -> Pipeline:
    """Chain feature engineering, preprocessing, an imbalance step and the model.

    Strategies:
        none          - train on the data as it is (tune the threshold later)
        class_weight  - tell the model to weigh buyers more heavily
        smote         - create synthetic buyer rows, on training folds only
    """
    settings = settings or get_settings()
    strategy = strategy or settings.imbalance_strategy

    if strategy not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown imbalance strategy '{strategy}'. Use one of {IMBALANCE_STRATEGIES}"
        )

    model = clone(model)
    steps = [
        ("features", FunctionTransformer(add_behavioural_features)),
        ("preprocess", build_preprocessor(settings)),
    ]

    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=settings.random_seed)))
    elif strategy == "class_weight":
        if "class_weight" not in model.get_params():
            raise ValueError(f"{type(model).__name__} does not support class_weight")
        model.set_params(class_weight="balanced")

    steps.append(("model", model))
    return Pipeline(steps=steps)
