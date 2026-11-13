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


def original_column(encoded_name: str, categorical: list[str]) -> str:
    """Map a preprocessed name like 'categorical__Month_Nov' back to 'Month'."""
    name = encoded_name.split("__", 1)[1]
    for col in categorical:
        if name.startswith(f"{col}_"):
            return col
    return name


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


def _balance_params(model: BaseEstimator, neg_pos_ratio: float | None) -> dict:
    """Return the settings that make a model weigh buyers more heavily."""
    name = type(model).__name__

    if name == "CatBoostClassifier":
        return {"auto_class_weights": "Balanced"}
    if name == "XGBClassifier":
        if neg_pos_ratio is None:
            raise ValueError("XGBClassifier needs neg_pos_ratio for class weighting")
        return {"scale_pos_weight": neg_pos_ratio}
    if "class_weight" in model.get_params():
        return {"class_weight": "balanced"}
    raise ValueError(f"{name} does not support class_weight")


def build_pipeline(
    model: BaseEstimator,
    strategy: str | None = None,
    settings: Settings | None = None,
    neg_pos_ratio: float | None = None,
) -> Pipeline:
    """Chain feature engineering, preprocessing, an imbalance step and the model.

    Strategies:
        none          - train on the data as it is (tune the threshold later)
        class_weight  - tell the model to weigh buyers more heavily
        smote         - create synthetic buyer rows, on training folds only

    neg_pos_ratio (non-buyers / buyers in the training data) is only needed by
    XGBoost, which takes the class weight as a number.
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
        model.set_params(**_balance_params(model, neg_pos_ratio))

    steps.append(("model", model))
    return Pipeline(steps=steps)
