"""A common interface for training and cross-validating every model."""

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config.settings import Settings, get_settings
from src.models.evaluate import CV_SCORING
from src.preprocessing.pipeline import build_pipeline


def get_models(seed: int) -> dict[str, BaseEstimator]:
    """The six algorithms with sensible starting settings (tuned later).

    Each model uses one thread because cross-validation already runs the
    folds in parallel. To add a model, add one line here.
    """
    boosting = {"learning_rate": 0.05, "random_state": seed}
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1
        ),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, n_jobs=1, **boosting),
        "lightgbm": LGBMClassifier(n_estimators=300, n_jobs=1, verbose=-1, **boosting),
        "catboost": CatBoostClassifier(
            iterations=300,
            depth=4,
            thread_count=1,
            verbose=0,
            allow_writing_files=False,
            **boosting,
        ),
    }


class ModelTrainer:
    """Build, cross-validate and fit one model with one imbalance strategy."""

    def __init__(
        self,
        name: str,
        model: BaseEstimator,
        strategy: str = "none",
        settings: Settings | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.strategy = strategy
        self.settings = settings or get_settings()

    def build(self, y: pd.Series) -> Pipeline:
        """Create the unfitted pipeline. y is only used for the class ratio."""
        neg_pos_ratio = float((y == 0).sum() / (y == 1).sum())
        return build_pipeline(
            self.model,
            strategy=self.strategy,
            settings=self.settings,
            neg_pos_ratio=neg_pos_ratio,
        )

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Return the mean and standard deviation of each metric over stratified folds."""
        cv = StratifiedKFold(
            n_splits=self.settings.cv_folds,
            shuffle=True,
            random_state=self.settings.random_seed,
        )
        scores = cross_validate(
            self.build(y), X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1, error_score="raise"
        )

        results = {}
        for metric in CV_SCORING:
            fold_scores = scores[f"test_{metric}"]
            results[f"{metric}_mean"] = round(float(fold_scores.mean()), 4)
            results[f"{metric}_std"] = round(float(fold_scores.std()), 4)
        return results

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Train on all the given data and return the fitted pipeline."""
        return self.build(y).fit(X, y)
