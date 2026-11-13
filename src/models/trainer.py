"""A common interface for training and cross-validating every model."""

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
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


# Search spaces for tuning. "model__" points at the model step of the pipeline.
# uniform(a, b) samples from a to a + b.
PARAM_SPACES = {
    "logistic_regression": {"model__C": loguniform(1e-3, 10)},
    "decision_tree": {
        "model__max_depth": randint(3, 12),
        "model__min_samples_leaf": randint(5, 100),
    },
    "random_forest": {
        "model__n_estimators": randint(200, 600),
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_leaf": randint(1, 30),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(2, 7),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_weight": randint(1, 20),
    },
    "lightgbm": {
        "model__n_estimators": randint(100, 600),
        "model__num_leaves": randint(8, 64),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__min_child_samples": randint(10, 100),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_lambda": loguniform(1e-3, 10),
    },
    "catboost": {
        "model__iterations": randint(200, 600),
        "model__depth": randint(3, 8),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__l2_leaf_reg": loguniform(1, 10),
    },
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

    def _folds(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.settings.cv_folds,
            shuffle=True,
            random_state=self.settings.random_seed,
        )

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Return the mean and standard deviation of each metric over stratified folds."""
        scores = cross_validate(
            self.build(y),
            X,
            y,
            cv=self._folds(),
            scoring=CV_SCORING,
            n_jobs=-1,
            error_score="raise",
        )

        results = {}
        for metric in CV_SCORING:
            fold_scores = scores[f"test_{metric}"]
            results[f"{metric}_mean"] = round(float(fold_scores.mean()), 4)
            results[f"{metric}_std"] = round(float(fold_scores.std()), 4)
        return results

    def tune(self, X: pd.DataFrame, y: pd.Series, n_iter: int | None = None) -> dict:
        """Randomised search over PARAM_SPACES, picking the best mean PR-AUC.

        Returns the best parameters and the cross-validated metrics for them.
        Every candidate is scored on the same folds, so the comparison is fair.
        """
        search = RandomizedSearchCV(
            self.build(y),
            PARAM_SPACES[self.name],
            n_iter=n_iter or self.settings.tuning_iterations,
            scoring=CV_SCORING,
            refit=False,
            cv=self._folds(),
            random_state=self.settings.random_seed,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X, y)

        results = search.cv_results_
        best = int(results["rank_test_pr_auc"].argmin())
        scores = {}
        for metric in CV_SCORING:
            scores[f"{metric}_mean"] = round(float(results[f"mean_test_{metric}"][best]), 4)
            scores[f"{metric}_std"] = round(float(results[f"std_test_{metric}"][best]), 4)

        params = {
            key.removeprefix("model__"): value.item() if hasattr(value, "item") else value
            for key, value in results["params"][best].items()
        }
        return {"params": params, "scores": scores}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        """Train on all the given data and return the fitted pipeline."""
        return self.build(y).fit(X, y)
