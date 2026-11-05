"""Rank features with two methods and compare candidate feature sets.

Importance is measured inside cross-validation on the training split only:
    gain         - how much LightGBM's splits on a feature improve the fit
    permutation  - how much PR-AUC drops when a feature's values are shuffled

Run from the project root:
    python -m src.features.selection
"""

from dataclasses import replace

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.data.loader import load_data, split_data
from src.features.engineering import add_behavioural_features
from src.preprocessing.pipeline import build_pipeline
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

PERMUTATION_REPEATS = 5
# A feature is only a drop candidate when both methods call it weak.
MIN_GAIN_SHARE = 0.01
REPORT_DIR = PROJECT_ROOT / "reports"


def _lightgbm(settings: Settings) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        importance_type="gain",
        random_state=settings.random_seed,
        n_jobs=1,
        verbose=-1,
    )


def _original_column(encoded_name: str, categorical: list[str]) -> str:
    """Map a preprocessed name like 'categorical__Month_Nov' back to 'Month'."""
    name = encoded_name.split("__", 1)[1]
    for col in categorical:
        if name.startswith(f"{col}_"):
            return col
    return name


def rank_features(X: pd.DataFrame, y: pd.Series, settings: Settings) -> pd.DataFrame:
    """Return gain and permutation importance per column, averaged over the folds."""
    cv = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    gain_per_fold, permutation_per_fold = [], []

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        pipeline = build_pipeline(_lightgbm(settings), strategy="none", settings=settings)
        pipeline.fit(X.iloc[fit_idx], y.iloc[fit_idx])

        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        columns = [_original_column(name, settings.categorical_features) for name in names]
        gain = pd.Series(pipeline.named_steps["model"].feature_importances_, index=columns)
        gain = gain.groupby(level=0).sum()
        gain_per_fold.append(gain / gain.sum())

        # Shuffle the columns the model actually sees, after the features step.
        X_val = add_behavioural_features(X.iloc[val_idx])[list(gain.index)]
        result = permutation_importance(
            pipeline[1:],
            X_val,
            y.iloc[val_idx],
            scoring="average_precision",
            n_repeats=PERMUTATION_REPEATS,
            random_state=settings.random_seed,
        )
        permutation_per_fold.append(pd.Series(result.importances_mean, index=X_val.columns))
        logger.info("Finished fold %d of %d", fold, settings.cv_folds)

    gains = pd.concat(gain_per_fold, axis=1)
    permutations = pd.concat(permutation_per_fold, axis=1)
    ranking = pd.DataFrame(
        {
            "gain_share": gains.mean(axis=1),
            "permutation_mean": permutations.mean(axis=1),
            "permutation_std": permutations.std(axis=1),
        }
    )
    weak_gain = ranking["gain_share"] < MIN_GAIN_SHARE
    weak_permutation = ranking["permutation_mean"] <= ranking["permutation_std"]
    ranking["drop_candidate"] = weak_gain & weak_permutation
    return ranking.sort_values("permutation_mean", ascending=False)


def score_feature_set(
    X: pd.DataFrame, y: pd.Series, settings: Settings, dropped: list[str]
) -> dict[str, str]:
    """Cross-validated PR-AUC of LightGBM and Logistic Regression without `dropped`."""
    trial = replace(settings, dropped_features=dropped)
    cv = StratifiedKFold(n_splits=trial.cv_folds, shuffle=True, random_state=trial.random_seed)
    models = {
        "lightgbm": _lightgbm(trial),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=trial.random_seed),
    }

    scores = {}
    for name, model in models.items():
        pipeline = build_pipeline(model, strategy="none", settings=trial)
        fold_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
        scores[name] = f"{fold_scores.mean():.4f} ± {fold_scores.std():.4f}"
    return scores


def main() -> None:
    settings = replace(get_settings(), dropped_features=[])
    setup_logging(settings.logs_dir)

    X_train, _, y_train, _ = split_data(load_data(settings=settings), settings)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # PageValues is so strong that it can hide other features, and it may not be
    # available when we score a session, so rank the features with and without it.
    without_page_values = replace(settings, dropped_features=["PageValues"])
    scenarios = {"with_page_values": settings, "without_page_values": without_page_values}

    candidates_per_scenario = []
    for scenario, scenario_settings in scenarios.items():
        ranking = rank_features(X_train, y_train, scenario_settings)
        ranking.round(5).to_csv(REPORT_DIR / f"feature_importance_{scenario}.csv")
        logger.info("Ranking %s:\n%s", scenario, ranking.round(4).to_string())
        candidates_per_scenario.append(set(ranking.index[ranking["drop_candidate"]]))

    candidates = sorted(set.intersection(*candidates_per_scenario))
    logger.info("Weak in both scenarios: %s", candidates)

    engineered = settings.engineered_log_scaled_features + settings.engineered_scaled_features
    feature_sets = {
        "raw columns only": engineered,
        "raw + engineered": [],
        "without weak features": candidates,
        "no PageValues, raw only": engineered + ["PageValues"],
        "no PageValues, raw + engineered": ["PageValues"],
        "no PageValues, without weak features": candidates + ["PageValues"],
    }
    rows = {
        name: score_feature_set(X_train, y_train, settings, dropped)
        for name, dropped in feature_sets.items()
    }
    logger.info("PR-AUC by feature set:\n%s", pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
