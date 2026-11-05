"""Compare the three imbalance strategies with stratified cross-validation.

Every fold re-fits the whole pipeline (scaler, encoder, SMOTE, model) on its
own training part only, so no information leaks from the validation part.
Only the training split is used here; the test split stays untouched.

Run from the project root:
    python -m src.models.compare_imbalance
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config.settings import get_settings
from src.data.loader import load_data, split_data
from src.preprocessing.pipeline import IMBALANCE_STRATEGIES, build_pipeline
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

SCORING = {
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "roc_auc": "roc_auc",
}


def main() -> None:
    settings = get_settings()
    setup_logging(settings.logs_dir)

    df = load_data(settings=settings)
    X_train, _, y_train, _ = split_data(df, settings)

    cv = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    model = LogisticRegression(max_iter=1000, random_state=settings.random_seed)

    rows = []
    for strategy in IMBALANCE_STRATEGIES:
        pipeline = build_pipeline(model, strategy=strategy, settings=settings)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)

        row = {"strategy": strategy}
        for metric in SCORING:
            fold_scores = scores[f"test_{metric}"]
            row[metric] = f"{fold_scores.mean():.3f} ± {fold_scores.std():.3f}"
        rows.append(row)
        logger.info("Finished strategy '%s'", strategy)

    results = pd.DataFrame(rows).set_index("strategy")
    logger.info("Cross-validated results (threshold 0.5):\n%s", results.to_string())


if __name__ == "__main__":
    main()
