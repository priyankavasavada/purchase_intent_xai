"""Compare the six models with stratified cross-validation.

Every model runs with each imbalance strategy, with and without PageValues.
Only the training split is used; the test split stays untouched. Metrics that
depend on a threshold use 0.5 for now.

Run from the project root:
    python -m src.models.compare_models
"""

from dataclasses import replace

import pandas as pd

from src.config.settings import PROJECT_ROOT, get_settings
from src.data.loader import load_data, split_data
from src.models.trainer import ModelTrainer, get_models
from src.preprocessing.pipeline import IMBALANCE_STRATEGIES
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

REPORT_FILE = PROJECT_ROOT / "reports" / "model_comparison.csv"
SUMMARY_COLUMNS = ["pr_auc_mean", "pr_auc_std", "recall_mean", "precision_mean", "accuracy_mean"]


def main() -> None:
    settings = get_settings()
    setup_logging(settings.logs_dir)

    X_train, _, y_train, _ = split_data(load_data(settings=settings), settings)

    without_page_values = replace(
        settings, dropped_features=settings.dropped_features + ["PageValues"]
    )
    feature_sets = {"with_page_values": settings, "without_page_values": without_page_values}

    rows = []
    for feature_set, trial_settings in feature_sets.items():
        for name, model in get_models(settings.random_seed).items():
            for strategy in IMBALANCE_STRATEGIES:
                trainer = ModelTrainer(name, model, strategy, trial_settings)
                scores = trainer.cross_validate(X_train, y_train)
                rows.append(
                    {"feature_set": feature_set, "model": name, "strategy": strategy, **scores}
                )
                logger.info(
                    "%s | %s | %s | PR-AUC %.4f",
                    feature_set,
                    name,
                    strategy,
                    scores["pr_auc_mean"],
                )

    results = pd.DataFrame(rows)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_FILE, index=False)
    logger.info("Saved results to %s", REPORT_FILE)

    best = results.sort_values("pr_auc_mean", ascending=False).groupby(["feature_set", "model"])
    summary = best.head(1).set_index(["feature_set", "model"]).sort_index()
    logger.info("Best strategy per model:\n%s", summary[["strategy"] + SUMMARY_COLUMNS].to_string())


if __name__ == "__main__":
    main()
