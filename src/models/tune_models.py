"""Tune all six models and pick the best one by cross-validated PR-AUC.

Only the training split is used. The chosen parameters are saved to
reports/best_params.json for the final training step.

Run from the project root:
    python -m src.models.tune_models
"""

import json

import mlflow
import pandas as pd

from src.config.settings import PROJECT_ROOT, get_settings
from src.data.loader import load_data, split_data
from src.models.tracking import start_tracking
from src.models.trainer import ModelTrainer, get_models
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

REPORT_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    settings = get_settings()
    setup_logging(settings.logs_dir)
    start_tracking(settings)

    X_train, _, y_train, _ = split_data(load_data(settings=settings), settings)

    rows, best_params = [], {}
    for name, model in get_models(settings.random_seed).items():
        trainer = ModelTrainer(name, model, settings.imbalance_strategy, settings)
        before = trainer.cross_validate(X_train, y_train)
        tuned = trainer.tune(X_train, y_train)

        best_params[name] = tuned["params"]
        with mlflow.start_run(run_name=f"tune-{name}"):
            mlflow.set_tags({"stage": "tuning", "model_type": name})
            mlflow.log_params(
                {
                    **tuned["params"],
                    "imbalance_strategy": settings.imbalance_strategy,
                    "cv_folds": settings.cv_folds,
                    "tuning_iterations": settings.tuning_iterations,
                }
            )
            mlflow.log_metrics({f"cv_{key}": value for key, value in tuned["scores"].items()})
            mlflow.log_metric("cv_pr_auc_before_tuning", before["pr_auc_mean"])
        rows.append(
            {
                "model": name,
                "pr_auc_before": before["pr_auc_mean"],
                **tuned["scores"],
            }
        )
        logger.info(
            "%s: PR-AUC %.4f -> %.4f (± %.4f)",
            name,
            before["pr_auc_mean"],
            tuned["scores"]["pr_auc_mean"],
            tuned["scores"]["pr_auc_std"],
        )

    results = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    best_model = results.iloc[0]["model"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / "tuning_results.csv", index=False)
    with open(REPORT_DIR / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_model, "params": best_params}, f, indent=2)

    columns = ["model", "pr_auc_before", "pr_auc_mean", "pr_auc_std", "recall_mean", "roc_auc_mean"]
    logger.info("Tuning results:\n%s", results[columns].to_string(index=False))
    logger.info("Best model by cross-validated PR-AUC: %s", best_model)


if __name__ == "__main__":
    main()
