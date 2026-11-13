"""Train the chosen model, set its default threshold and save it.

Uses the best model and parameters from tune_models. Two versions are saved:
    purchase_intent                   - the real model, without PageValues
    purchase_intent_with_page_values  - for comparison in the dashboard only

The default threshold comes from out-of-fold predictions on the training
split. The test split is used once, at the end, to report final results.

Run from the project root:
    python -m src.models.train_final
"""

import json
from dataclasses import replace

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.data.loader import load_data, split_data
from src.models.artifacts import next_version_name, save_artifact
from src.models.evaluate import compute_metrics
from src.models.threshold import default_threshold, operating_points
from src.models.tracking import start_tracking
from src.models.trainer import ModelTrainer, get_models
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

REPORT_DIR = PROJECT_ROOT / "reports"
THRESHOLD_RULE = (
    "Best F1 on out-of-fold training predictions. A neutral default only: "
    "the growth/CRO team should pick the operating point from its own "
    "conversion value and nudge cost."
)


def train_variant(
    prefix: str,
    settings: Settings,
    model_name: str,
    params: dict,
    data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    register: bool,
) -> dict:
    """Train, evaluate and save one version of the model. Returns its summary."""
    X_train, X_test, y_train, y_test = data
    model = get_models(settings.random_seed)[model_name].set_params(**params)
    trainer = ModelTrainer(model_name, model, settings.imbalance_strategy, settings)

    # Out-of-fold probabilities: each training row is scored by a model that never saw it.
    cv = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    oof_proba = cross_val_predict(
        trainer.build(y_train), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = default_threshold(operating_points(y_train, oof_proba))

    pipeline = trainer.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, test_proba >= threshold).ravel()
    confusion = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # The trade-off on unseen data, for the dashboard.
    points_file = REPORT_DIR / f"operating_points_{prefix}.csv"
    operating_points(y_test, test_proba).to_csv(points_file, index=False)

    name = next_version_name(settings.models_dir, prefix)
    save_artifact(
        pipeline=pipeline,
        name=name,
        models_dir=settings.models_dir,
        threshold=threshold,
        feature_columns=settings.input_columns,
        metrics=metrics,
        extra={
            "model_type": model_name,
            "params": params,
            "threshold_rule": THRESHOLD_RULE,
            "test_confusion_matrix": confusion,
            "leakage_excluded_features": settings.leakage_excluded_features,
        },
    )
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"stage": "final", "model_type": model_name, "variant": prefix})
        mlflow.log_params(
            {
                **params,
                "imbalance_strategy": settings.imbalance_strategy,
                "leakage_excluded": ",".join(settings.leakage_excluded_features) or "none",
                "dropped_features": ",".join(settings.dropped_features),
            }
        )
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in confusion.items()})
        mlflow.log_artifact(str(points_file))

        # The signature records the expected input columns and the output type.
        X_input = X_test[settings.input_columns]
        signature = infer_signature(X_input.head(100), pipeline.predict_proba(X_input.head(100)))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_input.head(3),
            code_paths=[str(PROJECT_ROOT / "src")],
            # Same format as the joblib file. MLflow's default (skops) refuses the
            # pipeline's custom classes unless each one is listed as trusted.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=settings.mlflow_registered_model if register else None,
        )

    logger.info("%s: threshold %.2f, test metrics %s", name, threshold, metrics)
    logger.info("%s: test confusion matrix %s", name, confusion)
    return {"name": name, "threshold": threshold, "test_metrics": metrics, "confusion": confusion}


def main() -> None:
    settings = get_settings()
    setup_logging(settings.logs_dir)
    start_tracking(settings)

    params_file = REPORT_DIR / "best_params.json"
    if not params_file.exists():
        raise FileNotFoundError("Run `python -m src.models.tune_models` first.")
    tuning = json.loads(params_file.read_text(encoding="utf-8"))
    model_name = tuning["best_model"]
    params = tuning["params"][model_name]
    logger.info("Training %s with %s", model_name, params)

    data = split_data(load_data(settings=settings), settings)
    variants = {
        "purchase_intent": settings,
        "purchase_intent_with_page_values": replace(settings, leakage_excluded_features=[]),
    }
    # Only the real model goes into the registry; the PageValues one is for comparison.
    summary = {
        prefix: train_variant(
            prefix, variant_settings, model_name, params, data, register=prefix == "purchase_intent"
        )
        for prefix, variant_settings in variants.items()
    }

    with open(REPORT_DIR / "final_models.json", "w", encoding="utf-8") as f:
        json.dump({"model_type": model_name, "variants": summary}, f, indent=2)
    logger.info("Saved summary to %s", REPORT_DIR / "final_models.json")


if __name__ == "__main__":
    main()
