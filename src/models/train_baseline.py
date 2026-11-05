"""Train the Week-1 baseline and save it as an artifact.

This is the "thin slice": raw CSV -> pipeline -> saved model. It is not
meant to be a good model yet, only a working end-to-end path.

Run from the project root:
    python -m src.models.train_baseline
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.config.settings import get_settings
from src.data.loader import load_data, split_data
from src.models.artifacts import save_artifact
from src.models.evaluate import compute_metrics
from src.preprocessing.pipeline import build_pipeline
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

MODEL_NAME = "baseline_logreg"


def main() -> None:
    settings = get_settings()
    setup_logging(settings.logs_dir)

    df = load_data(settings=settings)
    X_train, X_test, y_train, y_test = split_data(df, settings)

    # Reference point: always predict "no purchase". High accuracy, zero recall.
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_metrics = compute_metrics(y_test, dummy.predict_proba(X_test)[:, 1], settings.threshold)
    logger.info("Dummy baseline metrics: %s", dummy_metrics)

    model = LogisticRegression(max_iter=1000, random_state=settings.random_seed)
    pipeline = build_pipeline(model, settings=settings)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_proba, settings.threshold)
    logger.info("Logistic regression metrics: %s", metrics)

    save_artifact(
        pipeline=pipeline,
        name=MODEL_NAME,
        models_dir=settings.models_dir,
        threshold=settings.threshold,
        feature_columns=settings.feature_columns,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
