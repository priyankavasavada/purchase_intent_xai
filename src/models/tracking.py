"""MLflow setup shared by the training scripts."""

import mlflow

from src.config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def start_tracking(settings: Settings) -> None:
    """Point MLflow at the project's local database and select the experiment."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    if mlflow.get_experiment_by_name(settings.mlflow_experiment) is None:
        # Keep run files inside the project, not in whatever folder we ran from.
        mlflow.create_experiment(
            settings.mlflow_experiment,
            artifact_location=settings.mlflow_artifacts_dir.as_uri(),
        )
    mlflow.set_experiment(settings.mlflow_experiment)
    logger.info("Logging to MLflow experiment '%s'", settings.mlflow_experiment)
