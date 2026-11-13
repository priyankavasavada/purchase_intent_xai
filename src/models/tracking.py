"""MLflow setup shared by the training scripts, and export of the run log.

Run from the project root to save the run log as CSV:
    python -m src.models.tracking
"""

from pathlib import Path

import mlflow
import pandas as pd

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


def export_runs(settings: Settings, path: Path) -> pd.DataFrame:
    """Save a table of all MLflow runs (names, tags, parameters, metrics) as CSV.

    The MLflow database stays on the machine that trained the model; this CSV
    is the run log that goes into the repository.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=[settings.mlflow_experiment])

    keep = ["tags.mlflow.runName", "tags.stage", "tags.model_type", "status", "start_time"]
    keep += sorted(col for col in runs.columns if col.startswith(("params.", "metrics.")))
    table = runs[keep].rename(columns={"tags.mlflow.runName": "run_name"})
    table.columns = [col.removeprefix("tags.") for col in table.columns]
    table = table.sort_values("start_time")

    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info("Exported %d MLflow runs to %s", len(table), path)
    return table


if __name__ == "__main__":
    from src.config.settings import PROJECT_ROOT, get_settings
    from src.utils.logger import setup_logging

    setup_logging()
    export_runs(get_settings(), PROJECT_ROOT / "reports" / "mlflow_runs.csv")
