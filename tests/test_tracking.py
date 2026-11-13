from dataclasses import replace

import mlflow

from src.config.settings import get_settings
from src.models.tracking import start_tracking


def test_tracking_creates_experiment_in_given_location(tmp_path):
    settings = replace(
        get_settings(),
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        mlflow_artifacts_dir=tmp_path / "runs",
        mlflow_experiment="test-experiment",
    )
    start_tracking(settings)

    experiment = mlflow.get_experiment_by_name("test-experiment")
    assert experiment is not None
    assert experiment.artifact_location == (tmp_path / "runs").as_uri()

    with mlflow.start_run():
        mlflow.log_metric("pr_auc", 0.5)
    runs = mlflow.search_runs(experiment_names=["test-experiment"])
    assert runs["metrics.pr_auc"].tolist() == [0.5]
