"""Save and load the trained pipeline together with its metadata."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from imblearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelArtifact:
    """A fitted pipeline (preprocessing + model) plus what we know about it."""

    pipeline: Pipeline
    metadata: dict

    @property
    def threshold(self) -> float:
        return self.metadata["threshold"]

    @property
    def feature_columns(self) -> list[str]:
        return self.metadata["feature_columns"]


def save_artifact(
    pipeline: Pipeline,
    name: str,
    models_dir: Path,
    threshold: float,
    feature_columns: list[str],
    metrics: dict,
) -> Path:
    """Write <name>.joblib and <name>.json into models_dir and return the model path."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    metadata_path = models_dir / f"{name}.json"

    metadata = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "feature_columns": feature_columns,
        "test_metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }

    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model to %s", model_path)
    return model_path


def load_artifact(name: str, models_dir: Path) -> ModelArtifact:
    """Load a pipeline and its metadata saved by save_artifact()."""
    model_path = models_dir / f"{name}.joblib"
    metadata_path = models_dir / f"{name}.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No saved model '{name}' in {models_dir}. Train one first with "
            "`python -m src.models.train_baseline`."
        )

    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata["sklearn_version"] != sklearn.__version__:
        logger.warning(
            "Model was saved with scikit-learn %s but %s is installed",
            metadata["sklearn_version"],
            sklearn.__version__,
        )

    logger.info("Loaded model '%s' (created %s)", name, metadata["created_at"])
    return ModelArtifact(pipeline=pipeline, metadata=metadata)
