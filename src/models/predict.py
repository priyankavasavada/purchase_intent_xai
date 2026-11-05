"""Score sessions with a saved model.

Run from the project root:
    python -m src.models.predict examples/sample_session.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.models.artifacts import ModelArtifact, load_artifact
from src.models.train_baseline import MODEL_NAME
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def predict_sessions(sessions: pd.DataFrame, artifact: ModelArtifact) -> pd.DataFrame:
    """Return the conversion probability and decision for each session."""
    missing = [col for col in artifact.feature_columns if col not in sessions.columns]
    if missing:
        raise ValueError(f"Sessions are missing required columns: {missing}")

    probabilities = artifact.pipeline.predict_proba(sessions[artifact.feature_columns])[:, 1]
    return pd.DataFrame(
        {
            "conversion_probability": probabilities.round(4),
            "will_convert": probabilities >= artifact.threshold,
        },
        index=sessions.index,
    )


def read_sessions(path: Path) -> pd.DataFrame:
    """Read a JSON file holding one session (object) or several (list)."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict purchase intent for sessions.")
    parser.add_argument("input", type=Path, help="JSON file with one session or a list")
    parser.add_argument("--model", default=MODEL_NAME, help="Name of the saved model")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging()

    artifact = load_artifact(args.model, settings.models_dir)
    sessions = read_sessions(args.input)
    predictions = predict_sessions(sessions, artifact)

    logger.info("Threshold: %.2f\n%s", artifact.threshold, predictions.to_string())


if __name__ == "__main__":
    main()
