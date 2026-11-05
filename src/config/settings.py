"""Load config.yaml into a typed, read-only Settings object."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class Settings:
    """All tunable values for the project, loaded from config.yaml."""

    random_seed: int
    data_file: Path
    models_dir: Path
    logs_dir: Path
    target: str
    test_size: float
    log_scaled_features: list[str]
    scaled_features: list[str]
    categorical_features: list[str]
    engineered_log_scaled_features: list[str]
    engineered_scaled_features: list[str]
    dropped_features: list[str]
    imbalance_strategy: str
    threshold: float
    cv_folds: int

    @property
    def numeric_features(self) -> list[str]:
        return self.log_scaled_features + self.scaled_features

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def _kept(self, columns: list[str]) -> list[str]:
        return [col for col in columns if col not in self.dropped_features]

    @property
    def model_log_scaled(self) -> list[str]:
        return self._kept(self.log_scaled_features + self.engineered_log_scaled_features)

    @property
    def model_scaled(self) -> list[str]:
        return self._kept(self.scaled_features + self.engineered_scaled_features)

    @property
    def model_categorical(self) -> list[str]:
        return self._kept(self.categorical_features)


@lru_cache
def get_settings(config_file: Path = CONFIG_FILE) -> Settings:
    """Read the YAML config once and return it as a Settings object.

    Relative paths in the config are resolved against the project root,
    so the code works no matter which folder it is run from.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    features = config["features"]
    training = config["training"]

    return Settings(
        random_seed=config["project"]["random_seed"],
        data_file=PROJECT_ROOT / paths["data_file"],
        models_dir=PROJECT_ROOT / paths["models_dir"],
        logs_dir=PROJECT_ROOT / paths["logs_dir"],
        target=config["data"]["target"],
        test_size=config["data"]["test_size"],
        log_scaled_features=features["log_scaled"],
        scaled_features=features["scaled"],
        categorical_features=features["categorical"],
        engineered_log_scaled_features=features["engineered_log_scaled"],
        engineered_scaled_features=features["engineered_scaled"],
        dropped_features=features["dropped"],
        imbalance_strategy=training["imbalance_strategy"],
        threshold=training["threshold"],
        cv_folds=training["cv_folds"],
    )
