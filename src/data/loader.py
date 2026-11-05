"""Load, validate and split the e-commerce sessions dataset.

Notebooks, training scripts, the API and the dashboard should all read the
data through load_data() so they see exactly the same thing.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset does not match the expected schema."""


def load_data(path: Path | None = None, settings: Settings | None = None) -> pd.DataFrame:
    """Read the sessions CSV and validate it before returning it."""
    settings = settings or get_settings()
    path = Path(path) if path else settings.data_file

    if not path.exists():
        logger.error("Data file not found: %s", path)
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as err:
        logger.error("Data file is empty: %s", path)
        raise DataValidationError(f"Data file is empty: {path}") from err

    validate_data(df, settings)
    logger.info("Loaded %d rows and %d columns from %s", len(df), df.shape[1], path.name)
    return df


def validate_data(df: pd.DataFrame, settings: Settings) -> None:
    """Check columns, dtypes, missing values and the target.

    Raises DataValidationError on any problem that would break training.
    Duplicates are only logged, because two sessions can legitimately look
    the same.
    """
    expected = settings.feature_columns + [settings.target]
    missing_columns = [col for col in expected if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing columns: {missing_columns}")

    non_numeric = [
        col for col in settings.numeric_features if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise DataValidationError(f"Columns should be numeric but are not: {non_numeric}")

    negative = [col for col in settings.numeric_features if (df[col] < 0).any()]
    if negative:
        raise DataValidationError(f"Columns contain negative values: {negative}")

    null_counts = df[expected].isna().sum()
    if null_counts.any():
        raise DataValidationError(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")

    target_values = set(df[settings.target].unique())
    if not target_values <= {0, 1}:
        raise DataValidationError(
            f"Target '{settings.target}' must be 0/1, got {sorted(target_values)}"
        )

    duplicates = int(df.duplicated().sum())
    if duplicates:
        logger.warning("Found %d duplicate rows (kept)", duplicates)

    positive_rate = df[settings.target].mean()
    logger.info("Validation passed. Conversion rate: %.1f%%", positive_rate * 100)


def split_data(
    df: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split, so both sets keep the ~16% conversion rate.

    The fixed random seed means every run produces the same split.
    """
    settings = settings or get_settings()
    X = df[settings.feature_columns]
    y = df[settings.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        stratify=y,
        random_state=settings.random_seed,
    )
    logger.info("Split data: %d train rows, %d test rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test
