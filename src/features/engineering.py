"""Behavioural features built from the raw session columns.

add_behavioural_features() runs as a step inside the model pipeline, so
training and serving always compute these columns the same way.
"""

import numpy as np
import pandas as pd

# Raw columns the engineered features are built from.
SOURCE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two columns, returning 0 where the denominator is 0."""
    return (numerator / denominator.replace(0, np.nan)).fillna(0.0)


def add_behavioural_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the sessions with the engineered columns added."""
    df = sessions.copy()

    total_pages = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    total_duration = (
        df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"]
    )

    df["total_pages"] = total_pages
    df["total_duration"] = total_duration
    # How long the visitor lingers on a page, overall and on products.
    df["avg_time_per_page"] = _safe_ratio(total_duration, total_pages)
    df["avg_time_per_product_page"] = _safe_ratio(
        df["ProductRelated_Duration"], df["ProductRelated"]
    )
    # How much of the visit is about products rather than account or info pages.
    df["product_page_share"] = _safe_ratio(df["ProductRelated"], total_pages)
    df["product_time_share"] = _safe_ratio(df["ProductRelated_Duration"], total_duration)

    return df
