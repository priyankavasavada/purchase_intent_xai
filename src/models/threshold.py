"""Operating points: what happens at each decision threshold.

The model gives every session a conversion probability. Where to draw the line
between "nudge" and "leave alone" is a business decision, owned by the growth /
CRO team. These helpers make that trade-off visible:

    operating_points()    precision, recall and counts at every threshold
    add_expected_value()  turns a conversion value and a nudge cost into money
    value_threshold()     the threshold with the highest net value
    default_threshold()   a neutral default (best F1) until the team picks one
"""

import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def operating_points(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: np.ndarray = THRESHOLDS
) -> pd.DataFrame:
    """Return one row per threshold with the confusion counts and the main metrics."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rows = []
    for threshold in thresholds:
        flagged = y_proba >= threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        tn = int((~flagged & (y_true == 0)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        rows.append(
            {
                "threshold": float(threshold),
                "flagged_share": round(float(flagged.mean()), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def add_expected_value(
    points: pd.DataFrame, conversion_value: float, intervention_cost: float
) -> pd.DataFrame:
    """Add captured value, intervention spend and net value to each operating point.

    conversion_value should be the *extra* value a nudge brings when it reaches a
    buyer, not the full order value. Many flagged buyers would have bought anyway,
    and this data cannot tell us how many.
    """
    points = points.copy()
    nudges = points["tp"] + points["fp"]
    points["captured_value"] = points["tp"] * conversion_value
    points["intervention_spend"] = nudges * intervention_cost
    points["net_value"] = points["captured_value"] - points["intervention_spend"]
    return points


def value_threshold(
    points: pd.DataFrame, conversion_value: float, intervention_cost: float
) -> float | None:
    """Return the threshold with the highest net value.

    Returns None when no threshold makes money, meaning nudging is not worth it
    at these numbers.
    """
    valued = add_expected_value(points, conversion_value, intervention_cost)
    best = valued.loc[valued["net_value"].idxmax()]
    if best["net_value"] <= 0:
        return None
    return float(best["threshold"])


def default_threshold(points: pd.DataFrame) -> float:
    """A neutral starting threshold: the one with the best F1 score.

    It balances precision and recall without assuming any money values. It is
    only a default; the team replaces it once it knows its own value and cost.
    """
    return float(points.loc[points["f1"].idxmax(), "threshold"])
