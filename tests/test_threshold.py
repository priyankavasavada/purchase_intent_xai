import numpy as np
import pytest

from src.models.threshold import (
    add_expected_value,
    default_threshold,
    operating_points,
    value_threshold,
)

# Four buyers and six non-buyers with made-up probabilities.
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PROBA = np.array([0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.2, 0.1, 0.1, 0.05])


@pytest.fixture
def points():
    return operating_points(Y_TRUE, Y_PROBA, thresholds=np.array([0.25, 0.5, 0.75]))


def test_counts_add_up(points):
    totals = points[["tp", "fp", "fn", "tn"]].sum(axis=1)
    assert (totals == len(Y_TRUE)).all()


def test_values_at_threshold_half(points):
    row = points[points["threshold"] == 0.5].iloc[0]

    # Flagged: 0.9, 0.8, 0.6 (buyers) and 0.7 (non-buyer).
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (3, 1, 1, 5)
    assert row["precision"] == 0.75
    assert row["recall"] == 0.75
    assert row["flagged_share"] == 0.4


def test_higher_threshold_flags_fewer_sessions(points):
    assert points["flagged_share"].is_monotonic_decreasing


def test_expected_value(points):
    valued = add_expected_value(points, conversion_value=10, intervention_cost=1)
    row = valued[valued["threshold"] == 0.5].iloc[0]

    assert row["captured_value"] == 30
    assert row["intervention_spend"] == 4
    assert row["net_value"] == 26


def test_cheap_nudges_favour_a_low_threshold(points):
    assert value_threshold(points, conversion_value=10, intervention_cost=0.1) == 0.25


def test_expensive_nudges_favour_a_high_threshold(points):
    assert value_threshold(points, conversion_value=10, intervention_cost=6) == 0.75


def test_no_threshold_when_nudging_never_pays(points):
    assert value_threshold(points, conversion_value=1, intervention_cost=5) is None


def test_default_threshold_has_the_best_f1(points):
    best = default_threshold(points)
    assert points.loc[points["threshold"] == best, "f1"].item() == points["f1"].max()
