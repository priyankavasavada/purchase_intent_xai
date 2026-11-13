import numpy as np
import pytest
from xgboost import XGBClassifier

from src.data.loader import load_data, split_data
from src.explain.lime_explainer import LimeExplainer
from src.preprocessing.pipeline import build_pipeline


@pytest.fixture(scope="module")
def setup():
    X_train, X_test, y_train, _ = split_data(load_data())
    X, y = X_train.head(1500), y_train.head(1500)
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=1)
    pipeline = build_pipeline(model, strategy="none").fit(X, y)
    explainer = LimeExplainer(pipeline, X, num_samples=500)
    return pipeline, explainer, X_test.head(5)


def test_dropped_and_leaky_columns_are_not_explained(setup):
    _, explainer, _ = setup
    assert "PageValues" not in explainer.columns
    assert "VisitorType" not in explainer.columns
    # Used through total_pages, so it stays.
    assert "Administrative" in explainer.columns


def test_encoding_round_trip_gives_the_same_prediction(setup):
    pipeline, explainer, sessions = setup
    session = sessions.head(1)

    encoded = explainer._encode(session).to_numpy(dtype=float)
    via_lime = explainer.predict_function(session)(encoded)[:, 1]
    direct = pipeline.predict_proba(session)[:, 1]
    assert via_lime == pytest.approx(direct)


def test_explanation_shape(setup):
    _, explainer, sessions = setup
    result = explainer.explain(sessions.head(1), n=4)

    assert len(result["conditions"]) == 4
    assert result["local_fit_r2"] <= 1
    for item in result["conditions"]:
        assert item["effect"] == ("increases" if item["weight"] > 0 else "decreases")


def test_same_session_gives_the_same_explanation(setup):
    _, explainer, sessions = setup
    first = explainer.explain(sessions.head(1))
    second = explainer.explain(sessions.head(1))
    assert first == second


def test_predict_function_returns_two_class_probabilities(setup):
    _, explainer, sessions = setup
    rows = explainer._encode(sessions).to_numpy(dtype=float)
    probabilities = explainer.predict_function(sessions.head(1))(rows)

    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities.sum(axis=1), 1)
