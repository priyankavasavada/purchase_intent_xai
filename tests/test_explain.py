import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data.loader import load_data, split_data
from src.explain.shap_explainer import ShapExplainer
from src.preprocessing.pipeline import build_pipeline


@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, _ = split_data(load_data())
    return X_train.head(1500), y_train.head(1500), X_test.head(50)


@pytest.fixture(scope="module")
def xgb_explainer(data):
    X, y, _ = data
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=1)
    pipeline = build_pipeline(model, strategy="none").fit(X, y)
    return pipeline, ShapExplainer(pipeline)


def test_contributions_add_up_to_the_model_score(data, xgb_explainer):
    _, _, sessions = data
    pipeline, explainer = xgb_explainer

    total = explainer.contributions(sessions).sum(axis=1) + explainer.base_value
    margin = pipeline.named_steps["model"].predict(
        pipeline[:-1].transform(sessions), output_margin=True
    )
    assert total.to_numpy() == pytest.approx(margin, abs=1e-4)


def test_contributions_use_original_column_names(data, xgb_explainer):
    _, _, sessions = data
    columns = list(xgb_explainer[1].contributions(sessions).columns)

    assert columns.count("Month") == 1
    assert "total_pages" in columns
    assert not any("__" in col for col in columns)
    assert "PageValues" not in columns


def test_top_contributors_are_sorted_by_size(data, xgb_explainer):
    _, _, sessions = data
    top = xgb_explainer[1].top_contributors(sessions.head(1), n=4)

    assert len(top) == 4
    sizes = [abs(item["contribution"]) for item in top]
    assert sizes == sorted(sizes, reverse=True)
    for item in top:
        assert item["effect"] == ("increases" if item["contribution"] > 0 else "decreases")


def test_top_contributors_show_the_raw_value(data, xgb_explainer):
    _, _, sessions = data
    session = sessions.head(1)
    top = xgb_explainer[1].top_contributors(session, n=20)
    month = next(item for item in top if item["feature"] == "Month")

    assert month["value"] == session["Month"].iloc[0]


def test_top_contributors_needs_one_session(data, xgb_explainer):
    _, _, sessions = data
    with pytest.raises(ValueError):
        xgb_explainer[1].top_contributors(sessions.head(2))


def test_logistic_regression_needs_background(data):
    X, y, sessions = data
    pipeline = build_pipeline(LogisticRegression(max_iter=1000), strategy="none").fit(X, y)

    with pytest.raises(ValueError, match="background"):
        ShapExplainer(pipeline)

    explainer = ShapExplainer(pipeline, background=X.head(200))
    assert len(explainer.global_importance(sessions)) > 0
