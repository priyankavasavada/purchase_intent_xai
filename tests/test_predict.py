import pytest
from sklearn.linear_model import LogisticRegression

from src.config.settings import get_settings
from src.data.loader import load_data, split_data
from src.models.artifacts import load_artifact, save_artifact
from src.models.predict import predict_sessions
from src.preprocessing.pipeline import build_pipeline

settings = get_settings()


@pytest.fixture(scope="module")
def saved_model_dir(tmp_path_factory):
    """Train a small model once and save it to a temporary folder."""
    models_dir = tmp_path_factory.mktemp("models")
    X_train, _, y_train, _ = split_data(load_data())

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)
    save_artifact(
        pipeline,
        name="test_model",
        models_dir=models_dir,
        threshold=0.5,
        feature_columns=settings.feature_columns,
        metrics={},
    )
    return models_dir


def test_saved_model_loads_and_predicts(saved_model_dir):
    artifact = load_artifact("test_model", saved_model_dir)
    sessions = load_data().head(5)

    result = predict_sessions(sessions, artifact)

    assert len(result) == 5
    assert result["conversion_probability"].between(0, 1).all()
    assert result["will_convert"].dtype == bool


def test_predict_rejects_missing_columns(saved_model_dir):
    artifact = load_artifact("test_model", saved_model_dir)
    sessions = load_data().head(1).drop(columns=["Month"])

    with pytest.raises(ValueError, match="Month"):
        predict_sessions(sessions, artifact)


def test_loading_unknown_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact("no_such_model", tmp_path)


def test_versions_count_up(tmp_path):
    from src.models.artifacts import latest_version_name, next_version_name

    assert next_version_name(tmp_path, "purchase_intent") == "purchase_intent_v1"
    for version in (1, 2):
        (tmp_path / f"purchase_intent_v{version}.joblib").touch()
    # A different model with a similar name must not be counted.
    (tmp_path / "purchase_intent_with_page_values_v7.joblib").touch()

    assert next_version_name(tmp_path, "purchase_intent") == "purchase_intent_v3"
    assert latest_version_name(tmp_path, "purchase_intent") == "purchase_intent_v2"


def test_latest_version_without_models_raises(tmp_path):
    from src.models.artifacts import latest_version_name

    with pytest.raises(FileNotFoundError):
        latest_version_name(tmp_path, "purchase_intent")
