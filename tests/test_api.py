from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from xgboost import XGBClassifier

from src.api import scoring
from src.api.main import create_app
from src.api.schemas import EXAMPLE_SESSION, Session
from src.api.scoring import confidence_label
from src.config.settings import get_settings
from src.data.loader import load_data, split_data
from src.models.artifacts import save_artifact
from src.preprocessing.pipeline import build_pipeline

settings = get_settings()


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    """Save a small model in a temporary folder, as the real one is not in the repo."""
    folder = tmp_path_factory.mktemp("api")
    X_train, _, y_train, _ = split_data(load_data())
    model = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=1)
    pipeline = build_pipeline(model, strategy="none").fit(X_train, y_train)
    save_artifact(
        pipeline,
        name="purchase_intent_v1",
        models_dir=folder,
        threshold=0.27,
        feature_columns=settings.input_columns,
        metrics={},
    )
    return folder


@pytest.fixture(scope="module")
def client(models_dir):
    app = create_app(replace(settings, models_dir=models_dir, logs_dir=models_dir))
    with TestClient(app) as test_client:
        yield test_client


def session(**changes) -> dict:
    return {**EXAMPLE_SESSION, **changes}


def test_schema_matches_config():
    assert list(Session.model_fields) == settings.input_columns


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "purchase_intent_v1"}


def test_predict_returns_everything(client):
    response = client.post("/predict", json=session())
    assert response.status_code == 200

    body = response.json()
    assert 0 <= body["conversion_probability"] <= 1
    assert body["will_convert"] == (body["conversion_probability"] >= body["threshold"])
    assert body["threshold"] == 0.27
    assert body["confidence"] in {"low", "medium", "high"}
    assert len(body["top_contributors"]) == 5
    assert "PageValues" not in [item["feature"] for item in body["top_contributors"]]


def test_custom_threshold_is_used(client):
    body = client.post("/predict", params={"threshold": 0.01}, json=session()).json()
    assert body["threshold"] == 0.01
    assert body["will_convert"] is True


@pytest.mark.parametrize("threshold", [0, 1, 1.5, -0.2])
def test_threshold_out_of_range_is_rejected(client, threshold):
    response = client.post("/predict", params={"threshold": threshold}, json=session())
    assert response.status_code == 422


@pytest.mark.parametrize(
    "bad_input",
    [
        {"ProductRelated": -3},
        {"ExitRates": 1.5},
        {"ProductRelated": "lots"},
        {"Month": ""},
        {"Weekend": "maybe"},
    ],
)
def test_invalid_values_are_rejected(client, bad_input):
    response = client.post("/predict", json=session(**bad_input))
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = session()
    del incomplete["ExitRates"]

    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    assert "ExitRates" in response.text


def test_unknown_categories_are_handled(client):
    response = client.post("/predict", json=session(Month="Jan", Browser=99, Region=42))
    assert response.status_code == 200


def test_page_values_is_ignored(client):
    without = client.post("/predict", json=session()).json()
    with_it = client.post("/predict", json=session(PageValues=300.0)).json()
    assert with_it["conversion_probability"] == without["conversion_probability"]


def test_docs_are_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


def test_model_loads_once(models_dir, monkeypatch):
    calls = []
    original_load = scoring.Scorer.load

    def counting_load(*args, **kwargs):
        calls.append(1)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(scoring.Scorer, "load", counting_load)
    app = create_app(replace(settings, models_dir=models_dir, logs_dir=models_dir))
    with TestClient(app) as fresh:
        for _ in range(3):
            assert fresh.post("/predict", json=session()).status_code == 200

    assert len(calls) == 1


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.30, "low"),
        (0.20, "low"),
        (0.45, "medium"),
        (0.10, "medium"),
        (0.90, "high"),
        (0.01, "high"),
    ],
)
def test_confidence_label(probability, expected):
    assert confidence_label(probability, threshold=0.27) == expected
