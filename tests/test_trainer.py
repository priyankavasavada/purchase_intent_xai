from dataclasses import replace

import pytest

from src.config.settings import get_settings
from src.data.loader import load_data, split_data
from src.models.trainer import ModelTrainer, get_models
from src.preprocessing.pipeline import IMBALANCE_STRATEGIES

settings = replace(get_settings(), cv_folds=2)
MODELS = get_models(settings.random_seed)


@pytest.fixture(scope="module")
def sample():
    X_train, _, y_train, _ = split_data(load_data())
    return X_train.head(600), y_train.head(600)


def test_all_six_models_are_available():
    assert set(MODELS) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    }


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("strategy", IMBALANCE_STRATEGIES)
def test_every_model_builds_with_every_strategy(sample, name, strategy):
    _, y = sample
    pipeline = ModelTrainer(name, MODELS[name], strategy, settings).build(y)
    assert ("smote" in pipeline.named_steps) == (strategy == "smote")


@pytest.mark.parametrize("name", MODELS)
def test_every_model_cross_validates_and_predicts(sample, name):
    X, y = sample
    trainer = ModelTrainer(name, MODELS[name], "class_weight", settings)

    scores = trainer.cross_validate(X, y)
    assert 0 <= scores["pr_auc_mean"] <= 1

    probabilities = trainer.fit(X, y).predict_proba(X.head(5))[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_xgboost_gets_the_class_ratio(sample):
    _, y = sample
    pipeline = ModelTrainer("xgboost", MODELS["xgboost"], "class_weight", settings).build(y)
    expected = (y == 0).sum() / (y == 1).sum()
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_class_weight_does_not_change_the_shared_model(sample):
    _, y = sample
    ModelTrainer("lightgbm", MODELS["lightgbm"], "class_weight", settings).build(y)
    assert MODELS["lightgbm"].get_params()["class_weight"] is None


def test_tuning_returns_params_and_scores(sample):
    X, y = sample
    result = ModelTrainer("decision_tree", MODELS["decision_tree"], "none", settings).tune(
        X, y, n_iter=2
    )

    assert set(result["params"]) == {"max_depth", "min_samples_leaf"}
    assert 0 <= result["scores"]["pr_auc_mean"] <= 1


def test_every_model_has_a_search_space():
    from src.models.trainer import PARAM_SPACES

    assert set(PARAM_SPACES) == set(MODELS)
