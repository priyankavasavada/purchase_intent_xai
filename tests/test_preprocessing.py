import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import load_data
from src.preprocessing.pipeline import build_pipeline, build_preprocessor


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_preprocessor_output_has_no_nans(data):
    output = build_preprocessor().fit_transform(data)
    assert not np.isnan(output).any()


def test_int_coded_ids_are_one_hot_encoded(data):
    preprocessor = build_preprocessor().fit(data)
    names = list(preprocessor.get_feature_names_out())

    # Region should become one column per region, not a single scaled number.
    assert "categorical__Region_3" in names
    assert "scaled__Region" not in names


def test_unknown_category_does_not_crash(data):
    preprocessor = build_preprocessor().fit(data)
    new_session = data.head(1).copy()
    new_session["Month"] = "Jan"  # never seen in training
    new_session["Browser"] = 99

    output = preprocessor.transform(new_session)
    assert output.shape[0] == 1


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown imbalance strategy"):
        build_pipeline(LogisticRegression(), strategy="oversample_everything")


def test_class_weight_is_set_on_a_copy():
    model = DecisionTreeClassifier()
    pipeline = build_pipeline(model, strategy="class_weight")

    assert pipeline.named_steps["model"].class_weight == "balanced"
    assert model.class_weight is None


def test_smote_step_is_added():
    pipeline = build_pipeline(LogisticRegression(), strategy="smote")
    assert "smote" in pipeline.named_steps
