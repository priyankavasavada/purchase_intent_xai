import pandas as pd
import pytest

from src.data.loader import load_data
from src.features.engineering import add_behavioural_features
from src.preprocessing.pipeline import original_column


def make_session(**overrides) -> pd.DataFrame:
    session = {
        "Administrative": 1,
        "Administrative_Duration": 10.0,
        "Informational": 1,
        "Informational_Duration": 10.0,
        "ProductRelated": 2,
        "ProductRelated_Duration": 80.0,
    }
    session.update(overrides)
    return pd.DataFrame([session])


def test_feature_values_are_correct():
    row = add_behavioural_features(make_session()).iloc[0]

    assert row["total_pages"] == 4
    assert row["total_duration"] == 100
    assert row["avg_time_per_page"] == 25
    assert row["avg_time_per_product_page"] == 40
    assert row["product_page_share"] == 0.5
    assert row["product_time_share"] == 0.8


def test_session_with_no_activity_gives_zeros_not_nan():
    empty = make_session(
        Administrative=0,
        Administrative_Duration=0.0,
        Informational=0,
        Informational_Duration=0.0,
        ProductRelated=0,
        ProductRelated_Duration=0.0,
    )
    row = add_behavioural_features(empty).iloc[0]

    assert row["avg_time_per_page"] == 0
    assert row["product_page_share"] == 0
    assert row["product_time_share"] == 0


def test_original_data_is_not_changed():
    sessions = make_session()
    add_behavioural_features(sessions)
    assert "total_pages" not in sessions.columns


def test_no_missing_or_negative_values_on_real_data():
    engineered = add_behavioural_features(load_data())
    new_columns = ["avg_time_per_page", "product_page_share", "product_time_share"]

    assert not engineered[new_columns].isna().any().any()
    assert (engineered[new_columns] >= 0).all().all()
    assert engineered["product_time_share"].max() == pytest.approx(1.0)


def test_encoded_names_map_back_to_their_column():
    categorical = ["Month", "Region"]

    assert original_column("categorical__Month_Nov", categorical) == "Month"
    assert original_column("categorical__Region_3", categorical) == "Region"
    assert original_column("log_scaled__PageValues", categorical) == "PageValues"
