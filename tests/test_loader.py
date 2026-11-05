import pandas as pd
import pytest

from src.config.settings import get_settings
from src.data.loader import DataValidationError, load_data, split_data

settings = get_settings()


def test_load_data_returns_expected_columns():
    df = load_data()
    assert len(df) > 0
    for col in settings.feature_columns + [settings.target]:
        assert col in df.columns


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "does_not_exist.csv")


def test_missing_column_raises(tmp_path):
    df = load_data().drop(columns=["PageValues"])
    bad_file = tmp_path / "bad.csv"
    df.to_csv(bad_file, index=False)

    with pytest.raises(DataValidationError, match="PageValues"):
        load_data(bad_file)


def test_missing_values_raise(tmp_path):
    df = load_data()
    df.loc[0, "ExitRates"] = None
    bad_file = tmp_path / "bad.csv"
    df.to_csv(bad_file, index=False)

    with pytest.raises(DataValidationError, match="Missing values"):
        load_data(bad_file)


def test_split_is_reproducible_and_stratified():
    df = load_data()
    X_train_1, X_test_1, y_train_1, _ = split_data(df)
    X_train_2, X_test_2, _, _ = split_data(df)

    pd.testing.assert_index_equal(X_train_1.index, X_train_2.index)
    pd.testing.assert_index_equal(X_test_1.index, X_test_2.index)
    assert y_train_1.mean() == pytest.approx(df[settings.target].mean(), abs=0.01)
