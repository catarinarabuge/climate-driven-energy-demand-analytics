import pandas as pd
import pytest

from Code.features.base_features import (
    create_temporal_features,
    create_lag_features,
    create_rolling_features,
)


def sample_df():
    dates = pd.date_range("2025-01-01 00:00:00", periods=48, freq="h")

    return pd.DataFrame({
        "timestamp": dates,
        "load_mw": range(5000, 5048),
        "2m_temperature": [10 + i * 0.1 for i in range(48)],
    })


def test_temporal_features():
    df = sample_df()
    result = create_temporal_features(df)

    assert "hour" in result.columns, "Should create hour"
    assert "day_of_week" in result.columns, "Should create day_of_week"
    assert "season" in result.columns, "Should create season"
    assert "month" in result.columns, "Should create month"
    assert "is_weekend" in result.columns, "Should create is_weekend"


def test_temporal_feature_values_first_row():
    df = sample_df()
    result = create_temporal_features(df)

    assert result.loc[0, "hour"] == 0, "First timestamp hour should be 0"
    assert result.loc[0, "month"] == 1, "January should map to month 1"
    assert result.loc[0, "day_of_week"] == 2, "2025-01-01 is Wednesday (2)"


def test_is_weekend_false_on_weekday():
    df = sample_df()
    result = create_temporal_features(df)

    assert result.loc[0, "is_weekend"] == 0, "Wednesday should not be weekend"


def test_is_weekend_true():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-04 10:00:00"]),  # Saturday
        "load_mw": [5000],
        "2m_temperature": [10.0],
    })

    result = create_temporal_features(df)

    assert result.loc[0, "is_weekend"] == 1, "Saturday should be weekend"


def test_lag_features():
    df = sample_df()
    result = create_lag_features(df)

    assert "load_mw_lag_1h" in result.columns, "Should create 1h lag"
    assert "load_mw_lag_24h" in result.columns, "Should create 24h lag"


def test_lag_1h_values():
    df = sample_df()
    result = create_lag_features(df)

    assert pd.isna(result.loc[0, "load_mw_lag_1h"]
                   ), "First 1h lag should be NaN"
    assert result.loc[1, "load_mw_lag_1h"] == result.loc[0,
                                                         "load_mw"], "1h lag should match previous row"


def test_lag_24h_values():
    df = sample_df()
    result = create_lag_features(df)

    assert pd.isna(result.loc[0, "load_mw_lag_24h"]
                   ), "First 24h lag should be NaN"
    assert result.loc[24, "load_mw_lag_24h"] == result.loc[0,
                                                           "load_mw"], "24h lag should match value 24 rows before"


def test_rolling_features():
    df = sample_df()
    result = create_rolling_features(df)

    assert "2m_temperature_roll3h" in result.columns, "Should create 3h rolling mean"
    assert "2m_temperature_roll24h" in result.columns, "Should create 24h rolling mean"


def test_rolling_3h_values():
    df = sample_df()
    result = create_rolling_features(df)

    expected = (10.0 + 10.1 + 10.2) / 3
    assert round(result.loc[2, "2m_temperature_roll3h"], 5) == round(
        expected, 5), "3h rolling mean should be correct"


def test_hour_range():
    df = sample_df()
    result = create_temporal_features(df)

    assert result["hour"].between(
        0, 23).all(), "Hour should be between 0 and 23"


def test_day_of_week_range():
    df = sample_df()
    result = create_temporal_features(df)

    assert result["day_of_week"].between(
        0, 6).all(), "Day of week should be between 0 and 6"


def test_missing_timestamp():
    df = sample_df().drop(columns=["timestamp"])

    with pytest.raises(KeyError):
        create_temporal_features(df)


def test_missing_load_for_lag():
    df = sample_df().drop(columns=["load_mw"])

    with pytest.raises(KeyError):
        create_lag_features(df)


def test_missing_temperature_for_rolling():
    df = sample_df().drop(columns=["2m_temperature"])

    with pytest.raises(KeyError):
        create_rolling_features(df)


def test_empty_dataframe_temporal():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([]),
        "load_mw": pd.Series(dtype=float),
        "2m_temperature": pd.Series(dtype=float),
    })
    result = create_temporal_features(df)

    assert result.empty


def test_empty_dataframe_lag():
    df = pd.DataFrame(columns=["timestamp", "load_mw", "2m_temperature"])
    result = create_lag_features(df)

    assert result.empty, "Lag features on empty dataframe should return empty dataframe"


def test_empty_dataframe_rolling():
    df = pd.DataFrame(columns=["timestamp", "load_mw", "2m_temperature"])
    result = create_rolling_features(df)

    assert result.empty, "Rolling features on empty dataframe should return empty dataframe"
