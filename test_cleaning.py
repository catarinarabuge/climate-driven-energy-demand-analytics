import pandas as pd
import pytest

from Code.ingestion.entsoe_client import clean_and_align_energy


def sample_raw_df():
    return pd.DataFrame({
        "datetime": [
            "2025-01-01 00:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 01:00:00",
            "2025-01-01 03:00:00",
        ],
        "load_mw": [5000, 5100, 5100, None],
    })


def test_removes_duplicates():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert result.index.is_unique


def test_fills_missing_values():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert result["load_mw"].isna().sum() == 0


def test_has_datetime_index():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert isinstance(result.index, pd.DatetimeIndex)


def test_raises_if_empty():
    df = pd.DataFrame(columns=["datetime", "load_mw"])

    with pytest.raises(Exception):
        clean_and_align_energy(df)


def test_index_is_sorted():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert result.index.is_monotonic_increasing


def test_missing_hour_is_created():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert pd.Timestamp("2025-01-01 02:00:00", tz="UTC") in result.index


def test_has_full_2025_hourly_index():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert len(result) == 8760


def test_index_start_and_end():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert result.index.min() == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    assert result.index.max() == pd.Timestamp("2025-12-31 23:00:00", tz="UTC")


def test_output_has_load_mw_column():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert "load_mw" in result.columns


def test_forward_fill_missing_value():
    df = sample_raw_df()
    result = clean_and_align_energy(df)

    assert result.loc[pd.Timestamp(
        "2025-01-01 03:00:00", tz="UTC"), "load_mw"] == 5100
