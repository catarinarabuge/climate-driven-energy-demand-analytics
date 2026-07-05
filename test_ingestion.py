import pandas as pd
import pytest

import Code.ingestion.entsoe_client as entsoe_client
from Code.ingestion.entsoe_client import load_api_key


def test_load_api_key_success(monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", "fake_key_123")

    result = entsoe_client.load_api_key()

    assert result == "fake_key_123"


def test_load_api_key_missing(monkeypatch):
    # impede que load_dotenv carregue o .env durante o teste
    monkeypatch.setattr(
        "Code.ingestion.entsoe_client.load_dotenv",
        lambda: None)
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        load_api_key()


def test_save_raw_energy_creates_file(tmp_path, monkeypatch):
    test_file = tmp_path / "entsoe_raw.csv"
    monkeypatch.setattr(entsoe_client, "RAW_ENERGY_FILE", test_file)

    df = pd.DataFrame(
        {"load_mw": [5000, 5100]},
        index=pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]),
    )

    result = entsoe_client.save_raw_energy(df)

    assert result == test_file
    assert test_file.exists()

    saved = pd.read_csv(test_file)
    assert "datetime" in saved.columns
    assert "load_mw" in saved.columns


def test_save_raw_energy_renames_single_column(tmp_path, monkeypatch):
    test_file = tmp_path / "entsoe_raw.csv"
    monkeypatch.setattr(entsoe_client, "RAW_ENERGY_FILE", test_file)

    df = pd.DataFrame(
        {"value": [5000, 5100]},
        index=pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]),
    )

    entsoe_client.save_raw_energy(df)

    saved = pd.read_csv(test_file)
    assert "load_mw" in saved.columns


def test_load_raw_energy_success(tmp_path, monkeypatch):
    test_file = tmp_path / "entsoe_raw.csv"
    monkeypatch.setattr(entsoe_client, "RAW_ENERGY_FILE", test_file)

    df = pd.DataFrame(
        {
            "datetime": ["2025-01-01 00:00:00"],
            "load_mw": [5000],
        }
    )
    df.to_csv(test_file, index=False)

    result = entsoe_client.load_raw_energy()

    assert not result.empty
    assert "datetime" in result.columns
    assert "load_mw" in result.columns


def test_load_raw_energy_missing_file(tmp_path, monkeypatch):
    test_file = tmp_path / "missing.csv"
    monkeypatch.setattr(entsoe_client, "RAW_ENERGY_FILE", test_file)

    with pytest.raises(FileNotFoundError):
        entsoe_client.load_raw_energy()


def test_build_client_uses_api_key(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setenv("ENTSOE_API_KEY", "fake_key_123")
    monkeypatch.setattr(entsoe_client, "EntsoePandasClient", FakeClient)

    client = entsoe_client.build_client()

    assert client.api_key == "fake_key_123"


def test_clean_align_without_datetime_column():
    df = pd.DataFrame(
        {"load_mw": [5000, 5100]},
        index=pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]),
    )

    result = entsoe_client.clean_and_align_energy(df)

    assert not result.empty
    assert isinstance(result.index, pd.DatetimeIndex)


def test_save_processed_energy_creates_file(tmp_path, monkeypatch):
    test_file = tmp_path / "processed.csv"
    monkeypatch.setattr(entsoe_client, "PROCESSED_FILE", test_file)

    df = pd.DataFrame({"load_mw": [5000, 5100]})

    result = entsoe_client.save_processed_energy(df)

    assert result == test_file
    assert test_file.exists()


def test_run_processing(monkeypatch):
    monkeypatch.setattr(
        entsoe_client,
        "load_raw_energy",
        lambda: pd.DataFrame(
            {
                "datetime": ["2025-01-01 00:00:00"],
                "load_mw": [5000],
            }
        ),
    )

    monkeypatch.setattr(
        entsoe_client,
        "save_processed_energy",
        lambda df: "ok")

    entsoe_client.run_processing()
