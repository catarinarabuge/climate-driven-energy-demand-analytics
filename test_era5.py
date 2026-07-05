from pathlib import Path

import pandas as pd
import pytest

import Code.ingestion.era5 as era5


def test_download_era5_skips_if_file_exists(tmp_path):
    raw_file = tmp_path / "era5_portugal_2025.grib"
    raw_file.write_text("dummy")

    result = era5.download_era5(str(tmp_path))

    assert result == str(raw_file)
    assert raw_file.exists()


def test_ingest_era5_raises_if_file_missing(tmp_path):
    missing_file = tmp_path / "does_not_exist.grib"
    output_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError):
        era5.ingest_era5(str(missing_file), str(output_dir))


def test_download_era5_without_cdsapi(monkeypatch, tmp_path):
    monkeypatch.setattr(era5, "CDSAPI_AVAILABLE", False)

    with pytest.raises(ImportError):
        era5.download_era5(str(tmp_path))


def test_ingest_era5_without_cfgrib(monkeypatch, tmp_path):
    file = tmp_path / "file.grib"
    file.write_text("dummy")

    monkeypatch.setattr(era5, "CFGRIB_AVAILABLE", False)

    with pytest.raises(ImportError):
        era5.ingest_era5(str(file), str(tmp_path))


def test_download_era5_with_fake_client(monkeypatch, tmp_path):
    class FakeClient:
        def retrieve(self, dataset, request, target):
            assert dataset == "reanalysis-era5-single-levels"
            assert "2m_temperature" in request["variable"]
            Path(target).write_text("fake grib")

    fake_cdsapi = type(
        "FakeCdsapi",
        (),
        {
            "Client": lambda: FakeClient(),
        },
    )

    monkeypatch.setattr(era5, "CDSAPI_AVAILABLE", True)
    monkeypatch.setattr(era5, "cdsapi", fake_cdsapi)

    result = era5.download_era5(str(tmp_path))

    assert Path(result).exists()


def test_ingest_era5_full_flow(monkeypatch, tmp_path):
    file = tmp_path / "file.grib"
    file.write_text("dummy")

    # Fake dataset estilo cfgrib
    class FakeDS:
        def __init__(self):
            self.latitude = pd.Series([40])
            self.longitude = pd.Series([-8])
            self.data_vars = ["t2m", "v10"]

        def __getitem__(self, keys):
            return self

        def where(self, *args, **kwargs):
            return self

        def mean(self, *args, **kwargs):
            return self

        def to_dataframe(self):
            return pd.DataFrame(
                {
                    "time": pd.date_range("2025-01-01", periods=3, freq="h"),
                    "t2m": [280, 281, 282],
                    "v10": [1, 2, 3],
                    "ssrd": [100, 200, 300],
                    "tp": [0.1, 0.2, 0.3],
                }
            )

    fake_cfgrib = type(
        "FakeCfgrib",
        (),
        {
            "open_datasets": lambda path: [FakeDS(), FakeDS()],
        },
    )

    monkeypatch.setattr(era5, "CFGRIB_AVAILABLE", True)
    monkeypatch.setattr(era5, "cfgrib", fake_cfgrib)

    era5.ingest_era5(str(file), str(tmp_path))

    output = tmp_path / "era5_portugal_processed.csv"
    assert output.exists()
