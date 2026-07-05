import os
import pandas as pd
from pathlib import Path

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    cdsapi = None
    CDSAPI_AVAILABLE = False

try:
    import cfgrib
    CFGRIB_AVAILABLE = True
except ImportError:
    cfgrib = None
    CFGRIB_AVAILABLE = False


def download_era5(raw_dir: str) -> str:
    raw_path = Path(raw_dir) / "era5_portugal_2025.grib"

    if raw_path.exists():
        print(f"Ficheiro já existe, a saltar download: {raw_path}")
        return str(raw_path)

    if not CDSAPI_AVAILABLE:
        raise ImportError(
            "cdsapi nao esta instalado. "
            "Instala com: pip install cdsapi"
        )

    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature",
                "10m_v_component_of_wind",
                "surface_solar_radiation_downwards",
                "total_precipitation",
            ],
            "year": ["2025"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [42, -10, 36, -6],  # Portugal: N,W,S,E
            "data_format": "grib",
        },
        str(raw_path),
    )
    print(f"Download concluído: {raw_path}")
    return str(raw_path)


def ingest_era5(grib_path: str, output_dir: str) -> None:
    if not os.path.exists(grib_path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {grib_path}")

    if not CFGRIB_AVAILABLE:
        raise ImportError(
            "cfgrib nao esta instalado. "
            "Instala com: pip install cfgrib"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datasets = cfgrib.open_datasets(grib_path)
    print(f"{len(datasets)} datasets encontrados")
    for i, ds in enumerate(datasets):
        print(f"  Dataset {i}: {list(ds.data_vars)}")

    ds_instant = datasets[0][['t2m', 'v10']]
    portugal_mask = (
        (ds_instant.latitude >= 36) & (ds_instant.latitude <= 44) &
        (ds_instant.longitude >= -10) & (ds_instant.longitude <= -6)
    )
    df_instant = (
        ds_instant.where(portugal_mask, drop=True)
        .mean(['latitude', 'longitude'])
        .to_dataframe()
        .reset_index()[['time', 't2m', 'v10']]
    )
    df_instant['time'] = pd.to_datetime(
        df_instant['time']).dt.tz_localize(None)

    ds_accum = datasets[1][['ssrd', 'tp']]
    portugal_mask_a = (
        (ds_accum.latitude >= 36) & (ds_accum.latitude <= 44) &
        (ds_accum.longitude >= -10) & (ds_accum.longitude <= -6)
    )
    df_accum = (
        ds_accum.where(portugal_mask_a, drop=True)
        .mean(['latitude', 'longitude'])
        .to_dataframe()
        .reset_index()
    )

    time_col = 'valid_time' if 'valid_time' in df_accum.columns else 'time'
    df_accum = df_accum[[time_col, 'ssrd', 'tp']].rename(
        columns={time_col: 'time'})
    df_accum['time'] = pd.to_datetime(df_accum['time']).dt.tz_localize(None)

    df_instant['2m_temperature'] = df_instant['t2m'] - 273.15
    df_instant['10m_v_component_of_wind'] = df_instant['v10'].abs()
    df_accum['surface_solar_radiation_downwards'] = df_accum['ssrd'] / 3600
    df_accum['total_precipitation'] = df_accum['tp'] * 1000

    df = pd.merge_asof(
        df_instant[['time', '2m_temperature',
                    '10m_v_component_of_wind']].sort_values('time'),
        df_accum[['time', 'surface_solar_radiation_downwards',
                  'total_precipitation']].sort_values('time'),
        on='time', tolerance=pd.Timedelta('1h'), direction='nearest'
    )

    df['time'] = df['time'].dt.tz_localize(
        'UTC').dt.tz_convert('Europe/Lisbon')
    df = df.dropna(subset=['time']).drop_duplicates(
        'time').sort_values('time').reset_index(drop=True)

    output_path = Path(output_dir) / "era5_portugal_processed.csv"
    df.to_csv(output_path, index=False)

    print(f"\nERA5 processado: {output_path}")
    print(f"{df['time'].min()} → {df['time'].max()}")
    print(f"{len(df)} linhas | Colunas: {list(df.columns)}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    RAW_DIR = BASE_DIR / "data/raw/weather"
    OUTPUT_DIR = BASE_DIR / "data/processed"

    grib_path = download_era5(str(RAW_DIR))
    ingest_era5(grib_path, str(OUTPUT_DIR))
