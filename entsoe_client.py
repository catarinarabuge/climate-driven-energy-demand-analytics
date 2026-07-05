import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient


# --------------------------------------------------------------------
# Paths do projeto
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ENERGY_DIR = PROJECT_ROOT / "data" / "raw" / "energy"
RAW_ENERGY_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_ENERGY_FILE = RAW_ENERGY_DIR / "entsoe_load_PT_2025_hourly.csv"
PROCESSED_FILE = PROCESSED_DIR / "energy_load_PT_2025_hourly_clean.csv"


# --------------------------------------------------------------------
# API key + cliente ENTSO-E
# --------------------------------------------------------------------
def load_api_key() -> str:
    """
    Lê a ENTSOE_API_KEY do .env ou das variáveis de ambiente.
    """
    load_dotenv()
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ENTSOE_API_KEY não definida. "
            "Cria um ficheiro .env na raiz com ENTSOE_API_KEY=... "
            "ou define a variável de ambiente."
        )
    return api_key


def build_client() -> EntsoePandasClient:
    """
    Constrói o cliente entsoe-py com a tua API key.
    """
    api_key = load_api_key()
    client = EntsoePandasClient(api_key=api_key)
    return client


# --------------------------------------------------------------------
# 1) INGESTION RAW (ENTSO-E -> data/raw/energy)
# --------------------------------------------------------------------
def fetch_pt_load_2025_hourly(client: EntsoePandasClient) -> pd.DataFrame:
    """
    Usa entsoe-py para obter a carga hora a hora de Portugal (PT) em 2025.
    O entsoe-py trata internamente dos endpoints e domains correctos.[web:21][web:63]
    """
    # Timezone oficial ENTSO-E
    start = pd.Timestamp("2025-01-01T00:00:00", tz="Europe/Brussels")
    end = pd.Timestamp("2026-01-01T00:00:00", tz="Europe/Brussels")

    # PT mapeado internamente para o domínio/bidding zone correcto.[web:63]
    df = client.query_load("PT", start=start, end=end)
    return df


def save_raw_energy(df: pd.DataFrame) -> Path:
    """
    Garante índice datetime e grava CSV raw (datetime + load_mw).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.index.name = "datetime"

    if df.shape[1] == 1:
        df.columns = ["load_mw"]

    RAW_ENERGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_ENERGY_FILE)
    print(f">>> RAW guardado em: {RAW_ENERGY_FILE}")
    return RAW_ENERGY_FILE


def run_ingestion_raw():
    """
    Pipeline de ingestion RAW:
    - constrói cliente
    - descarrega 2025 hora a hora para PT
    - grava CSV raw em data/raw/energy
    """
    print(">>> [INGESTION] A construir cliente ENTSO-E (entsoe-py)...")
    client = build_client()

    print(">>> [INGESTION] A obter carga hora a hora de Portugal (PT) para 2025...")
    df = fetch_pt_load_2025_hourly(client)

    print(f">>> [INGESTION] Registos descarregados: {len(df)}")
    save_raw_energy(df)
    print(">>> [INGESTION] Concluído.")


# --------------------------------------------------------------------
# 2) CLEANING / ALIGNMENT -> data/processed (UTC, +00:00)
# --------------------------------------------------------------------
def load_raw_energy() -> pd.DataFrame:
    """
    Lê o CSV bruto do ENTSO-E gerado por run_ingestion_raw().
    """
    if not RAW_ENERGY_FILE.exists():
        raise FileNotFoundError(
            f"Ficheiro raw não encontrado: {RAW_ENERGY_FILE}. "
            "Garante que correste primeiro o run_ingestion_raw()."
        )

    df = pd.read_csv(RAW_ENERGY_FILE)
    return df


def clean_and_align_energy(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        raise ValueError("Input dataframe is empty")

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.drop(columns=["datetime"])
    else:
        dt = pd.to_datetime(df.index, utc=True, errors="coerce")

    dt = pd.DatetimeIndex(dt).tz_convert("UTC")
    df.index = dt
    df.index.name = "datetime"

    if df.shape[1] == 1:
        df.columns = ["load_mw"]

    # 🔴 FIX PRINCIPAL
    df = df[~df.index.duplicated(keep="first")]

    df = df.sort_index()

    full_index = pd.date_range(
        start=pd.Timestamp("2025-01-01T00:00:00", tz="UTC"),
        end=pd.Timestamp("2026-01-01T00:00:00", tz="UTC"),
        freq="h",
        inclusive="left",
    )

    df = df.reindex(full_index)

    df["load_mw"] = df["load_mw"].ffill().bfill()

    return df


def save_processed_energy(df: pd.DataFrame) -> Path:
    """
    Guarda o dataset processado em data/processed/energy_load_PT_2025_hourly_clean.csv.
    """
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE)
    print(f">>> [CLEANING] Dados processados guardados em: {PROCESSED_FILE}")
    return PROCESSED_FILE


def run_processing():
    """
    Pipeline de PROCESSING:
    - lê raw
    - limpa/alinha em UTC
    - grava em data/processed
    """
    print(">>> [CLEANING] A carregar dados raw de energia (ENTSO-E)...")
    df_raw = load_raw_energy()

    print(">>> [CLEANING] A limpar e alinhar série horária de 2025 (UTC)...")
    df_clean = clean_and_align_energy(df_raw)

    print(">>> [CLEANING] A guardar dados processados...")
    save_processed_energy(df_clean)

    print(">>> [CLEANING] Concluído.")


# --------------------------------------------------------------------
# 3) Pipeline completo (raw + processed)
# --------------------------------------------------------------------
def run_full_pipeline():
    """
    Corre ingestion RAW e depois PROCESSING no mesmo comando.
    """
    run_ingestion_raw()
    run_processing()


if __name__ == "__main__":
    # Pipeline completo (raw + processed):
    run_full_pipeline()
