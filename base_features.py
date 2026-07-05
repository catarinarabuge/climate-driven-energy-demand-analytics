import pandas as pd
import os
import logging

# ------------------------------------------------------------
# BASE FEATURE ENGINEERING
# Cumpre o ponto 3.2.3 do enunciado:
# - Temporal features: hora, dia da semana
# - Seasonal indicator: estação do ano
# - Lagged demand features: procura anterior (1h e 24h)
# - Rolling climate features: médias móveis de temperatura
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona variáveis temporais e um indicador sazonal baseado nas datas reais das estações."""
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Define a estação conforme a data (mudança real a 21 de cada estação)
    def get_season(row):
        m, d = row["month"], row["day"]

        # Primavera: 21 Mar – 20 Jun
        if (m == 3 and d >= 21) or (m in [4, 5]) or (m == 6 and d < 21):
            return "spring"
        # Verão: 21 Jun – 20 Set
        elif (m == 6 and d >= 21) or (m in [7, 8]) or (m == 9 and d < 21):
            return "summer"
        # Outono: 21 Set – 20 Dez
        elif (m == 9 and d >= 21) or (m in [10, 11]) or (m == 12 and d < 21):
            return "autumn"
        # Inverno: 21 Dez – 20 Mar
        else:
            return "winter"

    df["season"] = df.apply(get_season, axis=1)
    return df


def create_lag_features(
        df: pd.DataFrame,
        target_col="load_mw") -> pd.DataFrame:
    """Cria lags de 1h e 24h da carga elétrica."""
    df[f"{target_col}_lag_1h"] = df[target_col].shift(1)
    df[f"{target_col}_lag_24h"] = df[target_col].shift(24)
    return df


def create_rolling_features(df: pd.DataFrame,
                            temp_col="2m_temperature") -> pd.DataFrame:
    """Cria médias móveis de 3h e 24h da temperatura."""
    df[f"{temp_col}_roll3h"] = df[temp_col].rolling(
        window=3, min_periods=1).mean()
    df[f"{temp_col}_roll24h"] = df[temp_col].rolling(
        window=24, min_periods=1).mean()
    return df


def generate_features(
    input_path="data/processed/merged.csv",
    output_path="data/processed/features_v1.csv"
) -> None:
    """Pipeline principal de criação das features base."""
    if not os.path.exists(input_path):
        logging.error(f"Arquivo não encontrado: {input_path}")
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    logging.info("A criar features temporais e sazonais...")
    df = create_temporal_features(df)

    logging.info("A criar lags e médias móveis...")
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # Remove as primeiras 24h (NaN criados pelos lags)
    df = df.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"✅ Features base gravadas em {output_path}")


if __name__ == "__main__":
    generate_features()
