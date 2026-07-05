import pandas as pd
import os
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def add_temperature_anomaly(df, temp_col="2m_temperature"):
    """Desvio da temperatura comparado com a média mensal."""
    monthly_means = df.groupby(df["timestamp"].dt.month)[
        temp_col].transform("mean")
    df["temp_anomaly"] = df[temp_col] - monthly_means
    return df


def add_heatwave_indicator(df, temp_col="2m_temperature"):
    """Identifica períodos acima do percentil 90 durante 3 horas."""
    threshold = df[temp_col].quantile(0.90)
    df["is_hot"] = (df[temp_col] >= threshold).astype(int)

    # heatwave = 1 se >= 3h seguidas acima threshold
    df["heatwave"] = (
        df["is_hot"]
        .rolling(window=3, min_periods=1)
        .sum()
        .apply(lambda x: 1 if x >= 3 else 0)
    )
    df.drop(columns=["is_hot"], inplace=True)
    return df


def add_interaction_terms(df):
    """Cria interações não lineares entre clima."""

    df["temp_squared"] = df["2m_temperature"] ** 2

    solar_col = "surface_solar_radiation_downwards"
    if solar_col in df.columns:
        df["temp_x_solar"] = df["2m_temperature"] * df[solar_col]
    else:
        logging.warning(
            f"Coluna '{solar_col}' não encontrada. A ignorar temp_x_solar.")

    return df


def add_feels_like(df):
    """Temperatura sentida usando a componente v do vento como proxy."""

    df["feels_like_temp"] = df["2m_temperature"] - \
        0.7 * abs(df["10m_v_component_of_wind"])
    return df


def generate_advanced_features(
    input_path="data/processed/features_v1.csv",
    output_path="data/processed/features_v2_advanced.csv"
):
    start = time.time()

    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    logging.info("A calcular anomalia de temperatura…")
    df = add_temperature_anomaly(df)

    logging.info("A criar indicador de onda de calor…")
    df = add_heatwave_indicator(df)

    logging.info("A criar interações não lineares…")
    df = add_interaction_terms(df)

    logging.info("A calcular temperatura sentida…")
    df = add_feels_like(df)

    df.to_csv(output_path, index=False)
    logging.info(
        f"Advanced features guardadas em {output_path} ({time.time() - start:.2f}s)")


if __name__ == "__main__":
    generate_advanced_features()
