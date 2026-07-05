import pandas as pd
import os


def merge_datasets(
    energy_path="data/processed/clean_energy.csv",
    weather_path="data/processed/clean_weather.csv",
    output_path="data/processed/merged.csv"
):
    # Verificar existência dos ficheiros
    if not os.path.exists(energy_path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {energy_path}")
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {weather_path}")

    # Ler dados
    energy = pd.read_csv(energy_path)
    weather = pd.read_csv(weather_path)

    # Identificar colunas temporais
    time_col_e = "timestamp" if "timestamp" in energy.columns else energy.columns[0]
    time_col_w = "timestamp" if "timestamp" in weather.columns else weather.columns[0]

    # Converter e normalizar datetimes (remove fuso horário)
    energy[time_col_e] = pd.to_datetime(
        energy[time_col_e], utc=True).dt.tz_localize(None)
    weather[time_col_w] = pd.to_datetime(
        weather[time_col_w], utc=True).dt.tz_localize(None)

    # Ordenar e juntar com tolerância de 1h
    merged = pd.merge_asof(
        energy.sort_values(time_col_e),
        weather.sort_values(time_col_w),
        left_on=time_col_e,
        right_on=time_col_w,
        direction="nearest",
        tolerance=pd.Timedelta("1h")  # minúsculo 'h' para evitar FutureWarning
    )

    # Renomear e limpar duplicados
    merged = merged.rename(columns={time_col_e: "timestamp"})
    if time_col_w in merged.columns and time_col_w != time_col_e:
        merged = merged.drop(columns=[time_col_w])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"✅ Ficheiro unido guardado em {output_path}")
    print(f"Linhas: {merged.shape[0]} | Colunas: {merged.shape[1]}")


if __name__ == "__main__":
    merge_datasets()
