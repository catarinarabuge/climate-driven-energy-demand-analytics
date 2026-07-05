# Testing/test_model_training.py

import os
import pytest
import pandas as pd
import numpy as np

from Code.modeling.model_training import (
    split_time_series,
    encode_categorical_features,
    train_models,
)


# ──────────────────────────────────────────────
# Dataset sintetico para testes (sem precisar do CSV real)
# ──────────────────────────────────────────────

def make_synthetic_dataset(n=200):
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "load_mw": np.random.uniform(4000, 8000, n),
        "2m_temperature": np.random.uniform(5, 35, n),
        "10m_v_component_of_wind": np.random.uniform(-5, 5, n),
        "surface_solar_radiation_downwards": np.random.uniform(0, 800, n),
        "total_precipitation": np.random.uniform(0, 10, n),
        "hour": dates.hour,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "day": dates.day,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "season": np.random.choice(["winter", "spring", "summer", "autumn"], n),
        "load_mw_lag_1h": np.random.uniform(4000, 8000, n),
        "load_mw_lag_24h": np.random.uniform(4000, 8000, n),
        "2m_temperature_roll3h": np.random.uniform(5, 35, n),
        "2m_temperature_roll24h": np.random.uniform(5, 35, n),
        "temp_anomaly": np.random.uniform(-5, 5, n),
        "heatwave": np.random.randint(0, 2, n),
        "temp_squared": np.random.uniform(25, 1225, n),
        "temp_x_solar": np.random.uniform(0, 28000, n),
        "feels_like_temp": np.random.uniform(5, 35, n),
    })
    return df


# ──────────────────────────────────────────────
# Testes: encode_categorical_features
# ──────────────────────────────────────────────

def test_encode_categorical_features_removes_season_string():
    df = make_synthetic_dataset(50)
    df_encoded = encode_categorical_features(df)
    assert "season" not in df_encoded.columns


def test_encode_categorical_features_creates_dummies():
    df = make_synthetic_dataset(50)
    df_encoded = encode_categorical_features(df)
    season_cols = [c for c in df_encoded.columns if c.startswith("season_")]
    assert len(season_cols) > 0


def test_encode_categorical_features_all_numeric():
    df = make_synthetic_dataset(50)
    df_encoded = encode_categorical_features(df)
    drop = ["timestamp"]
    numeric_cols = df_encoded.drop(columns=drop, errors="ignore")
    assert all(numeric_cols.dtypes !=
               object), "Ainda existem colunas nao numericas apos encoding"


# ──────────────────────────────────────────────
# Testes: split_time_series
# ──────────────────────────────────────────────

def test_split_time_series_preserves_temporal_order():
    df = make_synthetic_dataset(200)
    X_train, X_test, y_train, y_test, ts_test = split_time_series(
        df, split_ratio=0.8)

    assert len(X_train) > 0
    assert len(X_test) > 0
    # ultimo timestamp de treino deve ser anterior ao primeiro de teste
    assert ts_test.iloc[0] > df["timestamp"].iloc[len(X_train) - 1]


def test_split_time_series_correct_sizes():
    df = make_synthetic_dataset(200)
    X_train, X_test, y_train, y_test, ts_test = split_time_series(
        df, split_ratio=0.8)

    assert len(X_train) == 160
    assert len(X_test) == 40


def test_split_time_series_no_shuffle():
    df = make_synthetic_dataset(200)
    X_train, X_test, y_train, y_test, ts_test = split_time_series(
        df, split_ratio=0.8)

    # timestamps de treino devem ser todos anteriores aos de teste
    ts_train = df["timestamp"].iloc[:len(X_train)]
    assert ts_train.max() < ts_test.min()


def test_split_time_series_missing_target_raises():
    df = make_synthetic_dataset(50)
    df = df.drop(columns=["load_mw"])
    with pytest.raises(ValueError, match="target"):
        split_time_series(df, target_col="load_mw")


def test_split_time_series_missing_timestamp_raises():
    df = make_synthetic_dataset(50)
    df = df.drop(columns=["timestamp"])
    with pytest.raises(ValueError, match="timestamp"):
        split_time_series(df)


# ──────────────────────────────────────────────
# Testes: train_models (com ficheiro temporario)
# ──────────────────────────────────────────────

def test_train_models_returns_two_models(tmp_path, monkeypatch):
    # guardar CSV sintetico e redirecionar MODELS_DIR para tmp_path
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = make_synthetic_dataset(200)
    csv_path = "data/processed/features_v2_advanced.csv"
    df.to_csv(csv_path, index=False)

    results = train_models(dataset_path=csv_path, target_col="load_mw")

    assert "linear_regression" in results
    assert "random_forest" in results


def test_train_models_creates_joblib_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = make_synthetic_dataset(200)
    csv_path = "data/processed/features_v2_advanced.csv"
    df.to_csv(csv_path, index=False)

    results = train_models(dataset_path=csv_path, target_col="load_mw")

    for model_name in ["linear_regression", "random_forest"]:
        model_path = results[model_name]["model_path"]
        assert os.path.exists(
            model_path), f"Ficheiro {model_path} nao foi criado"


def test_train_models_metrics_have_expected_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = make_synthetic_dataset(200)
    csv_path = "data/processed/features_v2_advanced.csv"
    df.to_csv(csv_path, index=False)

    results = train_models(dataset_path=csv_path, target_col="load_mw")

    for model_name in ["linear_regression", "random_forest"]:
        res = results[model_name]
        assert "train" in res
        assert "test" in res
        assert "overfitting_warning" in res
        for split in ["train", "test"]:
            assert "MAE" in res[split]
            assert "RMSE" in res[split]
            assert "R2" in res[split]


def test_train_models_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        train_models(dataset_path="nao_existe.csv")
