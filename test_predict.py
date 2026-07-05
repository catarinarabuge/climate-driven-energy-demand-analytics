# Testing/test_predict.py

import os
import pytest
import numpy as np
import pandas as pd

from Code.modeling.model_training import train_models
from Code.modeling.predict import load_model_and_predict, prepare_prediction_input


# ──────────────────────────────────────────────
# Fixture: dataset sintetico + modelos treinados
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


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    """
    Treina os dois modelos uma vez para todos os testes deste ficheiro.
    Devolve (results, csv_path, base_dir).
    """
    base_dir = tmp_path_factory.mktemp("modeling")
    (base_dir / "data" / "processed").mkdir(parents=True)
    (base_dir / "models").mkdir(parents=True)

    os.chdir(base_dir)

    df = make_synthetic_dataset(200)
    csv_path = str(
        base_dir /
        "data" /
        "processed" /
        "features_v2_advanced.csv")
    df.to_csv(csv_path, index=False)

    results = train_models(dataset_path=csv_path, target_col="load_mw")
    return results, csv_path, base_dir


# ──────────────────────────────────────────────
# Testes: prepare_prediction_input
# ──────────────────────────────────────────────

def test_prepare_prediction_input_removes_target(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    import joblib
    model = joblib.load(model_path)

    df = make_synthetic_dataset(10)
    X = prepare_prediction_input(df, model=model, target_col="load_mw")

    assert "load_mw" not in X.columns


def test_prepare_prediction_input_removes_timestamp(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    import joblib
    model = joblib.load(model_path)

    df = make_synthetic_dataset(10)
    X = prepare_prediction_input(df, model=model, target_col="load_mw")

    assert "timestamp" not in X.columns


def test_prepare_prediction_input_encodes_season(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    import joblib
    model = joblib.load(model_path)

    df = make_synthetic_dataset(10)
    X = prepare_prediction_input(df, model=model, target_col="load_mw")

    # season string deve ter desaparecido
    assert "season" not in X.columns

    # deve existir pelo menos uma coluna season_*
    season_cols = [c for c in X.columns if c.startswith("season_")]
    assert len(season_cols) > 0


def test_prepare_prediction_input_missing_timestamp_raises(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    import joblib
    model = joblib.load(model_path)

    df = make_synthetic_dataset(10).drop(columns=["timestamp"])
    with pytest.raises(ValueError, match="timestamp"):
        prepare_prediction_input(df, model=model, target_col="load_mw")


# ──────────────────────────────────────────────
# Testes: load_model_and_predict - random forest
# ──────────────────────────────────────────────

def test_predict_random_forest_returns_correct_n_samples(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    df = make_synthetic_dataset(24)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    assert result["n_samples"] == 24


def test_predict_random_forest_predictions_are_list(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    df = make_synthetic_dataset(24)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 24


def test_predict_random_forest_predictions_are_floats(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    df = make_synthetic_dataset(24)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    for val in result["predictions"]:
        assert isinstance(val, float)


def test_predict_random_forest_timestamps_match_input(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    df = make_synthetic_dataset(24)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    assert len(result["timestamps"]) == 24


def test_predict_random_forest_predictions_in_plausible_range(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    df = make_synthetic_dataset(50)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    for val in result["predictions"]:
        assert 0 < val < 50000, f"Previsao fora de intervalo plausivel: {val}"


# ──────────────────────────────────────────────
# Testes: load_model_and_predict - regressao linear
# ──────────────────────────────────────────────

def test_predict_linear_returns_correct_n_samples(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["linear_regression"]["model_path"]

    df = make_synthetic_dataset(12)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    assert result["n_samples"] == 12


def test_predict_linear_predictions_are_list(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["linear_regression"]["model_path"]

    df = make_synthetic_dataset(12)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 12


# ──────────────────────────────────────────────
# Testes: erros esperados
# ──────────────────────────────────────────────

def test_predict_model_not_found_raises():
    df = make_synthetic_dataset(10)
    with pytest.raises(FileNotFoundError):
        load_model_and_predict("models/nao_existe.joblib", df)


def test_predict_result_has_expected_keys(trained_models):
    results, csv_path, base_dir = trained_models
    model_path = results["random_forest"]["model_path"]

    df = make_synthetic_dataset(10)
    result = load_model_and_predict(model_path, df, target_col="load_mw")

    assert "model_path" in result
    assert "n_samples" in result
    assert "predictions" in result
    assert "timestamps" in result
