# Testing/test_evaluation.py

import pytest
import numpy as np
import pandas as pd

from Code.modeling.evaluation import (
    compute_metrics,
    residuals_dataframe,
    detect_overfitting,
    evaluate_model_from_predictions,
)


# ──────────────────────────────────────────────
# Testes: compute_metrics
# ──────────────────────────────────────────────

def test_compute_metrics_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    metrics = compute_metrics(y, y)
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    metrics = compute_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics


def test_compute_metrics_mae_positive():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0


def test_compute_metrics_rmse_gte_mae():
    y_true = np.random.uniform(4000, 8000, 100)
    y_pred = y_true + np.random.uniform(-200, 200, 100)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["RMSE"] >= metrics["MAE"]


def test_compute_metrics_r2_between_minus1_and_1():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([150.0, 250.0, 350.0, 450.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["R2"] <= 1.0


# ──────────────────────────────────────────────
# Testes: residuals_dataframe
# ──────────────────────────────────────────────

def test_residuals_dataframe_has_residual_column():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 295.0])
    df = residuals_dataframe(y_true, y_pred)
    assert "residual" in df.columns


def test_residuals_dataframe_correct_values():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 295.0])
    df = residuals_dataframe(y_true, y_pred)
    expected = y_true - y_pred
    np.testing.assert_array_almost_equal(df["residual"].values, expected)


def test_residuals_dataframe_with_timestamps():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 295.0])
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=3, freq="h"))
    df = residuals_dataframe(y_true, y_pred, timestamps=timestamps)
    assert df.index.name == "timestamp"
    assert len(df) == 3


# ──────────────────────────────────────────────
# Testes: detect_overfitting
# ──────────────────────────────────────────────

def test_detect_overfitting_true_when_test_much_worse():
    train_m = {"MAE": 50.0, "RMSE": 70.0, "R2": 0.99}
    test_m = {"MAE": 200.0, "RMSE": 300.0, "R2": 0.85}
    assert detect_overfitting(train_m, test_m) is True


def test_detect_overfitting_false_when_similar():
    train_m = {"MAE": 100.0, "RMSE": 140.0, "R2": 0.93}
    test_m = {"MAE": 120.0, "RMSE": 160.0, "R2": 0.91}
    assert detect_overfitting(train_m, test_m) is False


def test_detect_overfitting_false_on_missing_keys():
    # nao deve lançar excecao se as chaves estiverem em falta
    assert detect_overfitting({}, {}) is False


# ──────────────────────────────────────────────
# Testes: evaluate_model_from_predictions
# ──────────────────────────────────────────────

def test_evaluate_model_returns_full_structure():
    y_train = np.random.uniform(4000, 8000, 160)
    y_train_pred = y_train + np.random.uniform(-100, 100, 160)
    y_test = np.random.uniform(4000, 8000, 40)
    y_test_pred = y_test + np.random.uniform(-200, 200, 40)

    result = evaluate_model_from_predictions(
        y_train, y_train_pred, y_test, y_test_pred)

    assert "train" in result
    assert "test" in result
    assert "overfitting_warning" in result
    assert "residuals_head" in result


def test_evaluate_model_overfitting_flag_is_bool():
    y_train = np.random.uniform(4000, 8000, 160)
    y_train_pred = y_train + np.random.uniform(-50, 50, 160)
    y_test = np.random.uniform(4000, 8000, 40)
    y_test_pred = y_test + np.random.uniform(-500, 500, 40)

    result = evaluate_model_from_predictions(
        y_train, y_train_pred, y_test, y_test_pred)
    assert isinstance(result["overfitting_warning"], bool)
