# Code/modeling/evaluation.py

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def residuals_dataframe(
    y_true,
    y_pred,
    timestamps: Optional[pd.Series] = None
) -> pd.DataFrame:
    df = pd.DataFrame({
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
    })
    df["residual"] = df["y_true"] - df["y_pred"]

    if timestamps is not None:
        df["timestamp"] = pd.to_datetime(timestamps.values)
        df = df.set_index("timestamp")

    return df


def detect_overfitting(train_metrics: Dict, test_metrics: Dict) -> bool:
    try:
        mae_train = train_metrics["MAE"]
        mae_test = test_metrics["MAE"]
        rmse_train = train_metrics["RMSE"]
        rmse_test = test_metrics["RMSE"]
    except KeyError:
        return False

    overfit_mae = mae_test > 2.5 * mae_train
    overfit_rmse = rmse_test > 2.5 * rmse_train
    return bool(overfit_mae and overfit_rmse)


def residuals_head_as_json_safe_dict(
        residuals_df: pd.DataFrame,
        n: int = 20) -> Dict:
    """
    Converte as primeiras linhas dos residuos para um dicionario
    com chaves string, seguro para json.dump.
    """
    head_df = residuals_df.head(n).copy()

    result = {}
    for idx, row in head_df.iterrows():
        key = str(idx)
        result[key] = {
            "y_true": float(row["y_true"]),
            "y_pred": float(row["y_pred"]),
            "residual": float(row["residual"]),
        }
    return result


def evaluate_model_from_predictions(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    timestamps_test: Optional[pd.Series] = None
) -> Dict:
    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    overfitting = detect_overfitting(train_metrics, test_metrics)
    residuals_test_df = residuals_dataframe(
        y_test, y_test_pred, timestamps_test)

    return {
        "train": train_metrics,
        "test": test_metrics,
        "overfitting_warning": overfitting,
        "residuals_head": residuals_head_as_json_safe_dict(
            residuals_test_df,
            n=20),
    }
