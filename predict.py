# Code/modeling/predict.py

from typing import Dict

import pandas as pd

from .model_utils import load_model
from .model_training import encode_categorical_features


def prepare_prediction_input(
    features_df: pd.DataFrame,
    model,
    target_col: str = "load_mw"
) -> pd.DataFrame:
    if "timestamp" not in features_df.columns:
        raise ValueError("Coluna 'timestamp' em falta em features_df.")

    df = features_df.copy()
    df = encode_categorical_features(df)

    drop_cols = [target_col]
    if "timestamp" in df.columns:
        drop_cols.append("timestamp")

    X = df.drop(
        columns=[
            c for c in drop_cols if c in df.columns],
        errors="ignore")

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        X = X.reindex(columns=model_features, fill_value=0)

    return X


def load_model_and_predict(
    model_path: str,
    features_df: pd.DataFrame,
    target_col: str = "load_mw"
) -> Dict:
    model = load_model(model_path)
    X = prepare_prediction_input(
        features_df,
        model=model,
        target_col=target_col)

    y_pred = model.predict(X)

    result = {
        "model_path": model_path,
        "n_samples": len(X),
        "predictions": y_pred.tolist(),
        "timestamps": features_df["timestamp"].astype(str).tolist(),
    }
    return result
