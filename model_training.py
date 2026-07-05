# Code/modeling/model_training.py

import os
from typing import Tuple, Dict

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from .evaluation import evaluate_model_from_predictions
from .model_utils import save_model, append_model_metrics


DEFAULT_DATASET = "data/processed/features_v2_advanced.csv"
TARGET_COL = "load_mw"


def load_feature_dataset(path: str = DEFAULT_DATASET) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ficheiro de features nao encontrado: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").copy()

    # garantir booleanos como inteiros, se existirem
    for col in ["is_weekend", "heatwave"]:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(int)

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica colunas categóricas para numérico.
    Mantemos 'season' no modelo via one-hot encoding.
    """
    df = df.copy()

    categorical_cols = []
    if "season" in df.columns:
        categorical_cols.append("season")

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL):
    drop_cols = [target_col]
    if "timestamp" in df.columns:
        drop_cols.append("timestamp")

    return [c for c in df.columns if c not in drop_cols]


def split_time_series(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    split_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if "timestamp" not in df.columns:
        raise ValueError("Coluna 'timestamp' em falta no dataset de features.")

    if target_col not in df.columns:
        raise ValueError(f"Coluna target '{target_col}' nao encontrada.")

    df = df.sort_values("timestamp").copy()
    df = encode_categorical_features(df)

    feature_cols = get_feature_columns(df, target_col=target_col)

    n_samples = len(df)
    if n_samples < 10:
        raise ValueError("Poucos dados para treino/teste.")

    split_index = int(n_samples * split_ratio)

    X = df[feature_cols]
    y = df[target_col]
    timestamps = df["timestamp"]

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    ts_test = timestamps.iloc[split_index:]

    return X_train, X_test, y_train, y_test, ts_test


def build_linear_model() -> LinearRegression:
    return LinearRegression()


def build_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=50,
        min_samples_split=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )


def train_single_model(
    model,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    ts_test
) -> Dict:
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    eval_results = evaluate_model_from_predictions(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        timestamps_test=ts_test,
    )

    model_path = save_model(model, model_name)
    eval_results["model_path"] = model_path

    append_model_metrics(
        model_name=model_name,
        model_type=model.__class__.__name__,
        metrics=eval_results,
    )

    return eval_results


def train_models(
    dataset_path: str = DEFAULT_DATASET,
    target_col: str = TARGET_COL,
    split_ratio: float = 0.8
) -> Dict[str, Dict]:
    df = load_feature_dataset(dataset_path)

    X_train, X_test, y_train, y_test, ts_test = split_time_series(
        df, target_col=target_col, split_ratio=split_ratio
    )

    results = {}

    lin_model = build_linear_model()
    lin_results = train_single_model(
        lin_model,
        model_name="linear_regression",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        ts_test=ts_test,
    )
    results["linear_regression"] = lin_results

    rf_model = build_random_forest()
    rf_results = train_single_model(
        rf_model,
        model_name="random_forest",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        ts_test=ts_test,
    )
    results["random_forest"] = rf_results

    return results
