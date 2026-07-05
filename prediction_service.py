# Code/interface/prediction_service.py

import os
from typing import Dict

import pandas as pd

from Code.auth.auth_service import authenticate_user, get_user_role
from Code.modeling.model_training import train_models, load_feature_dataset
from Code.modeling.predict import load_model_and_predict
from Code.modeling.model_utils import load_metrics


DATA_FEATURES = "data/processed/features_v2_advanced.csv"
TARGET_COL = "load_mw"


class AuthorizationError(Exception):
    pass


def _ensure_authenticated(
        username: str,
        password: str,
        require_admin: bool = False):
    if not authenticate_user(username, password):
        raise AuthorizationError("Credenciais invalidas.")

    role = get_user_role(username)
    if require_admin and role != "admin":
        raise AuthorizationError("Apenas admins podem executar esta acao.")
    return role


def authenticated_train(
    username: str,
    password: str,
    dataset_path: str = DATA_FEATURES,
    target_col: str = TARGET_COL,
) -> Dict:
    """
    Treina modelos apenas se utilizador for admin.
    """
    role = _ensure_authenticated(username, password, require_admin=True)

    results = train_models(dataset_path=dataset_path, target_col=target_col)
    return {
        "user": username,
        "role": role,
        "results": results,
    }


def authenticated_predict_from_features(
    username: str,
    password: str,
    model_path: str,
    features_df: pd.DataFrame,
) -> Dict:
    """
    Gera previsoes usando um modelo treinado, para qualquer user autenticado.
    """
    role = _ensure_authenticated(username, password, require_admin=False)

    # garantir que existe coluna timestamp
    if "timestamp" not in features_df.columns:
        raise ValueError("features_df deve conter a coluna 'timestamp'.")

    result = load_model_and_predict(
        model_path, features_df, target_col=TARGET_COL)
    result["user"] = username
    result["role"] = role
    return result


def authenticated_latest_metrics(
    username: str,
    password: str,
) -> Dict:
    """
    Devolve métricas guardadas dos modelos.
    """
    role = _ensure_authenticated(username, password, require_admin=False)
    all_metrics = load_metrics()
    return {
        "user": username,
        "role": role,
        "metrics": all_metrics,
    }


def load_default_features_subset(
    n_last: int = 24,
    dataset_path: str = DATA_FEATURES,
) -> pd.DataFrame:
    """
    Carrega features_v2_advanced e devolve as ultimas n_last linhas
    (para previsao das proximas horas).
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Ficheiro de features nao encontrado: {dataset_path}")

    df = load_feature_dataset(dataset_path)
    if n_last is not None and n_last > 0:
        df = df.tail(n_last).copy()
    return df
