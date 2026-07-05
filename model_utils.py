# Code/modeling/model_utils.py

import os
import json
from typing import Dict, List
from datetime import datetime
import joblib
import pandas as pd


MODELS_DIR = "models"
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")


def ensure_models_dir(path: str = MODELS_DIR) -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_model(model, name: str) -> str:
    ensure_models_dir()
    fname = f"{name}_{timestamp_str()}.joblib"
    fpath = os.path.join(MODELS_DIR, fname)
    joblib.dump(model, fpath)
    return fpath


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo nao encontrado em: {path}")
    return joblib.load(path)


def load_metrics() -> Dict:
    if not os.path.exists(METRICS_FILE):
        return {}
    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_metrics(metrics: Dict):
    ensure_models_dir()
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)


def append_model_metrics(model_name: str, model_type: str, metrics: Dict):
    all_metrics = load_metrics()

    entry = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
    }

    if model_name not in all_metrics:
        all_metrics[model_name] = []
    all_metrics[model_name].append(entry)
    save_metrics(all_metrics)


def get_feature_columns(
        df: pd.DataFrame,
        target_col: str,
        extra_drop: List[str] = None) -> List[str]:
    drop_cols = [target_col]
    if "timestamp" in df.columns:
        drop_cols.append("timestamp")
    if extra_drop:
        drop_cols.extend(extra_drop)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols
