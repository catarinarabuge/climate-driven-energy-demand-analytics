"""
Climate-Driven Energy Demand Analytics System
Flask Web App — Milestone 2/3
"""

import os
import sys
import time
import logging
import datetime
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR))

# ── Auth ─────────────────────────────────────────────
try:
    from Code.auth.auth_service import register_user, authenticate_user
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False


# 2. Tentar importar as funções de Admin. Se não existirem (porque o
# colega ainda não as fez), criamos um Plano B seguro!
try:
    from Code.auth.auth_service import get_user_role, promote_to_admin
except ImportError:
    # PLANO B: Se o utilizador se chamar "admin", damos-lhe o painel de
    # administração!
    def get_user_role(username):
        if username.lower() == "admin":
            return "admin"
        return "user"

    def promote_to_admin(current_user, target_user):
        raise Exception(
            "O módulo do teu colega ainda não suporta esta função.")

# ── Pipeline ──────────────────────────────────────────
try:
    from Code.ingestion.entsoe_client import run_ingestion_raw, run_processing
    from Code.ingestion.era5 import download_era5, ingest_era5
    from Code.cleaning.merge_datasets import merge_datasets
    from Code.features.base_features import generate_features
    from Code.features.advanced_features import generate_advanced_features
    PIPELINE_AVAILABLE = True
except Exception as e:
    logging.warning(f"Pipeline modules unavailable: {e}")
    PIPELINE_AVAILABLE = False

# ── Modeling ──────────────────────────────────────────
try:
    from Code.interface.prediction_service import (
        authenticated_train,
        authenticated_latest_metrics,
        authenticated_predict_from_features,
        load_default_features_subset,
    )
    MODELING_AVAILABLE = True
except Exception as e:
    logging.warning(f"Modeling module unavailable: {e}")
    MODELING_AVAILABLE = False

# ── Logging ───────────────────────────────────────────
_log_file = os.path.join(BASE_DIR, "system_actions.log")
_logger = logging.getLogger("system_actions")
if not _logger.handlers:
    _handler = logging.FileHandler(_log_file, encoding="utf-8")
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    _logger.setLevel(logging.INFO)
    _logger.addHandler(_handler)

# ── Paths ─────────────────────────────────────────────
DATA_ENERGY = os.path.join(BASE_DIR, "data/processed/clean_energy.csv")
DATA_WEATHER = os.path.join(BASE_DIR, "data/processed/clean_weather.csv")
DATA_FEATURES = os.path.join(
    BASE_DIR, "data/processed/features_v2_advanced.csv")
DATA_FEATURES_FALLBACK = os.path.join(
    BASE_DIR, "data/processed/features_v1.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, template_folder="Design")
app.secret_key = os.environ.get("SECRET_KEY", "piacd-pl1g3-dev-key")


# ══════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════

def log_action(action, extra=""):
    user = session.get("user", "anonymous")
    _logger.info(f"user={user} | action={action} | {extra}")


def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar {path}: {e}")
        return None


def require_admin():
    """Returns error JSON if not admin, else None."""
    if not session.get("logged_in"):
        return jsonify({"error": "Não autenticado"}), 403
    if session.get("role") != "admin":
        return jsonify({"error": "Acesso negado. Apenas admins."}), 403
    return None


def confirm_password(password):
    """Re-authenticates the current session user with given password."""
    username = session.get("user")
    if not username or not password:
        return False
    try:
        return authenticate_user(username, password)
    except Exception:
        return False


# ══════════════════════════════════════════════════════
# Auth routes
# ══════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template(
        "landing_page.html",
        error=request.args.get("error"),
        reg_error=request.args.get("reg_error"),
        reg_success=request.args.get("reg_success"),
    )


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not username or not password:
        return redirect(
            url_for(
                "index",
                error="Username e password são obrigatórios."))

    if not AUTH_AVAILABLE:
        session["user"] = username
        session["logged_in"] = True

        session["role"] = "admin"

        session["role"] = "admin"  # Sem auth, pomos como admin para testares
        log_action("LOGIN_DEMO", f"username={username}")
        return redirect(url_for("dashboard"))

    try:
        if authenticate_user(username, password):
            session["user"] = username
            session["logged_in"] = True

            # Aqui chamamos o nosso Plano B ou a função do colega
            session["role"] = get_user_role(username)
            log_action("LOGIN_SUCCESS", f"username={username}")
            return redirect(url_for("dashboard"))
        log_action("LOGIN_FAILED", f"username={username}")
        return redirect(url_for("index", error="Credenciais inválidas."))
    except Exception:
        return redirect(
            url_for(
                "index",
                error="Erro interno na autenticação."))


@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")

    if not username or not password:
        log_action("REGISTER_FAILED", f"username={username} | reason=empty fields")
        return redirect(
            url_for(
                "index",
                reg_error="Todos os campos são obrigatórios."))
    if len(password) < 8:
        log_action("REGISTER_FAILED", f"username={username} | reason=password too short")
        return redirect(
            url_for(
                "index",
                reg_error="A password deve ter pelo menos 8 caracteres."))
    if password != confirm:
        log_action("REGISTER_FAILED", f"username={username} | reason=passwords do not match")
        return redirect(
            url_for(
                "index",
                reg_error="As passwords não coincidem."))
    if not AUTH_AVAILABLE:
        log_action("REGISTER_FAILED", f"username={username} | reason=auth module unavailable")
        return redirect(
            url_for(
                "index",
                reg_error="Módulo de autenticação não disponível."))

    try:
        register_user(username, password)
        log_action("REGISTER_SUCCESS", f"username={username}")
        return redirect(
            url_for(
                "index",
                reg_success=f"Utilizador '{username}' registado com sucesso."))
    except ValueError as e:
        log_action("REGISTER_FAILED", f"username={username} | reason={e}")
        return redirect(url_for("index", reg_error=str(e)))


@app.route("/logout")
def logout():
    log_action("LOGOUT")
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(
            url_for(
                "index",
                error="Tens de fazer login primeiro."))

    role = session.get("role", "user")
    if role == "admin":
        energy_df = load_csv(DATA_ENERGY)
        stats = {}
        if energy_df is not None:
            stats = {
                "min_date": str(energy_df.index.min().date()),
                "max_date": str(energy_df.index.max().date()),
                "n_rows": len(energy_df),
            }
        return render_template("admin.html", stats=stats)

    return render_template("user.html")


# ══════════════════════════════════════════════════════
# Admin: promote
# ══════════════════════════════════════════════════════

@app.route("/admin/promote", methods=["POST"])
def web_promote_user():
    if session.get("role") != "admin":
        return redirect(
            url_for(
                "index",
                error="Acesso negado. Apenas para Admins."))

    target_user = request.form.get("target_user")
    try:
        promote_to_admin(session.get("user"), target_user)
        log_action("PROMOTE_SUCCESS", f"target_user={target_user}")
        return redirect(
            url_for(
                "index",
                reg_success=f"Utilizador '{target_user}' promovido a Admin!"))
    except Exception as e:
        log_action("PROMOTE_FAILED", f"target_user={target_user} | reason={e}")
        return redirect(url_for("index", error=str(e)))


# ══════════════════════════════════════════════════════
# API: dados (energia + clima) — usados pelos gráficos
# ══════════════════════════════════════════════════════


@app.route("/api/energy")
def api_energy():
    if not session.get("logged_in"):
        return jsonify({"error": "Não autenticado"}), 403

    date_str = request.args.get("date")
    hour_str = request.args.get("hour")
    df = load_csv(DATA_ENERGY)
    if df is None:
        return jsonify({"error": "Dados não disponíveis"}), 404

    try:
        date = datetime.date.fromisoformat(date_str)
    except Exception:
        date = df.index.min().date()

    day = df[df.index.date == date]

    if hour_str is not None:
        try:
            h = int(hour_str)
            day = day[day.index.hour == h]
        except ValueError:
            pass

    col = df.columns[0]
    return jsonify({
        "labels": [f"{ts.hour:02d}:00" for ts in day.index],
        "values": [round(float(v), 2) for v in day[col]],
        "date": str(date)
    })


@app.route("/api/weather")
def api_weather():
    if not session.get("logged_in"):
        return jsonify({"error": "Não autenticado"}), 403

    date_str = request.args.get("date")
    hour_str = request.args.get("hour")
    df = load_csv(DATA_WEATHER)
    if df is None:
        return jsonify({"error": "Dados não disponíveis"}), 404

    try:
        date = datetime.date.fromisoformat(date_str)
    except Exception:
        date = df.index.min().date()

    day = df[df.index.date == date]

    if hour_str is not None:
        try:
            h = int(hour_str)
            day = day[day.index.hour == h]
        except ValueError:
            pass

    col = "2m_temperature" if "2m_temperature" in df.columns else df.columns[0]
    return jsonify({
        "labels": [f"{ts.hour:02d}:00" for ts in day.index],
        "temps": [round(float(v), 2) for v in day[col]],
        "date": str(date)
    })


# ══════════════════════════════════════════════════════
# API: features (opção 2 do admin)
# ══════════════════════════════════════════════════════

@app.route("/api/features")
def api_features():
    err = require_admin()
    if err:
        return err

    date_str = request.args.get("date")
    hour_str = request.args.get("hour")

    feat_path = DATA_FEATURES if os.path.exists(
        DATA_FEATURES) else DATA_FEATURES_FALLBACK
    if not os.path.exists(feat_path):
        return jsonify(
            {"error": "Ficheiro de features não encontrado. Corre o Feature Engineering primeiro."}), 404

    df = load_csv(feat_path)
    if df is None:
        return jsonify({"error": "Erro ao carregar features."}), 500

    try:
        date = datetime.date.fromisoformat(date_str)
    except Exception:
        date = df.index.min().date()

    try:
        day = df[df.index.date == date]

        if hour_str is not None:
            try:
                h = int(hour_str)
                day = day[day.index.hour == h]
            except (ValueError, AttributeError):
                pass

        if day.empty:
            return jsonify({"rows": [], "columns": list(
                df.columns), "date": str(date)})

        cols = list(day.columns)
        rows = []
        for ts, row in day.iterrows():
            r = {"timestamp": str(ts)}
            for c in cols:
                v = row[c]
                try:
                    r[c] = round(float(v), 6) if pd.notna(v) else None
                except (ValueError, TypeError):
                    r[c] = str(v) if pd.notna(v) else None
            rows.append(r)
    except Exception as e:
        return jsonify({"error": f"Erro ao processar features: {e}"}), 500

    log_action("FEATURES_VIEW", f"date={date} | rows={len(rows)}")
    return jsonify({"rows": rows, "columns": cols, "date": str(date)})


# ══════════════════════════════════════════════════════
# API: Pipeline routes (opções 3, 4, 5)
# ══════════════════════════════════════════════════════

@app.route("/api/pipeline/ingestion", methods=["POST"])
def api_pipeline_ingestion():
    err = require_admin()
    if err:
        return err

    if not PIPELINE_AVAILABLE:
        return jsonify({"error": "Módulos de pipeline não disponíveis."}), 503

    log_action("INGESTION_START")
    try:
        run_ingestion_raw()
        raw_dir = os.path.join(BASE_DIR, "data/raw/weather")
        output_dir = os.path.join(BASE_DIR, "data/processed")
        grib_path = download_era5(raw_dir)
        ingest_era5(grib_path, output_dir)
        log_action("INGESTION_SUCCESS")
        return jsonify(
            {"ok": True, "message": "Ingestão concluída com sucesso."})
    except Exception as e:
        log_action("INGESTION_FAILED", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/cleaning", methods=["POST"])
def api_pipeline_cleaning():
    err = require_admin()
    if err:
        return err

    if not PIPELINE_AVAILABLE:
        return jsonify({"error": "Módulos de pipeline não disponíveis."}), 503

    log_action("CLEANING_START")
    try:
        run_processing()
        merge_datasets(
            energy_path=os.path.join(
                BASE_DIR,
                "data/processed/energy_load_PT_2025_hourly_clean.csv"),
            weather_path=os.path.join(
                BASE_DIR,
                "data/processed/era5_portugal_processed.csv"),
            output_path=os.path.join(
                BASE_DIR,
                "data/processed/merged.csv"),
        )
        log_action("CLEANING_SUCCESS")
        return jsonify(
            {"ok": True, "message": "Cleaning e alinhamento concluídos com sucesso."})
    except Exception as e:
        log_action("CLEANING_FAILED", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/features", methods=["POST"])
def api_pipeline_features():
    err = require_admin()
    if err:
        return err

    if not PIPELINE_AVAILABLE:
        return jsonify({"error": "Módulos de pipeline não disponíveis."}), 503

    log_action("FEATURE_ENGINEERING_START")
    try:
        generate_features(
            input_path=os.path.join(
                BASE_DIR,
                "data/processed/merged.csv"),
            output_path=os.path.join(
                BASE_DIR,
                "data/processed/features_v1.csv"),
        )
        generate_advanced_features(
            input_path=os.path.join(
                BASE_DIR,
                "data/processed/features_v1.csv"),
            output_path=os.path.join(
                BASE_DIR,
                "data/processed/features_v2_advanced.csv"),
        )
        log_action("FEATURE_ENGINEERING_SUCCESS")
        return jsonify(
            {"ok": True, "message": "Feature engineering concluído com sucesso."})
    except Exception as e:
        log_action("FEATURE_ENGINEERING_FAILED", str(e))
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════
# API: Treinar modelos (opção 6)
# ══════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def api_train():
    err = require_admin()
    if err:
        return err

    if not MODELING_AVAILABLE:
        return jsonify({"error": "Módulo de modeling não disponível."}), 503

    data = request.get_json(force=True) or {}
    password = data.get("password", "")

    if not confirm_password(password):
        log_action("TRAINING_DENIED", "bad password")
        return jsonify({"error": "Password incorrecta."}), 403

    feat_path = DATA_FEATURES if os.path.exists(
        DATA_FEATURES) else DATA_FEATURES_FALLBACK
    if not os.path.exists(feat_path):
        return jsonify(
            {"error": "Ficheiro de features não encontrado. Corre o Feature Engineering primeiro."}), 404

    log_action("TRAINING_START")
    try:
        t0 = time.time()
        results = authenticated_train(
            username=session["user"],
            password=password,
            dataset_path=feat_path,
            target_col="load_mw",
        )
        elapsed = round(time.time() - t0, 4)
        log_action("TRAINING_SUCCESS", f"elapsed={elapsed}s")
        return jsonify({
            "ok": True,
            "elapsed": elapsed,
            "results": results.get("results", {}),
        })
    except Exception as e:
        log_action("TRAINING_FAILED", str(e))
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════
# API: Ver métricas (opção 7)
# ══════════════════════════════════════════════════════

@app.route("/api/metrics", methods=["POST"])
def api_metrics():
    err = require_admin()
    if err:
        return err

    if not MODELING_AVAILABLE:
        return jsonify({"error": "Módulo de modeling não disponível."}), 503

    data = request.get_json(force=True) or {}
    password = data.get("password", "")

    if not confirm_password(password):
        log_action("METRICS_DENIED", "bad password")
        return jsonify({"error": "Password incorrecta."}), 403

    log_action("METRICS_VIEW")
    try:
        res = authenticated_latest_metrics(
            username=session["user"], password=password)
        return jsonify({"ok": True, "metrics": res.get("metrics", {})})
    except Exception as e:
        log_action("METRICS_FAILED", str(e))
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════
# API: Gerar previsão admin (opção 8)
# ══════════════════════════════════════════════════════

@app.route("/api/predict_admin", methods=["POST"])
def api_predict_admin():
    err = require_admin()
    if err:
        return err

    if not MODELING_AVAILABLE:
        return jsonify({"error": "Módulo de modeling não disponível."}), 503

    data = request.get_json(force=True) or {}
    password = data.get("password", "")
    n_last = int(data.get("n_last", 24))

    if not confirm_password(password):
        log_action("PREDICTION_DENIED", "bad password")
        return jsonify({"error": "Password incorrecta."}), 403

    feat_path = DATA_FEATURES if os.path.exists(
        DATA_FEATURES) else DATA_FEATURES_FALLBACK
    if not os.path.exists(feat_path):
        return jsonify({"error": "Ficheiro de features não encontrado."}), 404

    # Escolher o modelo mais recente automaticamente
    model_path = None
    if os.path.exists(MODELS_DIR):
        files = sorted([f for f in os.listdir(
            MODELS_DIR) if f.endswith(".joblib")])
        if files:
            model_path = os.path.join(MODELS_DIR, files[-1])

    if not model_path:
        return jsonify(
            {"error": "Nenhum modelo treinado encontrado. Treina os modelos primeiro."}), 404

    log_action("PREDICTION_START", f"model={model_path} | n_last={n_last}")
    try:
        features_subset = load_default_features_subset(
            n_last=n_last, dataset_path=feat_path)
        features_subset = features_subset.reset_index()

        result = authenticated_predict_from_features(
            username=session["user"],
            password=password,
            model_path=model_path,
            features_df=features_subset,
        )
        log_action(
            "PREDICTION_SUCCESS",
            f"n={result['n_samples']} | model={model_path}")
        return jsonify({
            "ok": True,
            "model": os.path.basename(model_path),
            "n_samples": result["n_samples"],
            "timestamps": [str(t) for t in result["timestamps"]],
            "predictions": [round(float(p), 2) for p in result["predictions"]],
        })
    except Exception as e:
        log_action("PREDICTION_FAILED", str(e))
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════
# API: previsão user (user.html → /predict)
# ══════════════════════════════════════════════════════

@app.route("/forecast")
def forecast():
    if not session.get("logged_in"):
        return redirect(
            url_for(
                "index",
                error="Tens de fazer login primeiro."))
    return render_template("user.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not session.get("logged_in"):
        return jsonify({"error": "Não autenticado"}), 403

    data = request.get_json(force=True) or {}

    temperature = float(data.get("temperature",   16.8))
    wind = float(data.get("wind",           2.6))
    solar = float(data.get("solar",          200.0))
    precipitation = float(data.get("precipitation",  0.1))
    hour = int(data.get("hour",        12))
    day_of_week = int(data.get("day_of_week", 2))
    month = int(data.get("month",       6))
    day = int(data.get("day",         15))
    is_weekend = int(data.get("is_weekend",  0))
    season = str(data.get("season",      "spring"))
    temp_roll3h = float(data.get("temp_roll3h",  temperature))
    temp_roll24h = float(data.get("temp_roll24h", temperature))
    lag_1h = float(data.get("lag_1h",  6057.0))
    lag_24h = float(data.get("lag_24h", 6057.0))

    # Monthly mean temperatures precomputed from training dataset
    MONTHLY_MEANS = {
        1: 12.00, 2: 12.06, 3: 12.35, 4: 15.03,
        5: 17.39, 6: 21.79, 7: 22.74, 8: 23.37,
        9: 20.02, 10: 18.93, 11: 13.74, 12: 11.28,
    }
    HEATWAVE_THRESH = 24.21  # 90th percentile of 2m_temperature in training set

    def build_row(h, temp, wind_, solar_, precip):
        return {
            "2m_temperature":                    temp,
            "10m_v_component_of_wind":           wind_,
            "surface_solar_radiation_downwards": solar_,
            "total_precipitation":               precip,
            "hour":                              h,
            "day_of_week":                       day_of_week,
            "month":                             month,
            "day":                               day,
            "is_weekend":                        is_weekend,
            "load_mw_lag_1h":                    lag_1h,
            "load_mw_lag_24h":                   lag_24h,
            "2m_temperature_roll3h":             temp_roll3h,
            "2m_temperature_roll24h":            temp_roll24h,
            "temp_anomaly":  temp - MONTHLY_MEANS.get(month, 16.8),
            "heatwave":      1 if temp >= HEATWAVE_THRESH else 0,
            "temp_squared":  temp ** 2,
            "temp_x_solar":  temp * solar_,
            "feels_like_temp": temp - 0.7 * abs(wind_),
            "season_autumn": 1 if season == "autumn" else 0,
            "season_spring": 1 if season == "spring" else 0,
            "season_summer": 1 if season == "summer" else 0,
            "season_winter": 1 if season == "winter" else 0,
        }

    # Prefer the most recent random_forest model
    model_path = None
    if os.path.exists(MODELS_DIR):
        rf_files = sorted(
            f for f in os.listdir(MODELS_DIR)
            if f.startswith("random_forest") and f.endswith(".joblib")
        )
        if rf_files:
            model_path = os.path.join(MODELS_DIR, rf_files[-1])
        else:
            all_files = sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib"))
            if all_files:
                model_path = os.path.join(MODELS_DIR, all_files[-1])

    if not model_path:
        return jsonify({"error": "Nenhum modelo treinado. Treina os modelos primeiro."}), 404

    try:
        model = joblib.load(model_path)
        model_cols = list(getattr(model, "feature_names_in_", []))

        # Point prediction for the requested hour
        X_point = pd.DataFrame([build_row(hour, temperature, wind, solar, precipitation)])
        if model_cols:
            X_point = X_point.reindex(columns=model_cols, fill_value=0)
        point_pred = round(float(model.predict(X_point)[0]))

        # 24-hour load curve (hour varies, everything else fixed)
        chart_rows = [build_row(h, temperature, wind, solar, precipitation) for h in range(24)]
        X_chart = pd.DataFrame(chart_rows)
        if model_cols:
            X_chart = X_chart.reindex(columns=model_cols, fill_value=0)
        chart_vals = [round(float(p)) for p in model.predict(X_chart)]

        # Confidence interval = point ± MAE from stored metrics
        mae = 250
        try:
            from Code.modeling.model_utils import load_metrics
            all_metrics = load_metrics()
            model_key = "random_forest" if "random_forest" in os.path.basename(model_path) else "linear_regression"
            if model_key in all_metrics and all_metrics[model_key]:
                mae = round(all_metrics[model_key][-1].get("metrics", {}).get("test", {}).get("MAE", 250))
        except Exception:
            pass

        log_action("PREDICT_SUCCESS", f"model={os.path.basename(model_path)} | pred={point_pred}")
        return jsonify({
            "prediction":   point_pred,
            "interval":     [point_pred - mae, point_pred + mae],
            "confidence":   90,
            "chart_labels": [f"{h:02d}h" for h in range(24)],
            "chart_values": chart_vals,
        })

    except Exception as e:
        log_action("PREDICT_FAILED", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
