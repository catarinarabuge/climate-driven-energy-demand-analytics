"""
Climate-Driven Energy Demand Analytics System
CLI Interface - Milestone 3
"""

import os
import sys
import time
import getpass
import logging
import datetime
from pathlib import Path

LOG_FILE = str(Path(__file__).resolve().parent / "system_actions.log")
_cli_logger = logging.getLogger("cli_actions")
if not _cli_logger.handlers:
    _cli_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _cli_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    _cli_logger.setLevel(logging.INFO)
    _cli_logger.addHandler(_cli_handler)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Code"))

# Auth
try:
    from Code.auth.auth_service import (
        register_user,
        authenticate_user,
        get_user_role,
        promote_to_admin,
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False

# Pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Projeto: ingestion / cleaning / features
try:
    from Code.ingestion.entsoe_client import run_ingestion_raw, run_processing
    from Code.ingestion.era5 import download_era5, ingest_era5
    from Code.cleaning.merge_datasets import merge_datasets
    from Code.features.base_features import generate_features
    from Code.features.advanced_features import generate_advanced_features
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# Modeling / Prediction
try:
    from Code.interface.prediction_service import (
        authenticated_train,
        authenticated_latest_metrics,
        authenticated_predict_from_features,
        load_default_features_subset,
    )
    MODELING_AVAILABLE = True
except ImportError:
    MODELING_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent

DATA_ENERGY = BASE_DIR / "data" / "processed" / \
    "energy_load_PT_2025_hourly_clean.csv"
DATA_WEATHER = BASE_DIR / "data" / "processed" / "era5_portugal_processed.csv"
DATA_MERGED = BASE_DIR / "data" / "processed" / "merged.csv"
DATA_FEATURES = BASE_DIR / "data" / "processed" / "features_v2_advanced.csv"
DATA_FEATURES_FALLBACK = BASE_DIR / "data" / "processed" / "features_v1.csv"
MODELS_DIR = BASE_DIR / "models"

WIDTH = 70

session = {
    "user": None,
    "logged_in": False,
    "role": None,
}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def line(char="-"):
    print(char * WIDTH)


def header(title):
    print()
    line("=")
    print(f"  {title}")
    line("=")


def section(title):
    print()
    line("-")
    print(f"  {title}")
    line("-")


def info(msg):
    print(f"  [INFO] {msg}")


def success(msg):
    print(f"  [OK] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def error(msg):
    print(f"  [ERROR] {msg}")


def prompt(msg, default=None):
    suffix = f" [{default}]" if default else ""
    return input(f"\n  -> {msg}{suffix}: ").strip() or default


def numbered_menu(options, title="Escolhe uma opcao"):
    section(title)
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    print()
    choice = input("  -> ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return idx
    except ValueError:
        pass
    return None


def pause():
    input("\n  Pressiona Enter para continuar...")


def log_action(action, extra=""):
    user = session["user"] or "anonymous"
    _cli_logger.info(
        f"user={user} | role={session['role']} | action={action} | {extra}"
    )


def screen_welcome():
    clear()
    header("Climate-Driven Energy Demand Analytics System")
    print()
    print("  Universidade de Coimbra - DEI")
    print("  PIACD 2025/2026 - Grupo L1G3")
    print()
    line()
    print()
    print("    1. Login")
    print("    2. Registar novo utilizador")
    print("    3. Sair")
    print()


def do_login():
    section("Login")
    username = prompt("Username")
    password = getpass.getpass("  -> Password: ")

    if not username or not password:
        error("Username e password sao obrigatorios.")
        log_action("LOGIN_FAILED", "empty fields")
        pause()
        return False

    if not AUTH_AVAILABLE:
        warn("Modulo de autenticacao nao encontrado.")
        pause()
        return False

    try:
        ok = authenticate_user(username, password)
        if ok:
            session["user"] = username
            session["logged_in"] = True
            session["role"] = get_user_role(username)
            log_action("LOGIN_SUCCESS", f"username={username}")
            success(f"Bem-vindo, {username}! Role: {session['role']}")
            pause()
            return True

        error("Credenciais invalidas.")
        log_action("LOGIN_FAILED", f"username={username}")
        pause()
        return False

    except Exception as e:
        error(f"Erro interno na autenticacao: {e}")
        log_action("LOGIN_ERROR", f"username={username} | error={e}")
        pause()
        return False


def do_register():
    section("Registar novo utilizador")
    username = prompt("Username")
    print("  (minimo 8 caracteres)")
    password = getpass.getpass("  -> Password: ")
    password2 = getpass.getpass("  -> Confirmar password: ")

    if password != password2:
        error("As passwords nao coincidem.")
        pause()
        return

    if not AUTH_AVAILABLE:
        warn("Modulo de autenticacao nao encontrado.")
        pause()
        return

    try:
        register_user(username, password)
        log_action("REGISTER_SUCCESS", f"username={username}")
        success(
            f"Utilizador '{username}' registado com sucesso com role 'user'!")
    except ValueError as e:
        error(str(e))
        log_action("REGISTER_FAILED", f"username={username} | reason={e}")
    except Exception as e:
        error(f"Erro inesperado: {e}")
        log_action("REGISTER_ERROR", f"username={username} | error={e}")
    pause()


def prompt_credentials_again():
    if not session["user"]:
        error("Sem utilizador autenticado.")
        return None, None

    username = session["user"]
    password = getpass.getpass("  -> Confirma a tua password: ")
    return username, password


def load_data():
    if not PANDAS_AVAILABLE:
        error("pandas nao esta instalado. Instala com: pip install pandas")
        return None, None, None

    energy, weather, features = None, None, None

    if os.path.exists(DATA_ENERGY):
        try:
            energy = pd.read_csv(DATA_ENERGY, index_col=0, parse_dates=True)
            energy.index.name = "timestamp"
            energy.index = pd.to_datetime(
                energy.index, utc=True).tz_localize(None)
        except Exception as e:
            warn(f"Erro ao carregar energy: {e}")
    else:
        warn(f"Ficheiro nao encontrado: {DATA_ENERGY}")

    if os.path.exists(DATA_WEATHER):
        try:
            weather = pd.read_csv(DATA_WEATHER, index_col=0, parse_dates=True)
            weather.index.name = "timestamp"
            weather.index = pd.to_datetime(
                weather.index, utc=True).tz_localize(None)
        except Exception as e:
            warn(f"Erro ao carregar weather: {e}")
    else:
        warn(f"Ficheiro nao encontrado: {DATA_WEATHER}")

    feat_path = DATA_FEATURES if os.path.exists(
        DATA_FEATURES) else DATA_FEATURES_FALLBACK
    if os.path.exists(feat_path):
        try:
            features = pd.read_csv(feat_path, parse_dates=["timestamp"])
            features["timestamp"] = pd.to_datetime(
                features["timestamp"]).dt.tz_localize(None)
            features = features.set_index("timestamp")
        except Exception as e:
            warn(f"Erro ao carregar features: {e}")
    else:
        warn("Ficheiro de features nao encontrado.")

    return energy, weather, features


def ask_date(energy_df):
    if energy_df is not None and not energy_df.empty:
        min_d = energy_df.index.min().date()
        max_d = energy_df.index.max().date()
        info(f"Dados disponiveis: {min_d} -> {max_d}")
    else:
        min_d = datetime.date(2025, 1, 1)
        max_d = datetime.date(2025, 12, 31)

    while True:
        raw = prompt("Data (YYYY-MM-DD)", default=str(min_d))
        try:
            d = datetime.date.fromisoformat(raw)
            if min_d <= d <= max_d:
                return d
            error(f"Data fora do intervalo ({min_d} a {max_d}).")
        except ValueError:
            error("Formato invalido. Usa YYYY-MM-DD.")


def ask_hour():
    while True:
        raw = prompt("Hora (0-23) ou 'all' para todo o dia", default="all")
        if raw == "all":
            return None
        try:
            h = int(raw)
            if 0 <= h <= 23:
                return h
            error("Hora entre 0 e 23.")
        except ValueError:
            error("Introduz um numero ou 'all'.")


def show_energy_for_day(energy_df, date, hour=None):
    section(f"Energia Electrica - {date}")
    if energy_df is None:
        warn("Dados de energia nao disponiveis.")
        return

    mask = energy_df.index.date == date
    day_data = energy_df[mask]

    if hour is not None:
        day_data = day_data[day_data.index.hour == hour]

    if day_data.empty:
        warn("Sem dados para este periodo.")
        return

    print(f"\n  {'Hora':>6}  {'Carga (MW)':>12}")
    line(".")
    for ts, row in day_data.iterrows():
        h = ts.hour
        load = row["load_mw"] if "load_mw" in row.index else row.iloc[0]
        print(f"  {h:>5}h  {load:>11.1f}")

    vals = day_data.iloc[:, 0]
    print()
    print(f"  Media : {vals.mean():.1f} MW")
    print(f"  Maximo: {vals.max():.1f} MW")
    print(f"  Minimo: {vals.min():.1f} MW")


def show_weather_for_day(weather_df, date, hour=None):
    section(f"Dados Climaticos - {date}")
    if weather_df is None:
        warn("Dados climaticos nao disponiveis.")
        return

    mask = weather_df.index.date == date
    day_data = weather_df[mask]

    if hour is not None:
        day_data = day_data[day_data.index.hour == hour]

    if day_data.empty:
        warn("Sem dados para este periodo.")
        return

    print(day_data.head(24 if hour is None else 1))


def screen_explore_day(energy_df, weather_df):
    header("Explorar Dados de um Dia")
    log_action("EXPLORE_DAY")

    date = ask_date(energy_df)
    hour = ask_hour()

    while True:
        opts = [
            "Ver energia electrica",
            "Ver dados climaticos",
            "Ver ambos",
            "Voltar ao menu anterior",
        ]
        choice = numbered_menu(opts, title=f"O que queres ver para {date}?")

        if choice == 0:
            show_energy_for_day(energy_df, date, hour)
            pause()
        elif choice == 1:
            show_weather_for_day(weather_df, date, hour)
            pause()
        elif choice == 2:
            show_energy_for_day(energy_df, date, hour)
            show_weather_for_day(weather_df, date, hour)
            pause()
        else:
            break


def screen_features(features_df, energy_df):
    header("Explorar Features")
    log_action("EXPLORE_FEATURES")

    if features_df is None:
        warn("Ficheiro de features nao disponivel.")
        pause()
        return

    date = ask_date(energy_df)
    hour = ask_hour()

    mask = features_df.index.date == date
    day_data = features_df[mask]

    if hour is not None:
        day_data = day_data[day_data.index.hour == hour]

    if day_data.empty:
        warn("Sem features para este periodo.")
        pause()
        return

    print(day_data.head(24 if hour is None else 1))
    pause()


def screen_ingestion():
    header("Ingestao de Dados")
    log_action("INGESTION_MENU")

    if not PIPELINE_AVAILABLE:
        error("Modulos de pipeline nao disponiveis.")
        pause()
        return

    try:
        info("A correr ingestao de energia (ENTSO-E)...")
        run_ingestion_raw()

        info("A correr ingestao de weather (ERA5)...")
        raw_dir = "data/raw/weather"
        output_dir = "data/processed"
        grib_path = download_era5(raw_dir)
        ingest_era5(grib_path, output_dir)

        success("Ingestao concluida com sucesso.")
        log_action("INGESTION_SUCCESS")
    except Exception as e:
        error(f"Erro na ingestao: {e}")
        log_action("INGESTION_FAILED", str(e))

    pause()


def screen_cleaning():
    header("Cleaning e Alinhamento")
    log_action("CLEANING_MENU")

    if not PIPELINE_AVAILABLE:
        error("Modulos de pipeline nao disponiveis.")
        pause()
        return

    try:
        info("A correr cleaning de energia...")
        run_processing()

        info("A unir datasets processados...")
        merge_datasets(
            energy_path=str(DATA_ENERGY),
            weather_path=str(DATA_WEATHER),
            output_path=str(DATA_MERGED)
        )

        success("Cleaning e alinhamento concluidos com sucesso.")
        log_action("CLEANING_SUCCESS")
    except Exception as e:
        error(f"Erro no cleaning/alinhamento: {e}")
        log_action("CLEANING_FAILED", str(e))

    pause()


def screen_feature_engineering():
    header("Feature Engineering")
    log_action("FEATURE_ENGINEERING_MENU")

    if not PIPELINE_AVAILABLE:
        error("Modulos de pipeline nao disponiveis.")
        pause()
        return

    try:
        if not os.path.exists(DATA_MERGED):
            error(f"Ficheiro merged nao encontrado: {DATA_MERGED}")
            error("Corre primeiro a opcao 4 - Cleaning e alinhamento.")
            log_action("FEATURE_ENGINEERING_FAILED", "merged file missing")
            pause()
            return

        info("A gerar features base (v1)...")
        generate_features(
            input_path=str(DATA_MERGED),
            output_path=str(DATA_FEATURES_FALLBACK)
        )

        info("A gerar features avancadas (v2)...")
        t0 = time.time()
        generate_advanced_features(
            input_path=str(DATA_FEATURES_FALLBACK),
            output_path=str(DATA_FEATURES)
        )
        elapsed = time.time() - t0

        success(f"Feature engineering concluido em {elapsed:.2f} segundos.")
        log_action(
            "FEATURE_ENGINEERING_SUCCESS",
            f"elapsed={elapsed:.2f}s | output={DATA_FEATURES}")

    except Exception as e:
        error(f"Erro no feature engineering: {e}")
        log_action("FEATURE_ENGINEERING_FAILED", str(e))

    pause()


def screen_training():
    header("Treino de Modelos")
    log_action("TRAINING_MENU")

    if not MODELING_AVAILABLE:
        error("Modulo de modeling nao esta disponivel.")
        pause()
        return

    if session["role"] != "admin":
        error("Apenas admins podem treinar modelos.")
        log_action("TRAINING_DENIED", "not admin")
        pause()
        return

    username, password = prompt_credentials_again()
    if not username or not password:
        pause()
        return

    try:
        start = time.time()

        if os.path.exists(DATA_FEATURES):
            dataset_path = DATA_FEATURES
            info(f"A usar features avancadas: {dataset_path}")
        elif os.path.exists(DATA_FEATURES_FALLBACK):
            dataset_path = DATA_FEATURES_FALLBACK
            warn(
                f"Features avancadas nao encontradas. A usar fallback: {dataset_path}")
        else:
            error("Nenhum ficheiro de features encontrado.")
            error("Corre primeiro a opcao 5 - Feature engineering.")
            log_action("TRAINING_FAILED", "features file missing")
            pause()
            return

        info("A carregar dataset de features...")
        info("A treinar modelos (linear + random forest)...")

        results = authenticated_train(
            username=username,
            password=password,
            dataset_path=str(dataset_path),
            target_col="load_mw",
        )

        elapsed = time.time() - start
        success(f"Treino concluido em {elapsed:.2f} segundos.")
        log_action(
            "TRAINING_SUCCESS",
            f"elapsed={elapsed:.2f}s | dataset={dataset_path}")

        print()
        print("  Resultados das metricas:")

        for model_name, res in results["results"].items():
            print(f"\n  Modelo: {model_name}")

            train_m = res["train"]
            test_m = res["test"]

            print(
                f"    Treino - MAE: {train_m['MAE']:.2f}, "
                f"RMSE: {train_m['RMSE']:.2f}, "
                f"R2: {train_m['R2']:.3f}"
            )

            print(
                f"    Teste  - MAE: {test_m['MAE']:.2f}, "
                f"RMSE: {test_m['RMSE']:.2f}, "
                f"R2: {test_m['R2']:.3f}"
            )

            print(f"    Overfitting warning: {res['overfitting_warning']}")
            print(f"    Modelo guardado em: {res['model_path']}")

    except Exception as e:
        error(f"Erro no treino de modelos: {e}")
        log_action("TRAINING_FAILED", str(e))

    pause()


def screen_metrics():
    header("Metricas e Resultados")
    log_action("METRICS_MENU")

    if not MODELING_AVAILABLE:
        error("Modulo de modeling nao esta disponivel.")
        pause()
        return

    if not session["logged_in"]:
        error("Precisas de fazer login para ver metricas.")
        log_action("METRICS_DENIED", "not logged in")
        pause()
        return

    username, password = prompt_credentials_again()
    if not username or not password:
        pause()
        return

    try:
        res = authenticated_latest_metrics(
            username=username, password=password)
        metrics = res["metrics"]

        if not metrics:
            warn("Ainda nao ha metricas guardadas. Treina primeiro os modelos.")
            pause()
            return

        print()
        for model_name, entries in metrics.items():
            print(f"\n  Modelo: {model_name}")
            for entry in entries[-3:]:
                ts = entry.get("timestamp", "?")
                m_train = entry["metrics"]["train"]
                m_test = entry["metrics"]["test"]
                over = entry["metrics"].get("overfitting_warning", False)
                print(f"    [{ts}]")
                print(
                    f"      Treino - MAE: {m_train['MAE']:.2f}, RMSE: {m_train['RMSE']:.2f}, R2: {m_train['R2']:.3f}")
                print(
                    f"      Teste  - MAE: {m_test['MAE']:.2f}, RMSE: {m_test['RMSE']:.2f}, R2: {m_test['R2']:.3f}")
                print(f"      Overfitting warning: {over}")

        log_action("METRICS_VIEW")
    except Exception as e:
        error(f"Erro ao carregar metricas: {e}")
        log_action("METRICS_VIEW_FAILED", str(e))

    pause()


def choose_model_file():
    if not os.path.exists(MODELS_DIR):
        warn("Diretorio 'models' nao existe. Treina modelos primeiro.")
        return None

    files = sorted([f for f in os.listdir(
        MODELS_DIR) if f.endswith(".joblib")])
    if not files:
        warn("Nao ha modelos treinados. Treina modelos primeiro.")
        return None

    section("Modelos disponiveis")
    for i, f in enumerate(files, 1):
        print(f"    {i}. {f}")

    raw = input("\n  Escolhe modelo (numero): ").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(files):
            return os.path.join(MODELS_DIR, files[idx])
    except ValueError:
        pass

    warn("Escolha invalida.")
    return None


def screen_prediction():
    header("Gerar Previsao")
    log_action("PREDICTION_MENU")

    if not MODELING_AVAILABLE:
        error("Modulo de modeling nao esta disponivel.")
        pause()
        return

    if not session["logged_in"]:
        error("Precisas de fazer login para gerar previsoes.")
        log_action("PREDICTION_DENIED", "not logged in")
        pause()
        return

    username, password = prompt_credentials_again()
    if not username or not password:
        pause()
        return

    model_path = choose_model_file()
    if not model_path:
        pause()
        return

    try:
        n_last_str = prompt(
            "Numero de ultimas horas a usar para previsao",
            default="24")
        try:
            n_last = int(n_last_str)
        except ValueError:
            n_last = 24

        start = time.time()
        features_subset = load_default_features_subset(
            n_last=n_last,
            dataset_path=DATA_FEATURES
        )
        features_subset = features_subset.reset_index()

        result = authenticated_predict_from_features(
            username=username,
            password=password,
            model_path=model_path,
            features_df=features_subset,
        )
        elapsed = time.time() - start

        success(f"Previsoes geradas com sucesso em {elapsed:.4f} segundos.")
        log_action(
            "PREDICTION_SUCCESS",
            f"model={model_path} | n={result['n_samples']} | elapsed={elapsed:.4f}s")

        print()
        print(f"  Modelo: {os.path.basename(model_path)}")
        print(f"  Numero de pontos: {result['n_samples']}")
        print()
        print("  Primeiras previsoes:")
        for ts, pred in list(zip(result["timestamps"], result["predictions"]))[
                :10]:
            print(f"    {ts}  ->  {pred:.2f} MW")

        if elapsed > 1.0:
            warn("A previsao ultrapassou 1 segundo localmente.")
        else:
            success("Tempo de resposta inferior a 1 segundo.")
    except Exception as e:
        error(f"Erro ao gerar previsoes: {e}")
        log_action("PREDICTION_FAILED", str(e))

    pause()


def screen_promote_user():
    header("Promover Utilizador a Admin")
    log_action("PROMOTE_MENU")

    target_user = prompt("Username do utilizador a promover")

    if not target_user:
        error("Tens de indicar um username.")
        pause()
        return

    try:
        result = promote_to_admin(session["user"], target_user)

        if result is True:
            success(
                f"Utilizador '{target_user}' promovido a admin com sucesso.")
            log_action("PROMOTE_SUCCESS", f"target_user={target_user}")
        else:
            warn(f"O utilizador '{target_user}' ja era admin.")
            log_action("PROMOTE_SKIPPED", f"target_user={target_user}")
    except ValueError as e:
        error(str(e))
        log_action("PROMOTE_FAILED", f"target_user={target_user} | reason={e}")
    except PermissionError as e:
        error(str(e))
        log_action("PROMOTE_DENIED", f"target_user={target_user} | reason={e}")

    pause()


def screen_user_menu(energy_df, weather_df, features_df):
    while True:
        energy_df, weather_df, features_df = load_data()
        clear()
        header(f"Menu User  [utilizador: {session['user']}]")

        opts = [
            "Explorar dados de um dia (energia + clima)",
            "Explorar features calculadas",
            "Gerar previsao",
            "Ver metricas / resultados",
            "Logout",
        ]

        for i, opt in enumerate(opts, 1):
            print(f"    {i}. {opt}")
        print()

        raw = input("  -> ").strip()
        try:
            choice = int(raw) - 1
        except ValueError:
            continue

        if choice == 0:
            screen_explore_day(energy_df, weather_df)
        elif choice == 1:
            screen_features(features_df, energy_df)
        elif choice == 2:
            screen_prediction()
        elif choice == 3:
            screen_metrics()
        elif choice == 4:
            log_action("LOGOUT")
            session["user"] = None
            session["logged_in"] = False
            session["role"] = None
            success("Sessao terminada.")
            time.sleep(1)
            break


def screen_admin_menu(energy_df, weather_df, features_df):
    while True:
        energy_df, weather_df, features_df = load_data()
        clear()
        header(f"Menu Admin  [utilizador: {session['user']}]")

        opts = [
            "Explorar dados de um dia (energia + clima)",
            "Explorar features calculadas",
            "Ingestao de dados",
            "Cleaning e alinhamento",
            "Feature engineering",
            "Treinar modelos",
            "Ver metricas / resultados",
            "Gerar previsao",
            "Promover utilizador a admin",
            "Logout",
        ]

        for i, opt in enumerate(opts, 1):
            print(f"    {i}. {opt}")
        print()

        raw = input("  -> ").strip()
        try:
            choice = int(raw) - 1
        except ValueError:
            continue

        if choice == 0:
            screen_explore_day(energy_df, weather_df)
        elif choice == 1:
            screen_features(features_df, energy_df)
        elif choice == 2:
            screen_ingestion()
        elif choice == 3:
            screen_cleaning()
        elif choice == 4:
            screen_feature_engineering()
        elif choice == 5:
            screen_training()
        elif choice == 6:
            screen_metrics()
        elif choice == 7:
            screen_prediction()
        elif choice == 8:
            screen_promote_user()
        elif choice == 9:
            log_action("LOGOUT")
            session["user"] = None
            session["logged_in"] = False
            session["role"] = None
            success("Sessao terminada.")
            time.sleep(1)
            break


def main():
    energy_df, weather_df, features_df = load_data()

    while True:
        screen_welcome()
        raw = input("  -> ").strip()

        if raw == "1":
            ok = do_login()
            if ok:
                if session["role"] == "admin":
                    screen_admin_menu(energy_df, weather_df, features_df)
                else:
                    screen_user_menu(energy_df, weather_df, features_df)

        elif raw == "2":
            do_register()

        elif raw == "3":
            clear()
            print("\n  Ate logo!\n")
            sys.exit(0)

        else:
            error("Opcao invalida.")
            time.sleep(1)


if __name__ == "__main__":
    main()
