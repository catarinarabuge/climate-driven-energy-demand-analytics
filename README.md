# Climate-Driven Energy Demand Analytics System

Academic group project developed for the Project in Artificial Intelligence and Data Science course.

A modular AI-driven software system that uses publicly available climate data to model and predict national electricity demand in **Portugal**. Built for the PIACD 2025/2026 course at the Department of Informatics Engineering, University of Coimbra.

The system integrates real-world hourly electricity load data from the **ENTSO-E Transparency Platform** with **ERA5 climate data** from the Copernicus Climate Data Store, exposes prediction functionality through both a Command Line Interface and a Flask web application, and enforces authentication on all sensitive operations.

---

## 1. Research Question

**How can publicly available climate data explain and predict national electricity demand in Portugal?**

Sub-questions:
- To what extent do temperature, solar radiation, and wind speed influence electricity load?
- Can climate variables improve short-term electricity demand forecasting accuracy?
- Which engineered features contribute most to predictive performance?

---

## 2. Selected Country and Data Sources

- **Country:** Portugal
- **Electricity demand:** ENTSO-E Transparency Platform — total hourly electricity load (MW), one full year of 2025 data
- **Climate data:** ERA5 (Copernicus Climate Data Store) — 2-meter temperature, 10-meter wind component, surface solar radiation, total precipitation, all at hourly resolution
- **Alignment:** climate and electricity data are merged on the same hourly timestamps in UTC

---

## 3. Running the System

The system has two interfaces — a **Flask web application** (`app.py`) and a **Command Line Interface** (`cli.py`) — and two user roles (**regular user** and **admin**) with different capabilities.

### 3.1 Initial Setup (do once)

This section walks you through everything you need to do once before launching the system. By the end, you'll be able to run either the web app or the CLI.

#### Step 1 — Install Python

If you don't already have Python installed, download Python 3.11 or newer from [python.org/downloads](https://www.python.org/downloads/). During installation on Windows, make sure to tick the box that says **"Add Python to PATH"**.

To check it's installed, open a terminal and run:

```bash
python --version
```

You should see something like `Python 3.11.x`. If you get an error, restart your terminal or computer after installing.

#### Step 2 — Get the code

Open a terminal in the folder where you want to keep the project, then run:

```bash
git clone <repo-url>
cd pl1g3
```

If you don't have `git`, you can also download the project as a ZIP file from GitLab and unzip it.

#### Step 3 — Install the project's dependencies

The project relies on several Python libraries (Flask for the web app, pandas for data handling, scikit-learn for models, etc.). Install them all in one go:

```bash
pip install -r requirements.txt
```

This may take a couple of minutes the first time. You only need to do it once.

#### Step 4 — Create a `.env` file (optional)

Some parts of the system need configuration values that should not be saved in the public code (like API keys). These go in a file called `.env` at the root of the project.

**You don't need to set anything to use the web app or the CLI as a regular user or admin.** All the data is already in the `data/processed/` folder, the models are in `models/`, and the system has a working development default for everything else.

You only need a `.env` file if you want to run the **data ingestion** step, which downloads fresh data from the internet. In that case, create a file called exactly `.env` in the project root and add:

```
ENTSOE_API_KEY=your_entsoe_token_here
CDSAPI_KEY=your_uid:your_key_here
```

These are free tokens you can request from:
- **ENTSO-E**: register at https://transparency.entsoe.eu/, then request an API token in your account settings
- **Copernicus** (for climate data): register at https://cds.climate.copernicus.eu/, then find your UID and API key on your profile page

If you just want to log in, view data, train models, or generate predictions, you can skip this step entirely.

#### Step 5 — (Optional) Create an admin account

By default the system has no admin user. If you want to test the admin features (running the pipeline, training models, viewing metrics, promoting users), create one with:

```bash
python create_admin.py
```

If you don't run this, you can still create regular-user accounts through the normal registration form.

#### You're ready. Now choose how to use the system:

**To launch the web app:**

```bash
python app.py
```

Then open your browser at **http://127.0.0.1:5001**. Stop the server with `Ctrl+C` in the terminal.

**To launch the CLI:**

```bash
python cli.py
```

The CLI runs in your terminal — no browser needed. Stop it with `Ctrl+C` or the "Sair" option.

Both interfaces share the same users, the same trained models, and the same activity log. An account created in one works in the other.

---

### 3.2 Running as a Regular User

A regular user is a non-admin account. After registering through either the landing page or the CLI registration menu, regular users have read-only access to data exploration and the prediction interface.

**Through the web app:**

1. Open `http://127.0.0.1:5001` and click **Registar** (Register) to create an account, or **Login** if you already have one.
2. After logging in, you are taken to the **Painel de Previsão** (forecast page) at `/forecast`. This is the regular-user prediction interface.
3. On the prediction page:
   - Select which features you want to control through clickable chips (hour, temperature, season, wind, etc.). Any feature you don't select uses a historical default.
   - Adjust values through sliders, toggles, and presets.
   - Click **Gerar Previsão** to generate a prediction. The system returns a point prediction, a 24-hour load curve, and a 90% confidence interval.
4. Log out through the **Sair** button in the top right.

**Through the CLI:**

```bash
python cli.py
```

The welcome screen offers three options: **Login**, **Registar novo utilizador**, or **Sair**. After logging in as a regular user you land in the user menu, which offers:

1. **Explorar dados de um dia (energia + clima)** — view raw energy and weather data for a chosen date and hour
2. **Explorar features calculadas** — view the engineered features that feed into the model
3. **Gerar previsão** — pick a saved model, choose the number of recent hours to predict on, and view the predictions (the CLI prompts for your password as a re-authentication step and warns if the prediction takes more than 1 second)
4. **Ver métricas / resultados** — view the historical metrics of all training runs (MAE, RMSE, R²) — also requires password re-confirmation
5. **Logout**

---

### 3.3 Running as an Admin

An admin has all the regular-user capabilities plus the ability to drive the full data pipeline, train models, and promote other users. New registrations always create regular users; admins must be either seeded (`create_admin.py`) or promoted by another admin.

**Through the web app:**

1. Open `http://127.0.0.1:5001` and log in with admin credentials.
2. After logging in, you are taken to `/dashboard`, which serves the **Painel de Administração** with ten action cards:
   1. **Explorar Dados** — energy and climate data exploration for a chosen date/hour
   2. **Explorar Features** — view the engineered features
   3. **Ingestão de Dados** — run the ENTSO-E + ERA5 ingestion (requires `.env` with API keys)
   4. **Cleaning & Alinhamento** — clean and merge raw datasets
   5. **Feature Engineering** — generate base (v1) and advanced (v2) features
   6. **Treinar Modelos** — train Linear Regression + Random Forest (requires password re-confirmation)
   7. **Ver Métricas** — view stored metrics from previous training runs (requires password re-confirmation)
   8. **Gerar Previsão** — opens the scenario-based prediction page
   9. **Promover a Admin** — promote another user to admin
   10. **Logout**

**Through the CLI:**

```bash
python cli.py
```

Log in as an admin. The admin menu offers all the regular-user options plus pipeline and admin-only actions:

1. **Explorar dados de um dia (energia + clima)**
2. **Explorar features calculadas**
3. **Ingestão de dados** — run the full ingestion pipeline
4. **Cleaning e alinhamento**
5. **Feature engineering**
6. **Treinar modelos** — admin only, prompts for password re-confirmation, trains both models and shows train/test MAE, RMSE, R², and the overfitting flag
7. **Ver métricas / resultados**
8. **Gerar previsão**
9. **Promover utilizador a admin**
10. **Logout**

---

## 4. System Architecture

The system is built as modular components with a clear separation of concerns:

- **Data Ingestion** — retrieves raw data from ENTSO-E and ERA5
- **Cleaning & Alignment** — produces consistent hourly time series
- **Feature Engineering** — generates base and advanced predictive features
- **Modeling** — trains Linear Regression (baseline) and a regularised Random Forest
- **Prediction Interface** — exposes predictions through CLI and web
- **Authentication Layer** — gates all sensitive operations
- **Application Service Layer** — mediates between user interfaces and the modeling subsystem
- **CLI** and **Flask Web App** — two complementary interaction layers

For the full architecture, see [`Architecture/ARCHITECTURE.md`](Architecture/ARCHITECTURE.md).

---

## 5. Repository Structure

```text
.
├── Code/                       # Source code (auth, ingestion, cleaning, features, modeling, interface)
├── Architecture/               # ARCHITECTURE.md
├── Requirements/               # USE_CASES.md, QUALITY_ATTRIBUTES.md
├── Design/                     # Web templates (admin.html, user.html, landing_page.html)
├── Testing/                    # Unit tests (pytest)
├── data/
│   ├── raw/{energy,weather}/   # Original datasets
│   └── processed/              # Cleaned, aligned, and engineered datasets
├── models/                     # Trained model artifacts + model_metrics.json
├── Management/                 # Team profiles
├── app.py                      # Flask web application
├── cli.py                      # Command line interface
├── conftest.py                 # Pytest configuration
├── create_admin.py             # Development helper to seed an admin account
├── users.json                  # Hashed user credentials
├── system_actions.log          # Logged system events
├── .gitlab-ci.yml              # CI pipeline (5 stages)
└── README.md
```

---

## 6. Documentation Map

- **System architecture and components** → [`Architecture/ARCHITECTURE.md`](Architecture/ARCHITECTURE.md)
- **Structured use cases** → [`Requirements/USE_CASES.md`](Requirements/USE_CASES.md)
- **Structured quality attributes** → [`Requirements/QUALITY_ATTRIBUTES.md`](Requirements/QUALITY_ATTRIBUTES.md)

---

## 7. Performance Measurements

The system measures execution time for the most relevant components and enforces a strict latency budget on predictions:

- **Advanced feature engineering** — logs total duration to `system_actions.log`
- **Model training** — logged in both CLI (`elapsed=X.XXs`) and as part of the web `/api/train` JSON response
- **Prediction (CLI and web)** — measured per request; the CLI warns when latency exceeds 1 second
- **Automated 1-second budget enforcement** — the GitLab CI `prediction_performance` job re-runs an end-to-end timing check on every push and fails the build if a prediction takes 1 second or more

Detailed performance scenarios and limitations are documented in [`Requirements/QUALITY_ATTRIBUTES.md`](Requirements/QUALITY_ATTRIBUTES.md) §2.

---

## 8. Security Considerations

The system enforces several security measures:

- **Password hashing** with bcrypt, never plaintext, with an 8-character minimum length enforced at registration
- **No hardcoded credentials** in the repository; API keys and the Flask `SECRET_KEY` are loaded from environment variables; `.env` is git-ignored
- **Authentication required** for every sensitive operation: pipeline execution, training, metrics inspection, prediction, and admin promotion
- **Role-based access control** (user vs admin) on both CLI and web
- **Password re-confirmation** on the web app before training and metrics inspection, on top of the session cookie
- **Static security scanning** with `bandit` runs in CI on every push
- **Action logging** in `system_actions.log` records authentication attempts and all modeling actions with timestamps and usernames

Known limitations: the development fallback `SECRET_KEY`, the basic level of input validation, and the lack of session-token authentication are documented in [`Requirements/QUALITY_ATTRIBUTES.md`](Requirements/QUALITY_ATTRIBUTES.md) §6.

---

## 9. Reliability Strategies

The system handles failures gracefully:

- **Missing or incomplete data** → forward/backward filling, duplicate removal, full-hourly reindexing
- **External API failures** → caught and logged; no internal stack traces exposed
- **Invalid authentication** → access denied with a generic message; attempt logged
- **Optional dependencies** (`cdsapi`, `cfgrib`) → defensive imports so the system runs even when not installed
- **Unit tests** cover normal paths, edge cases, and failure scenarios (missing files, empty datasets, missing columns, invalid credentials, missing API keys)
- **GitLab CI** validates every push with five stages: unit tests, code style/complexity, smoke imports, coverage (≥85% on `Code/`), security scanning, and prediction-latency enforcement. The build fails if any stage fails.

---

## 10. Key Findings

The system was trained and evaluated end-to-end on the 2025 Portuguese electricity demand dataset aligned with ERA5 climate data. Latest results stored in `models/model_metrics.json`:

| Model                       | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² | Overfitting flag |
|-----------------------------|-----------|------------|----------|----------|-----------|---------|------------------|
| Linear Regression           | 212.87    | 282.66     | 0.928    | 256.79   | 319.19    | 0.931   | No               |
| Random Forest (regularised) | 164.50    | 278.05     | 0.930    | 342.16   | 486.98    | 0.839   | No               |

*All errors in MW; R² unitless.*

**Key observations:**
- Linear Regression is the more accurate model on test data in the current configuration (test MAE ≈ 257 MW, test R² ≈ 0.93). The train/test metrics are very close, indicating clean generalisation.
- The Random Forest has been strongly regularised (`max_depth=12`, `min_samples_leaf=50`, `min_samples_split=100`, `max_features='sqrt'`). Earlier unconstrained configurations achieved lower test errors (~112 MW MAE) but were flagged by the overfitting heuristic because train MAE was four times smaller than test MAE. The current configuration trades absolute accuracy for a "no overfitting" verdict.
- Climate-aware features (rolling temperature, lagged load, season encoding, feels-like temperature) are the principal driver of predictive power: the linear baseline alone reaches a test R² above 0.93 using only the engineered feature set.
- The time-aware split (no shuffling) ensures the reported test errors reflect true forward-prediction behaviour, not in-sample fitting.

---

## 11. Git Workflow

- No direct commits to `main`
- Each issue is implemented in a branch named `<issueID>-short-description`
- Every change goes through a Merge Request reviewed by at least one teammate before merging
- The GitLab CI pipeline must pass before a merge is accepted

---

