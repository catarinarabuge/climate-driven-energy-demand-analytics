# Quality Attributes Specification
## Climate-Driven Energy Demand Analytics System

---

## 1. Overview

This document defines the **non-functional requirements** of the Climate-Driven Energy Demand Analytics System.

The system focuses on electricity demand modeling for **Portugal**, integrating climate data to support future prediction tasks.

All system operations are designed to be **fully reproducible through code-based execution**, ensuring consistency, traceability, and scientific validity.

The quality attributes are aligned with:
- The project specification requirements
- The system architecture

Each quality attribute is defined using structured scenarios:

> **Source → Stimulus → Artifact → Environment → Response → Response Measure**

In addition, the system provides two interaction layers — a **CLI interface** and a **Flask web application** — which enable user-triggered execution of system operations and introduce interaction-driven scenarios that are reflected in the quality attributes below.

---

## 2. Performance

The system must ensure efficient execution of the data pipeline and meet strict performance constraints for future prediction tasks.

---

### Scenario 2.1 – Data Ingestion Execution

**Source:** Admin user
**Stimulus:** Execution of the data ingestion pipeline
**Artifact:** Data Ingestion Module (`entsoe_client.py`, `era5.py`)
**Environment:** Local execution environment with valid API credentials

**Response:**
- The system retrieves electricity demand data from ENTSO-E and climate data from ERA5
- The number of downloaded records is reported to the user
- The outcome (success or failure) is recorded in `system_actions.log`

**Response Measure:**
- The ingestion run completes and reports the number of records downloaded (e.g. `Registos descarregados: <N>`)
- The outcome is logged as `INGESTION_SUCCESS` or `INGESTION_FAILED` in `system_actions.log`
- Ingestion is an offline, one-off batch operation and is not bound by the 1-second budget that applies to prediction. Its execution time depends primarily on external API response time — ERA5 requests in particular are queued server-side by Copernicus and may take several minutes

---

### Scenario 2.2 – Feature Engineering Execution Time

**Source:** System / Admin User (via CLI)
**Stimulus:** Execution of feature engineering pipeline
**Artifact:** Feature Engineering Module (`base_features.py`, `advanced_features.py`)
**Environment:** Local execution with processed dataset available

**Response:**
- The system generates all base and advanced features
- Execution progress is shown via CLI
- The advanced feature stage measures and reports its total duration through the application logger

**Response Measure:**
- Advanced feature execution time is reported as `Advanced features guardadas em <path> (<elapsed>s)` and the outcome of the overall feature engineering step is logged in `system_actions.log` (`FEATURE_ENGINEERING_SUCCESS` / `FEATURE_ENGINEERING_FAILED`)

---

### Scenario 2.3 – Model Training Execution Time

**Source:** Authenticated Admin User
**Stimulus:** Triggering of model training via CLI (`screen_training`) or via the web route `POST /api/train`
**Artifact:** Modeling Component (`Code/modeling/model_training.py`) routed through the Application Service Layer (`Code/interface/prediction_service.authenticated_train`)
**Environment:** Local execution with feature dataset and admin credentials

**Response:**
- The Linear Regression baseline and the regularised Random Forest Regressor are trained
- The CLI screen `screen_training` measures elapsed time around the call to `authenticated_train`; the web route returns the same elapsed value in its JSON response
- Trained models are persisted to `models/`
- Train and test metrics are returned and displayed to the user

**Response Measure:**
- Training duration is reported in seconds in the CLI output / web JSON response and logged in `system_actions.log` as `action=TRAINING_SUCCESS | elapsed=<X.XX>s`

---

### Scenario 2.4 – Prediction Response Time

**Source:** Authenticated User (CLI: any role; Web: any role, through the `/forecast` prediction page)
**Stimulus:** Submission of a prediction request via the CLI (`screen_prediction`) or via the web route `POST /predict`
**Artifact:** Prediction logic — `Code/modeling/predict.py` for the CLI flow (routed through `Code/interface/prediction_service.authenticated_predict_from_features`), and the `POST /predict` route in `app.py` for the web flow
**Environment:** Local execution with at least one trained model in `models/`

**Response:**
- The system loads a trained model with `joblib` — the CLI lets the user choose the model file; the web route automatically loads the most recent Random Forest model
- The system aligns the input features to the model's `feature_names_in_`
- The system generates the prediction(s) for the requested input
- Results are returned to the user (CLI screen output, or a JSON response rendered by the `/forecast` page)
- Elapsed time is measured around the call

**Response Measure:**
- Prediction response time **< 1 second** in the local execution environment; the CLI explicitly checks `if elapsed > 1.0` and warns the user when the threshold is exceeded
- Each CLI prediction event is logged in `system_actions.log` as `action=PREDICTION_SUCCESS | elapsed=<X.XXXX>s`; each web prediction is logged as `action=PREDICT_SUCCESS` with the model used and the predicted value

---

### Scenario 2.5 – Automated Prediction Latency Enforcement in CI

**Source:** Continuous Integration pipeline (GitLab CI)
**Stimulus:** A new commit is pushed to any branch that triggers the pipeline
**Artifact:** GitLab CI `prediction_performance` job in the `performance` stage of `.gitlab-ci.yml`
**Environment:** Ephemeral CI runner with the project dependencies installed

**Response:**
- The job creates a temporary directory, generates a synthetic 240-row feature dataset, and trains both the Linear Regression baseline and the regularised Random Forest end-to-end
- The job calls `load_model_and_predict` for the most recent random-forest artifact and measures the elapsed time
- The job asserts `elapsed < 1.0` and fails the build if the assertion is not satisfied

**Response Measure:**
- The build fails if a synthetic-dataset prediction takes 1 second or more, providing automated, per-push enforcement of the spec §3.3.1 prediction-latency requirement that no longer depends on a developer running the CLI manually

---

## 3. Reliability

The system ensures robustness and fault tolerance, preventing crashes and ensuring stable operation under unexpected conditions. The automated test suite and CI pipeline that protect this robustness against regressions are documented under §5 Testability.

---

### Scenario 3.1 – Handling Missing or Incomplete Data

**Source:** Raw energy dataset
**Stimulus:** The raw ENTSO-E file contains duplicate timestamps, missing hours, or null load values
**Artifact:** Energy cleaning logic (`clean_and_align_energy` in `Code/ingestion/entsoe_client.py`)
**Environment:** Cleaning stage of the data pipeline

**Response:**
- Timestamps are parsed to UTC; any value that cannot be parsed becomes `NaT`
- Duplicate timestamps are detected and removed
- The series is reindexed against a complete hourly index covering all of 2025
- Remaining null load values are filled

**Response Measure:**
- **Duplicate timestamps:** removed with `df.index.duplicated(keep="first")` — when two or more rows share a timestamp, only the first is kept and the rest are discarded
- **Missing hours:** the dataset is reindexed onto a full hourly `date_range` from `2025-01-01 00:00` to `2025-12-31 23:00` UTC, so every one of the 8760 hours of 2025 becomes a row; hours absent from the raw data appear as rows with `NaN` in `load_mw`
- **Null load values:** filled by forward-fill (`ffill`) and then backward-fill (`bfill`), so a missing hour takes the value of the most recent known hour, and any missing values at the very start of the series take the first known value afterwards
- The resulting `energy_load_PT_2025_hourly_clean.csv` contains exactly 8760 rows, one per hour, with zero null values in `load_mw`
- An empty input dataframe is rejected upfront with a `ValueError("Input dataframe is empty")` rather than producing a corrupt output

---

### Scenario 3.2 – Handling External API Failures

**Source:** External data source (ENTSO-E or ERA5)
**Stimulus:** The API is unreachable, or the required API key is missing or invalid
**Artifact:** Data Ingestion Module (`entsoe_client.py`, `era5.py`)
**Environment:** Ingestion stage, triggered from the CLI or the web route `POST /api/pipeline/ingestion`

**Response:**
- The failure is caught before it can crash the process
- The cause is logged
- A controlled, readable message is shown to the user

**Response Measure:**
- **Missing ENTSO-E key:** `load_api_key()` raises `RuntimeError` with the message "ENTSOE_API_KEY não definida..." instructing the user to create a `.env` file — the pipeline stops cleanly before any network call
- **Missing optional dependency:** if `cdsapi` or `cfgrib` is not installed, an `ImportError` with an explicit "instala com: pip install ..." message is raised instead of an unhandled crash
- **Any ingestion failure on the web route:** the exception is caught, `action=INGESTION_FAILED` is written to `system_actions.log` together with the exception message, and the route returns an HTTP 500 JSON payload `{"error": <message>}` — no Python traceback is sent to the browser
- **Any ingestion failure on the CLI:** the exception is caught, `INGESTION_FAILED` is logged, and a single `[ERROR]` line is printed to the terminal 

---

### Scenario 3.3 – Handling Invalid User Authentication

**Source:** A user (unregistered, or entering a wrong password)
**Stimulus:** Submission of invalid login credentials
**Artifact:** Authentication Module (`Code/auth/auth_service.py`)

**Response:**
- The credentials are checked against the stored bcrypt hash
- Access is denied on mismatch
- The failed attempt is logged
- A generic message is shown

**Response Measure:**
- **Unknown username:** `authenticate_user` returns `False` without revealing whether the username exists
- **Wrong password:** `bcrypt.checkpw` returns `False`; access is denied
- **User-facing message:** the user sees a generic "Credenciais inválidas" / "Credenciais inválidas." message that does not distinguish between "no such user" and "wrong password", so an attacker cannot enumerate valid usernames
- **Logging:** the failed attempt is written to `system_actions.log` as `action=LOGIN_FAILED` with the attempted username
- No Python traceback or internal detail is exposed to the user in either interface

---

### Scenario 3.4 – Data Alignment Robustness

**Source:** Cleaned energy and weather datasets
**Stimulus:** The two datasets do not share perfectly identical timestamps
**Artifact:** Dataset merging logic (`Code/cleaning/merge_datasets.py`)

**Response:**
- Both timestamp columns are normalised to a common tz-naive UTC representation
- The datasets are joined by nearest-timestamp matching within a bounded tolerance

**Response Measure:**
- **Matching:** `pd.merge_asof` is used with `direction="nearest"` and `tolerance=pd.Timedelta("1h")` — each energy row is paired with the weather row whose timestamp is closest, provided it is within one hour
- **Unmatched rows:** an energy row with no weather observation within the 1-hour tolerance is kept, but its weather columns are left as `NaN` rather than dropped, so no energy observation is lost
- **Missing input files:** if either `energy_path` or `weather_path` does not exist, a `FileNotFoundError` is raised with the offending path before any processing begins
- The merged output is written to `merged.csv` with one row per energy timestamp

---

### Scenario 3.5 – Logging and Traceability

**Source:** System / any user action through the CLI or web app
**Stimulus:** Execution of any system operation (login, ingestion, cleaning, training, prediction, etc.)
**Artifact:** Logging system (`system_actions.log`)

**Response:**
- Each operation writes a structured entry to the shared log file

**Response Measure:**
- Every logged line follows the format `<timestamp> [LEVEL] user=<username> | role=<role> | action=<ACTION> | <extra>` (CLI) or `<timestamp> [LEVEL] user=<username> | action=<ACTION> | <extra>` (web)
- Each operation produces a paired outcome entry — for example a successful training run logs `action=TRAINING_SUCCESS | elapsed=<X.XX>s`, and a failed one logs `action=TRAINING_FAILED` with the exception message
- Because the timestamp, username, and action are always present, any operation can be traced after the fact to who triggered it, when, and whether it succeeded

---

## 4. Security

The system ensures protection of sensitive data and enforces secure access control.

---

### Scenario 4.1 – Secure Password Storage

**Source:** User  
**Stimulus:** Registration or login  
**Artifact:** Authentication Module (`auth_service.py`)  

**Response:**
- Passwords are hashed using **bcrypt**
- Passwords are never stored in plaintext
- Only hashed values are persisted

**Response Measure:**
- Zero plaintext passwords exist in system storage  

---

### Scenario 4.2 – Credential Protection

**Source:** Developer  
**Stimulus:** Configuration of API credentials  
**Artifact:** Environment configuration (`.env`)  

**Response:**
- Credentials stored in environment variables
- `.env` file excluded via `.gitignore`

**Response Measure:**
- No credentials exposed in repository  

---

### Scenario 4.3 – Input Validation

**Source:** A user registering or logging in
**Stimulus:** Submission of malformed or invalid input
**Artifact:** Registration and authentication logic (`Code/auth/auth_service.py`), plus the web form handlers in `app.py`

**Response:**
- Inputs are validated before any account is created or any hash is computed
- Invalid inputs are rejected with a specific error

**Response Measure:**
- **Empty username:** `register_user` raises `ValueError("Username cannot be empty")` and no account is created
- **Password shorter than 8 characters:** `register_user` raises `ValueError("Password must be at least 8 characters")`; on the web form, registration is rejected and the message "A password deve ter pelo menos 8 caracteres." is shown
- **Mismatched password confirmation (web):** the `/register` handler rejects the submission with "As passwords não coincidem." before calling `register_user`
- **Duplicate username:** `register_user` raises `ValueError("User already exists")` so an existing account is never silently overwritten
- In every case the input is rejected with a controlled `ValueError` and a human-readable message; no unhandled exception or stack trace reaches the user 

---

### Scenario 4.4 – Access Control Enforcement

**Source:** A user (authenticated or not) attempting a protected operation
**Stimulus:** An attempt to trigger pipeline execution, model training, prediction, metrics inspection, or admin promotion
**Artifact:** Authentication Layer + Application Service Layer (`Code/interface/prediction_service.py::_ensure_authenticated`) + Flask web helpers (`app.py::require_admin`, `app.py::confirm_password`)

**Response:**
- Every protected operation is gated by an authentication and role check before it executes
- Unauthorized attempts are rejected and recorded
- On the web, sensitive admin operations require a fresh password confirmation in addition to the session cookie

**Response Measure:**
- An unauthenticated or under-privileged attempt never reaches the protected operation: it raises `AuthorizationError` in the CLI, or returns an HTTP 403 JSON response in the web app
- Every rejected attempt is recorded in `system_actions.log` (e.g. `TRAINING_DENIED`, `PREDICTION_DENIED`, `METRICS_DENIED`)
- Sensitive web routes additionally reject the request with HTTP 403 if the password re-confirmation (`confirm_password()`) fails, so a valid session cookie alone is not sufficient — this mitigates session hijacking on shared workstations
- The full list of which operations are protected and which interface triggers each one is documented in `USE_CASES.md`

---

## 5. Testability

The system is designed for high testability due to its modular structure.

---

### Scenario 5.1 – Modular Unit Testing

**Source:** Developer
**Stimulus:** Execution of unit tests
**Artifact:** Individual modules

**Response:**
- Each module is tested independently:
  - ingestion (`test_ingestion.py`, `test_era5.py`) – including `monkeypatch`-driven fakes for `EntsoePandasClient`, `cdsapi`, and `cfgrib`, so external APIs are never called from the test suite
  - cleaning (`test_cleaning.py`)
  - feature engineering (`test_features.py`)
  - authentication (`test_auth.py`)
  - evaluation (`test_evaluation.py`)
  - model training (`test_model_training.py`)
  - prediction (`test_predict.py`)

**Response Measure:**
- The CI `test_coverage` job runs pytest with `--cov=Code --cov-report=term-missing --cov-report=xml --cov-fail-under=85`. The build fails if total coverage on `Code/` drops below 85 %; `coverage.xml` is exported as an artifact

---

### Scenario 5.2 – Smoke-Import Regression Detection

**Source:** Continuous Integration pipeline
**Stimulus:** A new commit is pushed
**Artifact:** GitLab CI `smoke_imports` job in the `integration` stage of `.gitlab-ci.yml`

**Response:**
- The job runs `python -c "import …"` for every top-level module of the project: `app`, `cli`, `Code.auth.auth_service`, `Code.ingestion.entsoe_client`, both feature modules, the modeling subpackage, and `Code.interface.prediction_service`
- Any import-time failure (syntax error, broken `import`, accidental side-effect) immediately fails the pipeline before the rest of the suite runs

**Response Measure:**
- Import-level regressions are detected within seconds of being pushed and never reach the full test, security, or performance stages

---

### Scenario 5.3 – Failure Scenario Testing

**Source:** Test suite
**Stimulus:** Execution of edge and failure cases
**Artifact:** System modules

**Response:**
- Tests validate:
  - missing files (ingestion, ERA5, cleaning, training, prediction)
  - invalid credentials (authentication)
  - empty datasets (cleaning, features)
  - missing columns (features, model training)
  - missing API keys (ingestion)
  - missing trained model files (prediction)
  - non-admin users attempting admin actions (auth)

**Response Measure:**
- Robustness validated under failure conditions; failures raise controlled exceptions instead of crashes

---

## 6. Limitations and Assumptions

- System designed for a **local execution environment**
- The Flask web app uses a fallback `SECRET_KEY` when no environment variable is provided; for production deployment a strong key must be configured via the `SECRET_KEY` environment variable
- The development helper `create_admin.py` resets the default admin account and is intended only for local seeding
- Both the CLI and the Flask web app expose the full pipeline (ingestion, cleaning, features, training, metrics, prediction). The regular-user web prediction endpoint `POST /predict` runs the most recent trained model; unlike the admin modeling routes, it does not require a password re-confirmation, since it is intended for low-risk, read-only scenario exploration by regular users
- Token-based / session-token authentication is not implemented; the web app relies on Flask server-side sessions, with an additional password re-confirmation for sensitive admin operations
- Input validation is basic (not hardened for production scenarios)
- Execution-time measurement is currently implemented for advanced feature engineering, model training, and prediction; ingestion and base feature stages report only outcome and progress
- The reported model performance reflects the latest training runs preserved in `models/model_metrics.json`. The Random Forest is currently trained with strong regularisation (`max_depth=12`, `min_samples_leaf=50`, `min_samples_split=100`, `max_features='sqrt'`), which removes the overfitting flag at the cost of higher absolute test errors than Linear Regression — see the README "Key Findings" section for the full discussion

---
