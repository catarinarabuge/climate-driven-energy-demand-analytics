# Climate-Driven Energy Demand Analytics System

Academic group project developed for the **Project in Artificial Intelligence and Data Science** course, as part of the Bachelor's Degree in Data Engineering and Data Science at the University of Coimbra.

This project explores how publicly available climate data can help explain and predict national electricity demand in Portugal. The system integrates electricity demand data with climate variables, applies data cleaning and feature engineering, trains predictive models, and provides prediction functionality through both a Flask web application and a command-line interface.

## Overview

Electricity demand is influenced by several external factors, including weather conditions, seasonality and daily consumption patterns. This project focuses on analysing the relationship between climate variables and electricity load, using a modular AI-driven system designed to support data ingestion, processing, modelling, evaluation and prediction.

The main research question is:

**How can publicly available climate data explain and predict national electricity demand in Portugal?**

Sub-questions explored in the project include:

* To what extent do temperature, solar radiation and wind speed influence electricity load?
* Can climate variables improve short-term electricity demand forecasting?
* Which engineered features contribute most to predictive performance?

## Main Features

* Data ingestion from public electricity and climate data sources
* Cleaning and temporal alignment of energy and weather datasets
* Feature engineering with temporal, lagged and climate-based features
* Training and evaluation of predictive models
* Time-aware train/test split
* Model evaluation using MAE, RMSE and R²
* Prediction functionality through a Flask web application
* Command-line interface for data exploration, training and prediction
* Authentication and role-based access control
* Logging of relevant system actions
* Unit tests for important components
* Documentation of architecture, use cases and quality attributes

## Data Sources

The project was designed to combine:

* **Electricity demand data:** ENTSO-E Transparency Platform
* **Climate data:** ERA5 dataset from the Copernicus Climate Data Store

The final dataset aligns hourly electricity demand observations with climate variables for the same time period.

Main climate variables considered:

* 2-meter air temperature
* solar radiation
* 10-meter wind speed
* precipitation, when available

## Models and Evaluation

The project includes both a baseline model and a more flexible machine learning model.

Models used:

* Linear Regression
* Random Forest Regressor

Evaluation metrics:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination

Since the data has a temporal structure, model evaluation was performed using a time-aware train/test split instead of random shuffling.

## Key Results

The system was evaluated using Portuguese electricity demand data aligned with ERA5 climate data.

The best-performing configuration achieved strong predictive performance with the Linear Regression model, showing that engineered temporal and climate-related features were useful for explaining demand patterns.

Example results from the project documentation:

| Model             |  Test MAE | Test RMSE | Test R² | Overfitting Flag |
| ----------------- | --------: | --------: | ------: | ---------------- |
| Linear Regression | 256.79 MW | 319.19 MW |   0.931 | No               |
| Random Forest     | 342.16 MW | 486.98 MW |   0.839 | No               |

Key observations:

* Linear Regression achieved the best test performance in the final configuration.
* Time-aware evaluation was used to better reflect real forecasting conditions.
* Climate-aware and temporal features contributed to predictive performance.
* Model evaluation considered both accuracy and generalisation.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Flask
* HTML
* pytest
* bcrypt
* python-dotenv
* Git / GitLab CI

## Repository Structure

```text
.
├── README.md
├── ARCHITECTURE.md
├── QUALITY_ATTRIBUTES.md
├── USE_CASES.md
├── app.py
├── cli.py
├── api.py
├── auth_service.py
├── advanced_features.py
├── base_features.py
├── merge_datasets.py
├── model_training.py
├── model_utils.py
├── predict.py
├── prediction_service.py
├── entsoe_client.py
├── era5.py
├── create_admin.py
├── conftest.py
├── test_auth.py
├── test_cleaning.py
├── test_era5.py
├── test_evaluation.py
├── test_features.py
├── test_ingestion.py
├── test_model_training.py
├── test_predict.py
├── admin.html
├── user.html
├── landing_page.html
└── requirements.txt
```

## Documentation

Additional documentation files are included in the repository:

* `ARCHITECTURE.md` — system architecture, components and data flow
* `USE_CASES.md` — structured use cases
* `QUALITY_ATTRIBUTES.md` — performance, reliability and security considerations

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/catarinarabuge/climate-driven-energy-demand-analytics.git
cd climate-driven-energy-demand-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask web application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5001
```

### 4. Run the command-line interface

```bash
python cli.py
```

## Configuration

Some parts of the original academic system require API keys to retrieve fresh data from external sources. These credentials should be stored locally in a `.env` file and must not be committed to the repository.

Example:

```text
ENTSOE_API_KEY=your_entsoe_token_here
CDSAPI_KEY=your_uid:your_key_here
```

For the public portfolio version, private configuration files, credentials, logs, large datasets and trained model artifacts are not included.

## Testing

The project includes unit tests for several components, including authentication, data cleaning, feature generation, model training, prediction and data ingestion.

To run the tests:

```bash
pytest
```

## Security and Reliability

The project includes several basic security and reliability measures:

* password hashing with bcrypt;
* minimum password length requirement;
* role-based access control;
* authentication required for sensitive operations;
* environment variables for private credentials;
* no hardcoded API keys in the public repository;
* input validation where appropriate;
* unit tests for normal and failure scenarios.

## Public Repository Note

This repository is a public portfolio version of an academic group project. Large datasets, trained model artifacts, logs, local configuration files and private credentials are not included.

The goal of this repository is to present the project structure, main implementation files, documentation and testing approach in a clean and safe way for portfolio purposes.

## Academic Context

**Course:** Project in Artificial Intelligence and Data Science
**Degree:** Bachelor's Degree in Data Engineering and Data Science
**University:** University of Coimbra / FCTUC
**Academic Year:** 2025/2026

## What I Learned

This project helped develop skills in:

* working with real-world energy and climate data;
* preparing and aligning time-series datasets;
* creating predictive features from temporal and climate variables;
* training and evaluating machine learning models;
* interpreting results using appropriate metrics;
* designing modular data science software;
* understanding the importance of authentication, testing, reliability and security in AI-driven systems;
* documenting a complete data science project.

## Status

Academic project completed.


