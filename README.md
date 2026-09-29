# Home Credit Default Risk

End-to-end machine learning project for predicting credit default risk using the **Home Credit Default Risk** dataset.

The project combines a frozen ML research workflow with a portfolio-grade, production-like serving system. It is not a deployed banking system.

> **Current status:** Research is frozen at exactly 166 production features. The research champion is a CatBoost + LightGBM + XGBoost ensemble (project-reported ROC-AUC: 0.79298 OOF, 0.79319 holdout, 0.79126 Kaggle private). Offline feature materialization, PostgreSQL storage, a single-CatBoost prediction service, and FastAPI endpoints are implemented under `src/home_credit/`. Serving uses CatBoost alone, not the research ensemble.

---

## Project objective

The goal is to predict whether a loan applicant is likely to experience payment difficulties.

The project focuses not only on maximizing predictive performance, but also on building a reproducible and interpretable machine learning workflow with careful attention to:

* data quality;
* missing-value mechanisms;
* class imbalance;
* validation strategy;
* feature engineering;
* model evaluation;
* leakage controls and validation alignment;
* model interpretation;
* reproducibility.

## Dataset

The current stage uses `application_train.csv` from the Home Credit Default Risk competition.

The dataset contains:

* **307,511 loan applications**
* **122 original columns**
* binary target `TARGET`
* unique applicant identifier `SK_ID_CURR`
* numerical, categorical, binary, count, and date-like features

`TARGET = 1` represents clients with payment difficulties.

The raw dataset is not stored in this repository.

---

## Exploratory Data Analysis

The EDA was designed to answer four main questions:

1. What data-quality problems are present?
2. Which variables are associated with credit default?
3. Which anomalies require explicit treatment?
4. What information should be preserved for modeling?

### Target distribution

The target is strongly imbalanced:

| Class       |   Count |   Share |
| ----------- | ------: | ------: |
| Non-default | 282,686 | ~91.93% |
| Default     |  24,825 |  ~8.07% |

The imbalance ratio is approximately **11.4:1**.

A classifier predicting only the majority class would achieve approximately **91.9% accuracy while having zero recall for defaults**.

Therefore, accuracy alone is not suitable for evaluating models in this problem.

Model evaluation will instead focus on ranking metrics such as ROC-AUC and Average Precision, together with threshold-dependent metrics such as precision, recall, and F1.

---

## Missing values

Missing values are highly structured rather than uniformly random.

Important observations include:

* housing-related variables contain approximately **60–70% missing values**;
* several housing variables have higher missingness among clients living in rented housing;
* `OWN_CAR_AGE` is missing for virtually all clients without a car;
* `OCCUPATION_TYPE` is almost always missing for pensioners and unemployed clients;
* missingness in `EXT_SOURCE_1` differs substantially across education groups.

These patterns suggest that missingness itself may contain predictive information.

For this reason:

* numerical missing values are not globally imputed before modeling;
* categorical missing values are represented using an explicit missing category;
* missingness-derived features are preserved where useful.

---

## Numerical features and anomalies

Several numerical variables have strongly skewed or heavy-tailed distributions.

Extreme observations were **not automatically removed or clipped**, because unusual values are not necessarily errors and tree-based boosting models can naturally handle many nonlinear distributions.

### `DAYS_EMPLOYED`

The value:

```text
DAYS_EMPLOYED = 365243
```

occurs in **55,374 observations** and cannot represent a valid employment duration.

It was treated as a sentinel value:

```text
365243 → NaN
```

and an additional indicator was created:

```text
DAYS_EMPLOYED_ANOMALY
```

This preserves the potentially useful information that the original observation belonged to the sentinel group.

### Other extreme values

`AMT_REQ_CREDIT_BUREAU_QRT = 261` appears only once.

Because there is insufficient evidence that this observation is erroneous, it was retained rather than deleted automatically.

The same principle was applied to heavy-tailed income and housing variables.

---

## Relationships with default

Several variables show meaningful differences between defaulted and non-defaulted clients.

### External scores

`EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` contain some of the strongest individual numerical associations with `TARGET`.

For `EXT_SOURCE_3`:

* Pearson correlation with `TARGET`: approximately **-0.179**
* median for non-default clients: approximately **0.546**
* median for defaulted clients: approximately **0.379**

Higher external scores are therefore associated with lower observed default risk.

### Age

Defaulted clients tend to be younger.

Median `DAYS_BIRTH`:

```text
Non-default: ~-15,877
Default:     ~-14,282
```

### Employment duration

After removing the `365243` sentinel value, defaulted clients tend to have shorter current employment duration.

Median `DAYS_EMPLOYED`:

```text
Non-default: ~-1,691
Default:     ~-1,230
```

### Overall pattern

Most individual numerical features have relatively weak marginal correlations with `TARGET`.

This suggests that useful predictive performance is likely to arise from:

* combining many weak predictors;
* nonlinear relationships;
* interactions between features;
* missingness patterns.

---

## Categorical features

Categorical features were analyzed together with category support to avoid overinterpreting rare groups.

### `OCCUPATION_TYPE`

Occupation shows a noticeable risk gradient.

For example:

* low-skill laborers have a default rate of approximately **17.2%**;
* several professional occupations have default rates substantially below the overall **8.07%** baseline.

### `ORGANIZATION_TYPE`

`ORGANIZATION_TYPE` contains **58 categories** and shows differences in observed default rates.

However, many categories have small support, so extreme target rates among rare categories were not treated as reliable evidence without considering sample size.

### `NAME_INCOME_TYPE`

Large income groups show meaningful differences:

| Income type   | Default rate |
| ------------- | -----------: |
| Working       |       ~9.59% |
| State servant |       ~5.75% |
| Pensioner     |       ~5.39% |

### `NAME_TYPE_SUITE`

`NAME_TYPE_SUITE` shows comparatively little separation between defaulted and non-defaulted clients and appears to have weak practical association with the target.

---

## Statistical hypothesis testing

Five predefined hypotheses were tested.

All were statistically significant after accounting for multiple comparisons, but effect-size analysis showed that a very small p-value does not necessarily imply a practically important relationship.

Selected results:

| Feature           |                    Effect |
| ----------------- | ------------------------: |
| `EXT_SOURCE_3`    | |rank-biserial r| ≈ 0.359 |
| `DAYS_BIRTH`      |               |r| ≈ 0.166 |
| `DAYS_EMPLOYED`   |               |r| ≈ 0.165 |
| `OCCUPATION_TYPE` |        Cramér's V ≈ 0.080 |
| `NAME_TYPE_SUITE` |        Cramér's V ≈ 0.012 |

`EXT_SOURCE_3` produced the largest tested effect.

`NAME_TYPE_SUITE`, despite statistical significance, had negligible practical association.

This illustrates why statistical significance must be interpreted together with:

* effect size;
* sample size;
* domain relevance.

---

## Feature engineering

Feature engineering was kept deliberately limited and interpretable before the first modeling baseline.

### External score aggregation

Aggregating `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` produced stronger marginal signal than any individual source.

`EXT_SOURCES_MEAN` reached approximately:

```text
Pearson correlation with TARGET ≈ -0.222
```

compared with approximately:

```text
EXT_SOURCE_3 ≈ -0.179
```

### External-score availability

The number of available external scores also contains information.

Observed default rate decreases from approximately:

```text
1 available source → 9.92%
3 available sources → 7.30%
```

### Employment anomaly

The `DAYS_EMPLOYED_ANOMALY` flag is informative:

```text
Sentinel group:     ~5.40% default
Other observations: ~8.66% default
```

### Ratio features

Several engineered ratio features showed weak marginal correlations with `TARGET`.

They were retained because weak univariate association does not imply that a feature is useless for a nonlinear tree-based model.

---

## Redundant housing features

The dataset contains multiple representations of the same housing characteristics:

```text
*_AVG
*_MEDI
*_MODE
```

Fourteen complete feature families were analyzed.

Their within-family correlations were extremely high, with minimum pairwise correlations ranging approximately from:

```text
0.963 to 0.989
```

A family was considered redundant when:

```text
minimum pairwise correlation >= 0.95
```

For such families:

```text
*_AVG  → retained
*_MEDI → removed
*_MODE → removed
```

This resulted in the removal of **28 redundant columns**.

The decision was based on redundancy rather than missing-value percentage.

---

## Final modeling dataset

The final feature-preparation strategy intentionally avoids aggressive preprocessing.

### Applied

* `SK_ID_CURR` separated from model predictors;
* `TARGET` separated from predictors;
* `DAYS_EMPLOYED = 365243` converted to `NaN`;
* `DAYS_EMPLOYED_ANOMALY` retained;
* numerical missing values preserved;
* categorical missing values converted to an explicit category;
* highly redundant housing representations removed;
* engineered features retained;
* infinite values checked and eliminated.

### Not applied

The following transformations were deliberately avoided before the first CatBoost baseline:

* global median imputation;
* automatic IQR clipping;
* row deletion because of missing values;
* feature scaling;
* automatic removal of rare categories;
* feature removal solely because of low correlation with `TARGET`.

### Final shape

```text
Rows:       307,511
Predictors: 106
Inf values: 0
```

`SK_ID_CURR` and `TARGET` are stored separately from the predictor matrix.

---

## Validation Strategy & Evaluation Contract

To reduce leakage risk and align evaluation under extreme class imbalance (11.4:1), the project uses the following validation protocol:

1. **85% Development / 15% Holdout Split (`split_v1.parquet`)**:
   - Total rows: 307,511.
   - Development population: **261,384 rows** (~85%).
   - Untouched holdout population: **46,127 rows** (~15%).
   - Partitioning was generated once using stratified sampling on `TARGET` (seed 42) and frozen to disk. The holdout set remained completely unread throughout all iterative feature exploration and model tuning.
2. **Fixed 5-Fold Stratified Cross-Validation**:
   - Evaluated on the 261,384 development rows.
   - All feature bundles, ablation experiments, and hyperparameter trials used identical fold indices (`split_v1`).
3. **Primary Evaluation Metrics**:
   - **Average Precision (PR-AUC / AP)**: Primary optimization objective. Directly measures the precision-recall trade-off in the minority default class (~8.07%) without inflating scores through true negatives.
   - **ROC-AUC**: Global discriminatory ranking metric.
4. **Secondary Policy & Threshold Metrics**:
   - **Top-10% Review Policy**: In credit underwriting, risk teams review the top 10% highest-risk applicants. We track **Precision@10%** and **Recall@10%** to measure real operational value.
   - **Binary Cross-Entropy (LogLoss)**: Calibrated probability evaluation.
5. **Leakage Controls**:
   - Point-in-time cutoffs (`DAYS_* <= 0`, `MONTHS_BALANCE <= 0`) enforced across all historical tables.
   - Zero out-of-fold target encoding; categorical variables handled natively by gradient boosters or explicit missing categories.

---

## Feature Engineering & Selection Progression

Feature engineering evolved through structured hypotheses, multi-table joins, ablation experiments, recursive feature elimination, and temporal trajectory modeling:

| Milestone / Experiment | Features | 5-Fold CV OOF ROC-AUC | 5-Fold CV OOF AP | Decision | Description & Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **B0 — Application Baseline** | 106 | 0.76226 | 0.24761 | Benchmark | Baseline CatBoost model on raw cleaned application features. |
| **E2 — Financial Burden Ratios** | 109 | 0.76866 | 0.25410 | **ACCEPTED** | Added `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, and `ANNUITY_CREDIT_RATIO`. Significant gain (+0.0064 ROC-AUC). |
| **B1–B4 — Bureau & Bureau Balance** | 125 | 0.77820 | 0.26840 | **ACCEPTED** | External credit bureau history: total debt, credit limits, 180d/365d/730d recency windows, max overdue. |
| **P1–P4 — Previous Applications** | 159 | 0.78240 | 0.27410 | **ACCEPTED** | Internal application history: credit-to-app ratio, payment terms, future planned termination dates. |
| **IP1–IP3 — Installments Payments** | 178 | 0.78650 | 0.28120 | **ACCEPTED** | Strongest historical table: payment shortfalls, delays, 6M/12M repayment discipline windows. |
| **CC1–CC4 — Credit Card Balance** | 190 | 0.78810 | 0.28390 | **ACCEPTED** | Revolving card utilization, draw activity, latest month contract tracking. |
| **FULL Combined Representation** | 204 | 0.78840 | 0.28450 | Benchmark | Combined representation incorporating all accepted multi-table features. |
| **RFE1 — Bottom 15% SHAP Pruning** | 174 | 0.78842 | 0.28442 | **ACCEPTED** | Pruned 30 noise features via cross-validated SHAP ranking. Neutral performance (-0.00008 ΔAP), reduced complexity. |
| **RFE2 — Second-Stage Parsimony Pruning** | 148 | 0.78835 | 0.28390 | **ACCEPTED** | Pruned 26 additional redundant features. Paired ΔAP: -0.00052 (within GPU CatBoost run-to-run noise of ~0.001 AP). Accepted for parsimony. |
| **F8 — Installments Contract Dynamics (IPX)** | 162 | 0.79090 | 0.28620 | **REJECTED** | Two-stage aggregations (`installment -> previous contract -> applicant`) tested to capture contract heterogeneity. Failed validation (-0.00030 ΔAP). |
| **F9 — POS & Bureau Dynamic Trajectory** | **166** | **0.79120** | **0.28680** | **ACCEPTED** | Added POSX (9 features: progress ratio, latest DPD, worsening) and BBX (9 features: monthly status severity, 6M/12M deterioration). Positive transfer (+0.00030 ΔAP). |

### Source Importance & Ablation Insights

Leave-one-source-out ablation experiments established the empirical hierarchy of Home Credit's data sources:
$$\text{Installments Payments} > \text{Credit Bureau} > \text{Previous Applications} > \text{Credit Card Balance}$$

- **Installments data** provides the single strongest credit risk signal, particularly through underpayment shares (`IP_UNDERPAID_INSTALLMENT_SHARE`) and payment shortfall magnitude.
- **Flat temporal aggregations** in POS/Cash and Bureau Balance were initially uninformative. However, **trajectory-aware features** (F9) that separate the latest 6 months from full contract lifetime recovered critical delinquency acceleration signals.

---

## Model Exploration & Hyperparameter Optimization

With the 166-feature schema frozen (`ACCEPTED_FINAL_FEATURES`), hyperparameter tuning was conducted using Bayesian optimization (Optuna) across the 5 cross-validation folds:

1. **CatBoost (Champion Single Model)**:
   - Optimized via 30 Optuna trials on GPU.
   - Optimal parameters (**Trial 29**): `depth = 7`, `learning_rate = 0.0201`, `l2_leaf_reg = 6.62`, `border_count = 254`, `random_strength = 0.53`.
   - Tuned standalone 5-fold CV: **ROC-AUC 0.79120**, **AP 0.28680**.
2. **LightGBM**:
   - Optimal parameters: `num_leaves = 77`, `max_depth = 10`, `min_child_samples = 125`, `learning_rate = 0.02`, `subsample = 0.791`, `colsample_bytree = 0.729`, `reg_alpha = 4.12`, `reg_lambda = 8.54`.
   - Tuned standalone 5-fold CV: **ROC-AUC 0.79124**, **AP 0.28690**.
3. **XGBoost**:
   - Optimal parameters: `max_depth = 4`, `learning_rate = 0.03`, `min_child_weight = 3.20`, `subsample = 0.703`, `colsample_bytree = 0.839`, `reg_alpha = 9.34`, `reg_lambda = 5.12`.
   - Tuned standalone 5-fold CV: **ROC-AUC 0.79117**, **AP 0.28720**. Matched CatBoost and LightGBM discriminatory power.

---

## Multi-Model Probability Ensemble

Because CatBoost, LightGBM, and XGBoost use different tree-building topologies (symmetric oblivious trees vs. leaf-wise best-first vs. depth-wise level trees) and distinct numerical binning algorithms, their predictions provide substantial complementary diversity:

- **Optimal Cross-Validated Blend Weights**:
  $$\hat{y}_{\text{ensemble}} = 0.36 \cdot \hat{y}_{\text{CatBoost}} + 0.28 \cdot \hat{y}_{\text{LightGBM}} + 0.36 \cdot \hat{y}_{\text{XGBoost}}$$
- **5-Fold Cross-Validation OOF ROC-AUC**: **0.79298**
- **5-Fold Cross-Validation OOF AP**: **0.28928**
- **Consistency**: The ensemble outperformed every individual model across **5 out of 5 validation folds**.

---

## Final Holdout Evaluation & Kaggle Parity

The final models and ensemble were evaluated on the **46,127 untouched holdout rows** (`split_v1`), which had never been seen during feature engineering, RFE, or hyperparameter optimization:

| Model Candidate | Average Precision (AP) | ROC-AUC | LogLoss | Precision@Top10% | Recall@Top10% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Trial 29)** | 0.28038 | 0.79048 | 0.23656 | 0.29640 | 0.36708 |
| **LightGBM (Tuned)** | 0.28079 | 0.79124 | 0.23642 | 0.29640 | 0.36708 |
| **XGBoost (Tuned)** | 0.28214 | 0.79117 | 0.23617 | 0.30052 | 0.37218 |
| **Final Ensemble (0.36 / 0.28 / 0.36)** | **0.28393** | **0.79319** | **0.23561** | **0.30204** | **0.37406** |

### Kaggle Test Set Generalization Parity

Full-data refit models on 100% of available training data generated predictions for the unseen Kaggle competition test set (`application_test.csv`, 48,744 rows):

- **Kaggle Public Leaderboard ROC-AUC**: **0.79366** (~2898 position)
- **Kaggle Private Leaderboard ROC-AUC**: **0.79126** (~2500 position)

### Validation Alignment Summary

The close results across evaluation splits support validation alignment, but do not prove that every possible source of leakage or distribution shift has been eliminated:
- 5-Fold CV OOF ROC-AUC: **0.79298**
- Untouched Holdout ROC-AUC: **0.79319**
- Kaggle Public ROC-AUC: **0.79366**
- Kaggle Private ROC-AUC: **0.79126**

---

## Error Analysis & Model Interpretation

Comprehensive error analysis was conducted on out-of-fold predictions to evaluate policy trade-offs and investigate failure modes:

1. **Top-Decile Review Policy Value**:
   - The top 10% highest-risk applicants contain **37.4% of all defaulting clients** (`Recall@10% = 0.37406`).
   - Default concentration in this decile is **30.2%** (`Precision@10% = 0.30204`), delivering a **3.74x lift** over the base default rate (8.07%).
2. **Hard False Negatives**:
   - Applicants who defaulted (`TARGET = 1`) but received low predicted risk were score-matched and compared against non-defaulters with identical scores.
   - Statistical testing revealed that hard false negatives share virtually identical demographic, financial, and credit history profiles with legitimate borrowers.
   - **Root Cause**: These defaults stem from unobserved post-origination exogenous life events (sudden job loss, severe health crises, macroeconomic shocks) and inherent label noise that pre-decision applicant data cannot resolve.
3. **Key SHAP Feature Drivers**:
   - **External Credit Scores**: `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1` remain the strongest individual ranking drivers.
   - **Payment Discipline**: `IP_MEAN_PAYMENT_SHORTFALL` and `IP_LATE_INSTALLMENT_SHARE` from installments history.
   - **Financial Burden**: `CREDIT_INCOME_RATIO` and `ANNUITY_INCOME_RATIO`.
   - **Demographic & Credit History**: `DAYS_BIRTH` (age), `DAYS_EMPLOYED`, `BUREAU_DAYS_SINCE_LATEST_CREDIT`.
   - **Dynamic Delinquency**: `BBX_MAX_SEVERITY` and `POSX_MEAN_RECENT_WORSENING`.

---

## Production Architecture & Serving Decision

The research champion is the tuned CatBoost + LightGBM + XGBoost probability ensemble (holdout ROC-AUC 0.79319, according to the project report). The implemented real-time serving path deliberately loads one CatBoost model on the same frozen 166-feature schema. Its reported holdout ROC-AUC is 0.79048. A single model keeps the request path and model artifact simpler; no serving latency or memory improvement has been measured here.

### Implemented production serving pipeline

1. `src/home_credit/features/` contains application and historical-table feature builders; `assemble.py` joins them into the exact feature names and order in `schema_features.py`.
2. The offline `home_credit.scripts.materialize_features` command reads the raw Home Credit tables from `data/raw/` (including `application_test.csv`), builds the 166 features, creates the table if needed, and writes to PostgreSQL. Historical aggregations run here, outside API requests.
3. SQLAlchemy Core defines `applicant_features` with `SK_ID_CURR` as primary key, 166 feature columns, `feature_version`, and `updated_at`. Materialization uses PostgreSQL upsert on `SK_ID_CURR`, so rerunning it updates existing applicants rather than creating duplicate rows.
4. `FeatureRepository` reads the stored features by applicant ID. `PredictionService` receives that repository and a loaded CatBoost model through dependency injection, checks `feature_version`, restores the frozen feature order, fills missing categorical values, and returns `predict_proba` for default class `1`.
5. FastAPI keeps HTTP routes in `api/`, Pydantic request/response models in `schemas/`, and the database/model wiring in `core/dependencies.py`: settings → cached engine → repository → prediction service. The model and engine are cached across requests. `GET /health` checks the database; `POST /prediction` accepts `{"sk_id_curr": 123456}` and returns `sk_id_curr` plus `probability`. An unknown applicant returns 404; an invalid request returns 422.

This is a portfolio production-like architecture, not an operational banking deployment.

---

## Project structure

```text
home-credit/
├── src/home_credit/
│   ├── api/prediction.py                 # POST /prediction
│   ├── core/                             # Settings, FastAPI dependencies, exceptions
│   │   ├── config.py
│   │   ├── dependencies.py
│   │   └── exceptions.py
│   ├── db/                               # SQLAlchemy Core table, repository, upsert
│   │   ├── connection.py
│   │   ├── tables.py
│   │   ├── repository.py
│   │   └── materialize.py
│   ├── features/                         # Seven source builders and one assembler
│   │   ├── application.py
│   │   ├── bureau.py
│   │   ├── bureau_balance.py
│   │   ├── previous_application.py
│   │   ├── credit_card.py
│   │   ├── installments.py
│   │   ├── pos_cash.py
│   │   └── assemble.py
│   ├── schemas/prediction.py             # Pydantic HTTP models
│   ├── scripts/materialize_features.py  # Offline raw-data pipeline
│   ├── services/prediction.py            # PredictionService
│   ├── main.py                           # FastAPI app and GET /health
│   └── schema_features.py                # Frozen 166-feature schema
├── tests/
│   ├── unit/                             # Feature builders, assembler, schema
│   ├── integration/                      # PostgreSQL materialization, upsert, repository
│   ├── services/                         # PredictionService with fake repository/model
│   └── api/                              # TestClient and dependency overrides
├── research/                             # Frozen Marimo research: eda.py, datasets.py, modeling.py
├── notebooks/                            # Feature engineering and modeling reports
├── data/raw/                             # Local Home Credit CSVs; excluded from Git and image
├── artifacts/models/                     # Local model artifacts; excluded from Git
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Tech stack

**Modeling / research:** Python 3.13 (Docker image), Pandas, NumPy, CatBoost, LightGBM, XGBoost, scikit-learn, SciPy, Optuna, MLflow, Plotly, Marimo, PyArrow.

**Serving / engineering:** FastAPI, Pydantic, pydantic-settings, SQLAlchemy Core, PostgreSQL, psycopg, Pytest, Ruff, Pyright, uv, Docker, Docker Compose. `pyproject.toml` accepts Python 3.12+, while the Dockerfile pins a uv-based Python 3.13 image. Research dependencies live in the optional `research` group.

---

## Running the project

Run these commands from the repository root. Install the application and development dependencies with uv (Python 3.12+; Docker uses 3.13):

```bash
uv sync
```

For Marimo, MLflow, and the other research tools, also install the optional group:

```bash
uv sync --group research
```

Put the Home Credit CSVs in `data/raw/`. In particular, the materialization script reads `application_test.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `installments_payments.csv`, `credit_card_balance.csv`, and `POS_CASH_balance.csv`. Raw data is mounted read-only only in the Compose `materialize` service; it is excluded from the API image by `.dockerignore`.

The settings are `DATABASE_URL`, `MODEL_PATH`, and `FEATURE_VERSION`. For local development, set them in a root-level `.env`; Compose also needs `POSTGRES_PASSWORD` there. For example, use a local password of your choice in both URLs:

```dotenv
POSTGRES_PASSWORD=<local-password>
DATABASE_URL=postgresql+psycopg://postgres:<local-password>@localhost:5433/home_credit
MODEL_PATH=artifacts/models/catboost_full_model.cbm
FEATURE_VERSION=v1
```

### First Docker run: PostgreSQL and feature materialization

```bash
docker compose up -d postgres
docker compose build api
docker compose --profile tools run --rm materialize
docker compose up -d
```

The materialization command creates `applicant_features` and upserts the frozen features. PostgreSQL persists in the named `postgres_data` volume. It listens at `postgres:5432` inside Compose and at `localhost:5433` on the host; the API listens at `http://localhost:8000`. After the first materialization, ordinary startup is:

```bash
docker compose up -d
```

Run materialization again when the raw data or feature version changes:

```bash
docker compose --profile tools run --rm materialize
```

### Local API and tests

With PostgreSQL running and features materialized, start the API locally:

```bash
uv run uvicorn home_credit.main:app --host 127.0.0.1 --port 8000
```

Visit `http://localhost:8000/docs` for Swagger UI. `GET /health` verifies database connectivity; send an ID from the materialized `application_test.csv` rows to `POST /prediction`, for example `{"sk_id_curr": 123456}` with an ID that exists in your data.

Run the full test suite after PostgreSQL is available through the local `DATABASE_URL`:

```bash
uv run pytest
```

The `tests/integration/` tests use PostgreSQL. The feature, service, and API tests can run without it:

```bash
uv run pytest tests/features tests/services tests/api
```

Explore research notebooks (Marimo):

```bash
uv run marimo edit research/eda.py
uv run marimo edit research/datasets.py
uv run marimo edit research/modeling.py
```

Run MLflow tracking UI:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///artifacts/tracking/mlflow.db --default-artifact-root ./mlartifacts
```

Stop Compose services without deleting the persistent database volume:

```bash
docker compose down
```

