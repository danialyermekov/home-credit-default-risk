# Home Credit Default Risk

End-to-end machine learning project for predicting credit default risk using the **Home Credit Default Risk** dataset.

The project is built as a complete applied ML workflow: from exploratory data analysis and feature engineering to model validation, optimization, interpretation, and deployment.

> **Current status:** Exploratory data analysis and feature preparation are complete. Modeling is the next stage.

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
* leakage prevention;
* model interpretation;
* reproducibility.

---

## Project status

| Stage                          | Status      |
| ------------------------------ | ----------- |
| Data loading and optimization  | ✅ Completed |
| Exploratory data analysis      | ✅ Completed |
| Missing-value analysis         | ✅ Completed |
| Statistical hypothesis testing | ✅ Completed |
| Feature engineering            | ✅ Completed |
| Final feature preparation      | ✅ Completed |
| Baseline modeling              | ⬜ Planned   |
| Validation strategy            | ⬜ Planned   |
| Model comparison               | ⬜ Planned   |
| Hyperparameter optimization    | ⬜ Planned   |
| Error analysis                 | ⬜ Planned   |
| Model interpretation           | ⬜ Planned   |
| Final model                    | ⬜ Planned   |
| Inference pipeline             | ⬜ Planned   |
| Deployment                     | ⬜ Planned   |

---

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

## Modeling

> This section will be expanded as modeling progresses.

Planned workflow:

1. establish a reproducible train/validation strategy;
2. train a simple baseline;
3. train a CatBoost baseline;
4. evaluate ranking and threshold-dependent metrics;
5. analyze model errors;
6. compare feature sets;
7. optimize hyperparameters;
8. interpret the final model;
9. evaluate the final configuration on untouched test data.

### Baseline

*To be added.*

### Validation strategy

*To be added.*

### Model comparison

*To be added.*

### Hyperparameter optimization

*To be added.*

### Error analysis

*To be added.*

### Model interpretation

*To be added.*

---

## Final model

*To be added after model selection.*

Expected contents:

* selected algorithm;
* final hyperparameters;
* validation performance;
* test performance;
* selected decision threshold;
* most important features;
* limitations.

---

## Inference pipeline

*To be added.*

This section will describe how raw input data is transformed into model-ready features and passed to the trained model.

---

## Deployment

*To be added.*

---

## Project structure

The repository structure will evolve together with the project.

```text
home-credit-default-risk/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── eda.py
├── README.md
├── requirements.txt
└── .gitignore
```

Large datasets and generated data files are excluded from version control.

As the modeling and deployment stages are implemented, the project structure will be expanded accordingly.

---

## Tech stack

Current tools:

* Python
* Pandas
* NumPy
* SciPy
* scikit-learn
* Plotly
* Marimo
* PyArrow

Additional tools will be documented when they are actually introduced into the project.

---

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Marimo analysis:

```bash
marimo edit eda.py
```

Place the Home Credit training dataset in the expected local data directory before running the analysis.

The raw dataset is intentionally excluded from Git.

---

## Next step

The next stage is to establish a modeling baseline and validation framework before attempting optimization.
