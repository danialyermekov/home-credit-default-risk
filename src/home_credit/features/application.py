"""Application feature engineering and preprocessing module.

Extracts demographic, financial ratio, and housing information features
from the raw application table, preserving the exact sentinel replacement
and categorical missing representation validated during model development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from home_credit.schema import ACCEPTED_CATEGORICAL_FEATURES

# Tokens used to identify raw housing columns for computing HOUSING_INFO_MISSING_PCT
HOUSING_COLUMN_TOKENS: tuple[str, ...] = (
    "APARTMENTS",
    "BASEMENTAREA",
    "COMMONAREA",
    "ELEVATORS",
    "ENTRANCES",
    "FLOORSMAX",
    "FLOORSMIN",
    "LANDAREA",
    "LIVINGAPARTMENTS",
    "LIVINGAREA",
    "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
    "YEARS_BUILD",
    "YEARS_BEGINEXPLUATATION",
)


def build_application_features(
    applications: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare application-level features and transformations from raw application data.

    Preserves the exact preprocessing validated in EDA and modeling:
    - Identifies 365243 anomaly in DAYS_EMPLOYED and preserves DAYS_EMPLOYED_ANOMALY
    - Replaces 365243 anomaly in DAYS_EMPLOYED with NaN
    - Computes AGE_YEARS and EMPLOYED_YEARS
    - Computes financial ratios: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, ANNUITY_CREDIT_RATIO
    - Computes HOUSING_INFO_MISSING_PCT across housing fields
    - Replaces missing values in categorical columns with '__MISSING__'
    - Sets categorical columns to category dtype

    Parameters
    ----------
    applications : pd.DataFrame
        Raw application dataframe (e.g. application_train.csv or application_test.csv).

    Returns
    -------
    pd.DataFrame
        Cleaned application dataframe with engineered features.
    """
    df = applications.copy()

    # Anomaly handling: 365243 in DAYS_EMPLOYED indicates missing/unemployed
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Demographic and employment duration
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    # Core financial ratios
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # Housing missingness share
    housing_cols = [
        col
        for col in applications.columns
        if any(token in col for token in HOUSING_COLUMN_TOKENS)
    ]
    if housing_cols:
        df["HOUSING_INFO_MISSING_PCT"] = applications[housing_cols].isna().mean(axis=1)

    # Categorical missingness representation
    for cat_col in ACCEPTED_CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna("__MISSING__").astype("category")

    return df
