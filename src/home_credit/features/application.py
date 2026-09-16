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

APPLICATION_FEATURES: tuple[str, ...] = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
    "ORGANIZATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "TOTALAREA_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "FLAG_DOCUMENT_3",
    "FLAG_DOCUMENT_6",
    "FLAG_DOCUMENT_18",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "HOUSING_INFO_MISSING_PCT",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "ANNUITY_CREDIT_RATIO",
)


def build_application_features(
    applications: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare application-level features and transformations from raw application data.

    Preserves the exact preprocessing validated in EDA and modeling:
    - Identifies 365243 anomaly in DAYS_EMPLOYED and preserves DAYS_EMPLOYED_ANOMALY
    - Replaces 365243 anomaly in DAYS_EMPLOYED with NaN
    - Computes AGE_YEARS and EMPLOYED_YEARS
    - Computes financial ratios:
        CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, ANNUITY_CREDIT_RATIO
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

    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Demographic and employment duration
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    # Core financial ratios
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(
        0, np.nan
    )

    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(
        0, np.nan
    )

    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)

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
        if isinstance(df[cat_col].dtype, pd.CategoricalDtype):
            if "__MISSING__" not in df[cat_col].cat.categories:
                df[cat_col] = df[cat_col].cat.add_categories(["__MISSING__"])

            df[cat_col] = df[cat_col].fillna("__MISSING__")

        else:
            df[cat_col] = df[cat_col].fillna("__MISSING__").astype("category")

    return df
