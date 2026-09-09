import numpy as np
import pandas as pd
import pytest

from home_credit.features.application import build_application_features
from home_credit.schema import ACCEPTED_CATEGORICAL_FEATURES
from home_credit.features.application import APPLICATION_FEATURES

@pytest.fixture
def application_df():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100001, 100002],
            "DAYS_BIRTH": [-15000, -18000],
            "DAYS_EMPLOYED": [-1000, 365243],
            "AMT_INCOME_TOTAL": [200000.0, 0.0],
            "AMT_CREDIT": [500000.0, 300000.0],
            "AMT_ANNUITY": [25000.0, 18000.0],
            "NAME_CONTRACT_TYPE": ["Cash loans", np.nan],
            "LIVINGAREA_AVG": [0.5, np.nan]
        }
    )

    for column in ACCEPTED_CATEGORICAL_FEATURES:
        df[column] = ["VALUE", "VALUE"]

    df.loc[1, "NAME_CONTRACT_TYPE"] = np.nan

    return df

@pytest.fixture
def categorical_application_df(application_df):
    df = application_df.copy()

    df["NAME_CONTRACT_TYPE"] = (
        df["NAME_CONTRACT_TYPE"].astype("category")
    )

    return df


def test_input_dataframe_is_not_mutated(application_df):
    original = application_df.copy(deep=True)

    build_application_features(application_df)

    pd.testing.assert_frame_equal(application_df, original)

def test_days_employed_sentinel(application_df):
    df = build_application_features(application_df)

    assert df.loc[1, "DAYS_EMPLOYED_ANOMALY"] == 1
    assert pd.isna(df.loc[1, "DAYS_EMPLOYED"])

    assert df.loc[0, "DAYS_EMPLOYED_ANOMALY"] == 0
    assert df.loc[0, "DAYS_EMPLOYED"] == -1000

def test_no_inf_in_financial_ratios(application_df):
    df = build_application_features(application_df)
    ratio_columns = [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",
    ]
    assert not np.isinf(df[ratio_columns].to_numpy()).any()

def test_zero_division_produces_nan_in_financial_ratios(application_df):
    df = build_application_features(application_df)

    assert pd.isna(df.loc[1, "CREDIT_INCOME_RATIO"])
    assert pd.isna(df.loc[1, "ANNUITY_INCOME_RATIO"])

def test_housing_missing_pct(application_df):
    df = build_application_features(application_df)

    assert df.loc[1, "HOUSING_INFO_MISSING_PCT"] == 1

def test_category_dtype_missing_values_filling(categorical_application_df):
    df = build_application_features(categorical_application_df)

    assert df.loc[1, "NAME_CONTRACT_TYPE"] == "__MISSING__"
    assert isinstance(
        df["NAME_CONTRACT_TYPE"].dtype,
        pd.CategoricalDtype,
    )

# def test_application_features_are_created(application_df):
#     df = build_application_features(application_df)

#     assert set(APPLICATION_FEATURES).issubset(df.columns)
