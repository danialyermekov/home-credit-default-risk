import numpy as np
import pandas as pd
import pytest

from home_credit.features.previous_application import (
    PREVIOUS_APPLICATION_ACCEPTED_FEATURES,
    build_previous_application_features,
)
from home_credit.schema_features import ACCEPTED_FINAL_FEATURES


@pytest.fixture
def previous_application_df():
    return pd.DataFrame(
        [
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "AMT_APPLICATION": 1000.0,
                "AMT_CREDIT": 1200.0,
                "AMT_GOODS_PRICE": 900.0,
                "AMT_ANNUITY": 100.0,
                "AMT_DOWN_PAYMENT": 50.0,
                "CNT_PAYMENT": 12.0,
                "DAYS_DECISION": -100,
                "DAYS_FIRST_DRAWING": -95,
                "DAYS_FIRST_DUE": -90,
                "DAYS_LAST_DUE_1ST_VERSION": 100,
                "DAYS_LAST_DUE": 90,
                "DAYS_TERMINATION": -50,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 102,
                "AMT_APPLICATION": 2000.0,
                "AMT_CREDIT": 1800.0,
                "AMT_GOODS_PRICE": 1900.0,
                "AMT_ANNUITY": 200.0,
                "AMT_DOWN_PAYMENT": 100.0,
                "CNT_PAYMENT": 24.0,
                "DAYS_DECISION": -500,
                "DAYS_FIRST_DRAWING": -480,
                "DAYS_FIRST_DUE": -450,
                "DAYS_LAST_DUE_1ST_VERSION": -200,
                "DAYS_LAST_DUE": -210,
                "DAYS_TERMINATION": -300,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 103,
                "AMT_APPLICATION": 0.0,
                "AMT_CREDIT": 500.0,
                "AMT_GOODS_PRICE": 400.0,
                "AMT_ANNUITY": 50.0,
                "AMT_DOWN_PAYMENT": 0.0,
                "CNT_PAYMENT": 6.0,
                "DAYS_DECISION": -1000,
                "DAYS_FIRST_DRAWING": 365243,
                "DAYS_FIRST_DUE": -950,
                "DAYS_LAST_DUE_1ST_VERSION": 365243,
                "DAYS_LAST_DUE": 365243,
                "DAYS_TERMINATION": 365243,
            },
            {
                "SK_ID_CURR": 2,
                "SK_ID_PREV": 201,
                "AMT_APPLICATION": 3000.0,
                "AMT_CREDIT": 3300.0,
                "AMT_GOODS_PRICE": 2800.0,
                "AMT_ANNUITY": 300.0,
                "AMT_DOWN_PAYMENT": 150.0,
                "CNT_PAYMENT": 36.0,
                "DAYS_DECISION": -200,
                "DAYS_FIRST_DRAWING": -190,
                "DAYS_FIRST_DUE": -180,
                "DAYS_LAST_DUE_1ST_VERSION": 50,
                "DAYS_LAST_DUE": 40,
                "DAYS_TERMINATION": -20,
            },
            {
                "SK_ID_CURR": 3,
                "SK_ID_PREV": 301,
                "AMT_APPLICATION": 3000.0,
                "AMT_CREDIT": 3300.0,
                "AMT_GOODS_PRICE": 2800.0,
                "AMT_ANNUITY": 300.0,
                "AMT_DOWN_PAYMENT": 150.0,
                "CNT_PAYMENT": 36.0,
                "DAYS_DECISION": -200,
                "DAYS_FIRST_DRAWING": -190,
                "DAYS_FIRST_DUE": -300,
                "DAYS_LAST_DUE_1ST_VERSION": -100,
                "DAYS_LAST_DUE": 40,
                "DAYS_TERMINATION": -20,
            },
        ]
    )


def test_origin_dataset_is_not_mutated(previous_application_df: pd.DataFrame) -> None:
    original = previous_application_df.copy(deep=True)
    build_previous_application_features(original)

    pd.testing.assert_frame_equal(original, previous_application_df)


def test_unique_applicant_id(previous_application_df: pd.DataFrame) -> None:
    prev_apps = build_previous_application_features(previous_application_df)

    assert prev_apps["SK_ID_CURR"].is_unique


def test_len_prev_apps_schema() -> None:
    assert len(PREVIOUS_APPLICATION_ACCEPTED_FEATURES) == 34


def test_no_duplicates_in_prev_apps_schema() -> None:
    assert len(set(PREVIOUS_APPLICATION_ACCEPTED_FEATURES)) == len(
        PREVIOUS_APPLICATION_ACCEPTED_FEATURES
    )


def test_prev_apps_schema_is_subset_of_final_features() -> None:
    assert set(PREVIOUS_APPLICATION_ACCEPTED_FEATURES).issubset(ACCEPTED_FINAL_FEATURES)


def test_previous_application_output_schema(
    previous_application_df: pd.DataFrame,
) -> None:
    prev_apps = build_previous_application_features(previous_application_df)

    assert list(prev_apps.columns) == [
        "SK_ID_CURR",
        *PREVIOUS_APPLICATION_ACCEPTED_FEATURES,
    ]


def test_no_inf_in_ratio(previous_application_df: pd.DataFrame) -> None:
    prev_apps = build_previous_application_features(previous_application_df)
    ratios = [
        "MEAN_CREDIT_APPLICATION_RATIO",
        "MIN_CREDIT_APPLICATION_RATIO",
        "MAX_CREDIT_APPLICATION_RATIO",
    ]

    assert not np.isinf(prev_apps[ratios].to_numpy()).any()


def test_planned_days_remaining_is_clipped(
    previous_application_df: pd.DataFrame,
) -> None:
    prev_apps = build_previous_application_features(previous_application_df)
    row = prev_apps.set_index("SK_ID_CURR").loc[3]

    assert row["MAX_PREV_PLANNED_DAYS_REMAINING"] == 0


def test_temporal_sentinel_processing(previous_application_df: pd.DataFrame) -> None:
    prev_apps = build_previous_application_features(previous_application_df)
    row = prev_apps.set_index("SK_ID_CURR").loc[1]

    assert row["MEAN_PREV_PLANNED_DURATION_DAYS"] == pytest.approx(220.0)
    assert row["MAX_PREV_PLANNED_DURATION_DAYS"] == pytest.approx(250.0)

    assert row["PREV_FUTURE_PLANNED_END_COUNT"] == 1
    assert row["PREV_FUTURE_PLANNED_END_SHARE"] == pytest.approx(1 / 3)

    assert row["MAX_PREV_PLANNED_DAYS_REMAINING"] == 100
    assert row["PREV_DAYS_SINCE_LAST_TERMINATION"] == 50
