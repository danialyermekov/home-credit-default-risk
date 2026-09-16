import numpy as np
import pandas as pd
import pytest

from home_credit.features.installments import (
    INSTALLMENTS_ACCEPTED_FEATURES,
    build_installments_features,
)


@pytest.fixture
def installments_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Client 1, contract 101, installment 1.
            # One installment paid with TWO payment records.
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "DAYS_INSTALMENT": -100,
                "DAYS_ENTRY_PAYMENT": -110,
                "AMT_INSTALMENT": 1000.0,
                "AMT_PAYMENT": 400.0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "DAYS_INSTALMENT": -100,
                "DAYS_ENTRY_PAYMENT": -90,
                "AMT_INSTALMENT": 1000.0,
                "AMT_PAYMENT": 600.0,
            },
            # Underpaid + late.
            # Older than 6m, but inside 12m.
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 2,
                "DAYS_INSTALMENT": -300,
                "DAYS_ENTRY_PAYMENT": -290,
                "AMT_INSTALMENT": 1000.0,
                "AMT_PAYMENT": 700.0,
            },
            # Overpayment.
            # Older than 12m.
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 3,
                "DAYS_INSTALMENT": -500,
                "DAYS_ENTRY_PAYMENT": -510,
                "AMT_INSTALMENT": 1000.0,
                "AMT_PAYMENT": 1200.0,
            },
            # Second contract, zero scheduled amount.
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 102,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "DAYS_INSTALMENT": -50,
                "DAYS_ENTRY_PAYMENT": -50,
                "AMT_INSTALMENT": 0.0,
                "AMT_PAYMENT": 0.0,
            },
            # Client 2.
            {
                "SK_ID_CURR": 2,
                "SK_ID_PREV": 201,
                "NUM_INSTALMENT_VERSION": 1,
                "NUM_INSTALMENT_NUMBER": 1,
                "DAYS_INSTALMENT": -20,
                "DAYS_ENTRY_PAYMENT": -20,
                "AMT_INSTALMENT": 500.0,
                "AMT_PAYMENT": 500.0,
            },
        ]
    )


def test_input_dataframe_is_not_mutated(
    installments_df: pd.DataFrame,
) -> None:
    original = installments_df.copy(deep=True)

    build_installments_features(installments_df)

    pd.testing.assert_frame_equal(installments_df, original)


def test_installments_output_contract(
    installments_df: pd.DataFrame,
) -> None:
    features = build_installments_features(installments_df)

    assert features["SK_ID_CURR"].is_unique
    assert list(features.columns) == [
        "SK_ID_CURR",
        *INSTALLMENTS_ACCEPTED_FEATURES,
    ]


def test_payment_records_are_collapsed_to_installment_level(
    installments_df: pd.DataFrame,
) -> None:
    features = build_installments_features(installments_df)
    row = features.set_index("SK_ID_CURR").loc[1]

    # Five raw payment rows, but only four scheduled installments.
    assert row["IP_PAYMENT_RECORD_COUNT"] == 5
    assert row["IP_UNIQUE_INSTALLMENT_COUNT"] == 4

    # Installment 1:
    # 400 + 600 = 1000 -> fully paid.
    #
    # Late installments:
    # installment 1 -> late by 10
    # installment 2 -> late by 10
    # installment 3 -> early
    # contract 102 -> on time
    assert row["IP_LATE_INSTALLMENT_SHARE"] == pytest.approx(0.5)

    # Only installment 2 is underpaid.
    assert row["IP_UNDERPAID_INSTALLMENT_SHARE"] == pytest.approx(0.25)


def test_payment_coverage_handles_overpayment_and_zero_amount(
    installments_df: pd.DataFrame,
) -> None:
    features = build_installments_features(installments_df)
    row = features.set_index("SK_ID_CURR").loc[1]

    # Coverage per installment:
    # 1000 / 1000 = 1.0
    # 700 / 1000  = 0.7
    # 1200 / 1000 = 1.2 -> clipped to 1.0
    # 0 / 0        = NaN
    #
    # pandas mean ignores NaN:
    # (1.0 + 0.7 + 1.0) / 3 = 0.9
    assert row["IP_MEAN_PAYMENT_COVERAGE_RATIO"] == pytest.approx(0.9)

    assert not np.isinf(features["IP_MEAN_PAYMENT_COVERAGE_RATIO"].to_numpy()).any()


def test_recent_windows_use_scheduled_installment_date(
    installments_df: pd.DataFrame,
) -> None:
    features = build_installments_features(installments_df)
    row = features.set_index("SK_ID_CURR").loc[1]

    # Last 6 months:
    # -100, -50
    assert row["IP_RECENT_6M_INSTALLMENT_COUNT"] == 2
    assert row["IP_RECENT_6M_LATE_SHARE"] == pytest.approx(0.5)

    # Last 12 months:
    # -100, -300, -50
    assert row["IP_RECENT_12M_INSTALLMENT_COUNT"] == 3
    assert row["IP_RECENT_12M_UNDERPAID_SHARE"] == pytest.approx(1 / 3)
