import pandas as pd
import pytest

from home_credit.features.bureau_balance import (
    BBX_ACCEPTED_FEATURES,
    build_bureau_balance_features,
)


@pytest.fixture
def bureau_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SK_ID_BUREAU": 101, "SK_ID_CURR": 1},
            {"SK_ID_BUREAU": 102, "SK_ID_CURR": 1},
            {"SK_ID_BUREAU": 201, "SK_ID_CURR": 2},
        ]
    )


@pytest.fixture
def bureau_balance_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Client 1, bureau account 101
            {
                "SK_ID_BUREAU": 101,
                "MONTHS_BALANCE": -12,
                "STATUS": "0",
            },
            {
                "SK_ID_BUREAU": 101,
                "MONTHS_BALANCE": -5,
                "STATUS": "1",
            },
            {
                "SK_ID_BUREAU": 101,
                "MONTHS_BALANCE": -1,
                "STATUS": "2",
            },
            # Future row: must be ignored
            {
                "SK_ID_BUREAU": 101,
                "MONTHS_BALANCE": 1,
                "STATUS": "5",
            },
            # Client 1, bureau account 102
            # Never delinquent
            {
                "SK_ID_BUREAU": 102,
                "MONTHS_BALANCE": -10,
                "STATUS": "0",
            },
            {
                "SK_ID_BUREAU": 102,
                "MONTHS_BALANCE": -2,
                "STATUS": "0",
            },
            # Client 2
            {
                "SK_ID_BUREAU": 201,
                "MONTHS_BALANCE": -3,
                "STATUS": "3",
            },
        ]
    )


def test_requires_bureau_mapping_when_curr_id_is_missing(
    bureau_balance_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError):
        build_bureau_balance_features(bureau_balance_df)


def test_input_dataframes_are_not_mutated(
    bureau_balance_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
) -> None:
    original_balance = bureau_balance_df.copy(deep=True)
    original_bureau = bureau_df.copy(deep=True)

    build_bureau_balance_features(
        bureau_balance_df,
        bureau_df,
    )

    pd.testing.assert_frame_equal(
        bureau_balance_df,
        original_balance,
    )
    pd.testing.assert_frame_equal(
        bureau_df,
        original_bureau,
    )


def test_bureau_balance_output_contract(
    bureau_balance_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
) -> None:
    features = build_bureau_balance_features(
        bureau_balance_df,
        bureau_df,
    )

    assert features["SK_ID_CURR"].is_unique
    assert list(features.columns) == [
        "SK_ID_CURR",
        *BBX_ACCEPTED_FEATURES,
    ]


def test_future_rows_do_not_leak_into_severity(
    bureau_balance_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
) -> None:
    features = build_bureau_balance_features(
        bureau_balance_df,
        bureau_df,
    )
    indexed = features.set_index("SK_ID_CURR")

    # Valid maximum severity for client 1 is 2.
    # STATUS=5 exists only in MONTHS_BALANCE=+1 and must be excluded.
    assert indexed.at[1, "BBX_MAX_SEVERITY"] == pytest.approx(2.0)

    # Most recent delinquency is at month -1.
    assert indexed.at[
        1,
        "BBX_MIN_MONTHS_SINCE_DELINQUENCY",
    ] == pytest.approx(1.0)


def test_recent_worsening_is_calculated_per_bureau_account(
    bureau_balance_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
) -> None:
    features = build_bureau_balance_features(
        bureau_balance_df,
        bureau_df,
    )
    indexed = features.set_index("SK_ID_CURR")

    # Account 101:
    # full delinquency share = 2/3
    # recent 6m share = 2/2 = 1
    # worsening = 1 - 2/3 = 1/3
    #
    # Account 102:
    # worsening = 0
    #
    # client mean = (1/3 + 0) / 2 = 1/6
    assert indexed.at[
        1,
        "BBX_MEAN_RECENT_WORSENING",
    ] == pytest.approx(1 / 6)

    assert indexed.at[
        1,
        "BBX_MAX_RECENT_WORSENING",
    ] == pytest.approx(1 / 3)

    # One of two bureau accounts is currently delinquent in recent 6m.
    assert indexed.at[
        1,
        "BBX_RECENT_DELINQUENT_ACCOUNT_SHARE",
    ] == pytest.approx(0.5)
