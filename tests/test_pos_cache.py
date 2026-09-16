import pandas as pd
import pytest

from home_credit.features.pos_cash import (
    POS_ACCEPTED_FEATURES,
    build_pos_cash_features,
)


@pytest.fixture
def pos_cash_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Client 1, contract 101
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "MONTHS_BALANCE": -10,
                "CNT_INSTALMENT": 10,
                "CNT_INSTALMENT_FUTURE": 8,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "MONTHS_BALANCE": -5,
                "CNT_INSTALMENT": 10,
                "CNT_INSTALMENT_FUTURE": 5,
                "SK_DPD": 10,
                "SK_DPD_DEF": 0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "MONTHS_BALANCE": -1,
                "CNT_INSTALMENT": 10,
                "CNT_INSTALMENT_FUTURE": 2,
                "SK_DPD": 20,
                "SK_DPD_DEF": 5,
            },
            # Future row: must be ignored
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "MONTHS_BALANCE": 1,
                "CNT_INSTALMENT": 10,
                "CNT_INSTALMENT_FUTURE": 1,
                "SK_DPD": 999,
                "SK_DPD_DEF": 999,
            },
            # Client 1, contract 102
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 102,
                "MONTHS_BALANCE": -8,
                "CNT_INSTALMENT": 5,
                "CNT_INSTALMENT_FUTURE": 5,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 102,
                "MONTHS_BALANCE": -2,
                "CNT_INSTALMENT": 5,
                "CNT_INSTALMENT_FUTURE": 1,
                "SK_DPD": 0,
                "SK_DPD_DEF": 0,
            },
            # Client 2
            {
                "SK_ID_CURR": 2,
                "SK_ID_PREV": 201,
                "MONTHS_BALANCE": -3,
                "CNT_INSTALMENT": 4,
                "CNT_INSTALMENT_FUTURE": 2,
                "SK_DPD": 40,
                "SK_DPD_DEF": 10,
            },
        ]
    )


def test_input_dataframe_is_not_mutated(
    pos_cash_df: pd.DataFrame,
) -> None:
    original = pos_cash_df.copy(deep=True)

    build_pos_cash_features(pos_cash_df)

    pd.testing.assert_frame_equal(pos_cash_df, original)


def test_pos_output_contract(
    pos_cash_df: pd.DataFrame,
) -> None:
    features = build_pos_cash_features(pos_cash_df)

    assert features["SK_ID_CURR"].is_unique
    assert list(features.columns) == [
        "SK_ID_CURR",
        *POS_ACCEPTED_FEATURES,
    ]


def test_latest_state_and_future_rows(
    pos_cash_df: pd.DataFrame,
) -> None:
    features = build_pos_cash_features(pos_cash_df)
    indexed = features.set_index("SK_ID_CURR")

    # Contract 101 latest valid month = -1 -> DPD 20
    # Contract 102 latest valid month = -2 -> DPD 0
    #
    # mean = 10, max = 20
    #
    # If MONTHS_BALANCE=+1 leaked in, these values would be corrupted by DPD=999.
    assert indexed.at[1, "POSX_MEAN_LATEST_DPD"] == pytest.approx(10.0)
    assert indexed.at[1, "POSX_MAX_LATEST_DPD"] == pytest.approx(20.0)
    assert indexed.at[1, "POSX_BAD_LATEST_SHARE"] == pytest.approx(0.5)


def test_progress_and_recent_worsening(
    pos_cash_df: pd.DataFrame,
) -> None:
    features = build_pos_cash_features(pos_cash_df)
    indexed = features.set_index("SK_ID_CURR")

    # Contract 101 progress:
    # 8/10 - 2/10 = 0.6
    #
    # Contract 102 progress:
    # 5/5 - 1/5 = 0.8
    #
    # Client mean = 0.7, min = 0.6
    assert indexed.at[1, "POSX_MEAN_PROGRESS"] == pytest.approx(0.7)
    assert indexed.at[1, "POSX_MIN_PROGRESS"] == pytest.approx(0.6)

    # Contract 101:
    # full DPD share = 2/3
    # recent 6m DPD share = 2/2 = 1
    # worsening = 1 - 2/3 = 1/3
    #
    # Contract 102 worsening = 0
    #
    # client mean = (1/3 + 0) / 2 = 1/6
    assert indexed.at[1, "POSX_MEAN_RECENT_WORSENING"] == pytest.approx(1 / 6)
    assert indexed.at[1, "POSX_MAX_RECENT_WORSENING"] == pytest.approx(1 / 3)
