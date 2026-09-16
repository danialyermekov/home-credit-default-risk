import numpy as np
import pandas as pd
import pytest

from home_credit.features.credit_card import (
    CREDIT_CARD_ACCEPTED_FEATURES,
    build_credit_card_features,
)
from home_credit.schema import ACCEPTED_FINAL_FEATURES


@pytest.fixture
def credit_card_df():
    return pd.DataFrame(
        [
            # Applicant 1, contract 101
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "MONTHS_BALANCE": -3,
                "AMT_BALANCE": 300.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 1000.0,
                "AMT_DRAWINGS_CURRENT": 100.0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 101,
                "MONTHS_BALANCE": -1,  # latest month for contract 101
                "AMT_BALANCE": 500.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 1000.0,
                "AMT_DRAWINGS_CURRENT": 0.0,
            },
            # Applicant 1, contract 102
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 102,
                "MONTHS_BALANCE": -4,
                "AMT_BALANCE": 200.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 500.0,
                "AMT_DRAWINGS_CURRENT": 50.0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_PREV": 102,
                "MONTHS_BALANCE": -2,  # latest month for contract 102
                "AMT_BALANCE": 0.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 0.0,  # utilization -> NaN
                "AMT_DRAWINGS_CURRENT": 0.0,
            },
            # Applicant 2
            {
                "SK_ID_CURR": 2,
                "SK_ID_PREV": 201,
                "MONTHS_BALANCE": -1,
                "AMT_BALANCE": 100.0,
                "AMT_CREDIT_LIMIT_ACTUAL": 200.0,
                "AMT_DRAWINGS_CURRENT": 20.0,
            },
        ]
    )


def test_origin_dataset_is_not_mutated(credit_card_df: pd.DataFrame) -> None:
    original = credit_card_df.copy(deep=True)
    build_credit_card_features(original)

    pd.testing.assert_frame_equal(original, credit_card_df)


def test_unique_applicant_id(credit_card_df: pd.DataFrame) -> None:
    credit_card = build_credit_card_features(credit_card_df)

    assert credit_card["SK_ID_CURR"].is_unique


def test_len_credit_card_schema() -> None:
    assert len(CREDIT_CARD_ACCEPTED_FEATURES) == 12


def test_no_duplicates_in_credit_card_schema() -> None:
    assert len(set(CREDIT_CARD_ACCEPTED_FEATURES)) == len(CREDIT_CARD_ACCEPTED_FEATURES)


def test_credit_card_schema_is_subset_of_final_features() -> None:
    assert set(CREDIT_CARD_ACCEPTED_FEATURES).issubset(ACCEPTED_FINAL_FEATURES)


def test_credit_card_output_schema(
    credit_card_df: pd.DataFrame,
) -> None:
    credit_card = build_credit_card_features(credit_card_df)

    assert list(credit_card.columns) == [
        "SK_ID_CURR",
        *CREDIT_CARD_ACCEPTED_FEATURES,
    ]


def test_no_inf_in_utilization(credit_card_df: pd.DataFrame) -> None:
    credit_card = build_credit_card_features(credit_card_df)
    utilization = ["CC_MEAN_UTILIZATION", "CC_MAX_UTILIZATION", "CC_LATEST_UTILIZATION"]

    assert not np.isinf(credit_card[utilization].to_numpy()).any()


def test_latest_credit_card_features(credit_card_df):
    credit_card = build_credit_card_features(credit_card_df)

    row = credit_card.set_index("SK_ID_CURR").loc[1]

    assert row["CC_LATEST_BALANCE"] == pytest.approx(250.0)
    assert row["CC_LATEST_CREDIT_LIMIT"] == pytest.approx(500.0)
    assert row["CC_LATEST_UTILIZATION"] == pytest.approx(0.5)
