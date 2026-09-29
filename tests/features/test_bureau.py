import pandas as pd
import pytest

from home_credit.features.bureau import BUREAU_ACCEPTED_FEATURES, build_bureau_features


@pytest.fixture
def bureau_df() -> pd.DataFrame:
    test_bureau = pd.DataFrame(
        [
            {
                "SK_ID_CURR": 1,
                "SK_ID_BUREAU": 101,
                "DAYS_CREDIT": -100,
                "DAYS_CREDIT_UPDATE": -5,
                "CREDIT_ACTIVE": "Active",
                "AMT_CREDIT_SUM": 1000.0,
                "AMT_CREDIT_SUM_LIMIT": 200.0,
                "AMT_CREDIT_SUM_DEBT": 300.0,
                "AMT_CREDIT_MAX_OVERDUE": 0.0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_BUREAU": 102,
                "DAYS_CREDIT": -400,
                "DAYS_CREDIT_UPDATE": -20,
                "CREDIT_ACTIVE": "Closed",
                "AMT_CREDIT_SUM": 2000.0,
                "AMT_CREDIT_SUM_LIMIT": 0.0,
                "AMT_CREDIT_SUM_DEBT": 0.0,
                "AMT_CREDIT_MAX_OVERDUE": 150.0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_BUREAU": 103,
                "DAYS_CREDIT": -800,
                "DAYS_CREDIT_UPDATE": -150,
                "CREDIT_ACTIVE": "Sold",
                "AMT_CREDIT_SUM": 3000.0,
                "AMT_CREDIT_SUM_LIMIT": 0.0,
                "AMT_CREDIT_SUM_DEBT": 0.0,
                "AMT_CREDIT_MAX_OVERDUE": 500.0,
            },
            {
                "SK_ID_CURR": 1,
                "SK_ID_BUREAU": 104,
                "DAYS_CREDIT": -20,
                "DAYS_CREDIT_UPDATE": 10,  # leakage
                "CREDIT_ACTIVE": "Active",
                "AMT_CREDIT_SUM": 1_000_000.0,
                "AMT_CREDIT_SUM_LIMIT": 100_000.0,
                "AMT_CREDIT_SUM_DEBT": 500_000.0,
                "AMT_CREDIT_MAX_OVERDUE": 50_000.0,
            },
            {
                "SK_ID_CURR": 2,
                "SK_ID_BUREAU": 201,
                "DAYS_CREDIT": -50,
                "DAYS_CREDIT_UPDATE": 10,
                "CREDIT_ACTIVE": "Active",
                "AMT_CREDIT_SUM": 5000.0,
                "AMT_CREDIT_SUM_LIMIT": 500.0,
                "AMT_CREDIT_SUM_DEBT": 1000.0,
                "AMT_CREDIT_MAX_OVERDUE": 0.0,
            },
        ]
    )
    return test_bureau


def test_input_dataframe_is_not_mutated(bureau_df: pd.DataFrame) -> None:
    original_df = bureau_df.copy(deep=True)

    build_bureau_features(original_df)

    pd.testing.assert_frame_equal(original_df, bureau_df)


def test_unique_applicant_id(bureau_df: pd.DataFrame) -> None:
    df = build_bureau_features(bureau_df)

    assert df["SK_ID_CURR"].duplicated().sum() == 0


def test_temporal_cutoff(bureau_df: pd.DataFrame) -> None:
    df = build_bureau_features(bureau_df)

    assert 1 in df["SK_ID_CURR"].values

    assert 2 not in df["SK_ID_CURR"].values


def test_credit_status_aggregation(bureau_df: pd.DataFrame) -> None:
    df = build_bureau_features(bureau_df)
    row = df.loc[df["SK_ID_CURR"] == 1].iloc[0]

    assert row["BUREAU_CREDIT_COUNT"] == 3
    assert row["BUREAU_ACTIVE_COUNT"] == 1
    assert row["BUREAU_CLOSED_COUNT"] == 1
    assert row["BUREAU_SOLD_COUNT"] == 1
    assert row["BUREAU_ACTIVE_SHARE"] == pytest.approx(1 / 3)


def test_recency_aggregation(bureau_df: pd.DataFrame) -> None:
    df = build_bureau_features(bureau_df)
    row = df.loc[df["SK_ID_CURR"] == 1].iloc[0]

    assert row["BUREAU_DAYS_SINCE_LATEST_CREDIT"] == 100
    assert row["BUREAU_HISTORY_AGE_DAYS"] == 800
    assert row["BUREAU_CREDITS_LAST_180D"] == 1
    assert row["BUREAU_CREDITS_LAST_365D"] == 1
    assert row["BUREAU_CREDITS_LAST_730D"] == 2


def test_monetary_aggregation(bureau_df: pd.DataFrame) -> None:
    df = build_bureau_features(bureau_df)
    row = df.loc[df["SK_ID_CURR"] == 1].iloc[0]

    assert row["BUREAU_TOTAL_CREDIT_SUM"] == pytest.approx(6000.0)
    assert row["BUREAU_MAX_CREDIT_SUM"] == pytest.approx(3000.0)
    assert row["BUREAU_MEAN_CREDIT_SUM"] == pytest.approx(2000.0)
    assert row["BUREAU_TOTAL_CREDIT_LIMIT"] == pytest.approx(200.0)
    assert row["BUREAU_TOTAL_CREDIT_DEBT"] == pytest.approx(300.0)
    assert row["MAX_CREDIT_OVERDUE_AMT"] == pytest.approx(500.0)


def test_bureau_schema(bureau_df: pd.DataFrame) -> None:
    df = build_bureau_features(bureau_df)

    assert list(df.columns) == ["SK_ID_CURR", *BUREAU_ACCEPTED_FEATURES]
