import pandas as pd
import pytest

import home_credit.features.assemble as assemble_module
from home_credit.features.assemble import assemble_features
from home_credit.features.bureau import BUREAU_COUNT_FEATURES


@pytest.fixture(autouse=True)
def mock_application_feature_builder(monkeypatch):
    """
    Application feature engineering is tested separately.
    Here we test only assemble_features mechanics.
    """
    monkeypatch.setattr(
        assemble_module,
        "build_application_features",
        lambda df: df.copy(),
    )


@pytest.fixture
def application_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "BASE_FEATURE": [10.0, 20.0],
        }
    )


def test_left_join_preserves_applicants_and_feature_order(
    application_df: pd.DataFrame,
) -> None:
    historical_features = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            "HIST_FEATURE": [100.0],
        }
    )

    result = assemble_features(
        application=application_df,
        previous_application_features_df=historical_features,
        accepted_features=["BASE_FEATURE", "HIST_FEATURE"],
        include_id=True,
    )

    assert list(result.columns) == [
        "SK_ID_CURR",
        "BASE_FEATURE",
        "HIST_FEATURE",
    ]

    assert len(result) == 2
    assert result["SK_ID_CURR"].tolist() == [1, 2]

    # Client 2 has no historical record but must remain after LEFT JOIN.
    assert pd.isna(result.set_index("SK_ID_CURR").at[2, "HIST_FEATURE"])


def test_include_id_false_moves_id_to_index(
    application_df: pd.DataFrame,
) -> None:
    result = assemble_features(
        application=application_df,
        accepted_features=["BASE_FEATURE"],
        include_id=False,
    )

    assert "SK_ID_CURR" not in result.columns
    assert result.index.name == "SK_ID_CURR"
    assert result.index.tolist() == [1, 2]
    assert list(result.columns) == ["BASE_FEATURE"]


def test_duplicate_applicant_in_feature_table_raises(
    application_df: pd.DataFrame,
) -> None:
    duplicated_features = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1],
            "HIST_FEATURE": [100.0, 200.0],
        }
    )

    with pytest.raises(pd.errors.MergeError):
        assemble_features(
            application=application_df,
            previous_application_features_df=duplicated_features,
            accepted_features=["BASE_FEATURE", "HIST_FEATURE"],
        )


def test_missing_required_feature_raises_key_error(
    application_df: pd.DataFrame,
) -> None:
    with pytest.raises(KeyError, match="Missing"):
        assemble_features(
            application=application_df,
            accepted_features=[
                "BASE_FEATURE",
                "NON_EXISTENT_FEATURE",
            ],
        )


def test_missing_bureau_count_is_filled_with_zero(
    application_df: pd.DataFrame,
) -> None:
    count_feature = BUREAU_COUNT_FEATURES[0]

    bureau_features = pd.DataFrame(
        {
            "SK_ID_CURR": [1],
            count_feature: [3],
        }
    )

    result = assemble_features(
        application=application_df,
        bureau_features_df=bureau_features,
        accepted_features=["BASE_FEATURE", count_feature],
        include_id=True,
    )

    indexed = result.set_index("SK_ID_CURR")

    assert indexed.at[1, count_feature] == 3
    assert indexed.at[2, count_feature] == 0
