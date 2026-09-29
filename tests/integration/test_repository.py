import pandas as pd
import pytest
from sqlalchemy import Engine

from home_credit.db.materialize import materialize_features
from home_credit.db.repository import FeatureRepository
from home_credit.schema_features import (
    ACCEPTED_CATEGORICAL_FEATURES,
    ACCEPTED_FINAL_FEATURES,
)


def get_numeric_feature() -> str:
    categorical_features = set(ACCEPTED_CATEGORICAL_FEATURES)

    return next(
        feature
        for feature in ACCEPTED_FINAL_FEATURES
        if feature not in categorical_features
    )


def test_repository_returns_features(
    engine: Engine,
    feature_df: pd.DataFrame,
    test_applicant_id: int,
    clean_applicant_features: None,
) -> None:
    materialize_features(
        engine=engine,
        features=feature_df,
        feature_version="v1",
    )

    repository = FeatureRepository(engine)

    result = repository.get_features(test_applicant_id)

    assert result is not None
    assert result["feature_version"] == "v1"
    assert list(result) == [
        *ACCEPTED_FINAL_FEATURES,
        "feature_version",
    ]


def test_repository_returns_feature_values(
    engine: Engine,
    feature_df: pd.DataFrame,
    test_applicant_id: int,
    clean_applicant_features: None,
) -> None:
    feature_name = get_numeric_feature()

    feature_df = feature_df.copy()
    feature_df[feature_name] = 42.0

    materialize_features(
        engine=engine,
        features=feature_df,
        feature_version="v1",
    )

    repository = FeatureRepository(engine)

    result = repository.get_features(test_applicant_id)

    assert result is not None
    assert result[feature_name] == pytest.approx(42.0)


def test_repository_returns_none_for_unknown_applicant(
    engine: Engine,
) -> None:
    repository = FeatureRepository(engine)

    result = repository.get_features(-1)

    assert result is None
