import pandas as pd
import pytest
from sqlalchemy import Engine, func, select

from home_credit.db.materialize import materialize_features
from home_credit.db.tables import applicant_features
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


def fetch_materialized_row(
    engine: Engine,
    applicant_id: int,
    feature_name: str,
) -> dict[str, object]:
    statement = select(
        applicant_features.c.SK_ID_CURR,
        applicant_features.c.feature_version,
        applicant_features.c[feature_name],
    ).where(applicant_features.c.SK_ID_CURR == applicant_id)

    with engine.connect() as connection:
        row = connection.execute(statement).mappings().one()

    return dict(row)


def count_applicant_rows(
    engine: Engine,
    applicant_id: int,
) -> int:
    statement = (
        select(func.count())
        .select_from(applicant_features)
        .where(applicant_features.c.SK_ID_CURR == applicant_id)
    )

    with engine.connect() as connection:
        return connection.execute(statement).scalar_one()


def test_materialize_inserts_features(
    engine: Engine,
    feature_df: pd.DataFrame,
    test_applicant_id: int,
    clean_applicant_features: None,
) -> None:
    feature_name = get_numeric_feature()

    materialize_features(
        engine=engine,
        features=feature_df,
        feature_version="v1",
    )

    row = fetch_materialized_row(
        engine,
        test_applicant_id,
        feature_name,
    )

    assert row["SK_ID_CURR"] == test_applicant_id
    assert row["feature_version"] == "v1"
    assert row[feature_name] == pytest.approx(1.0)


def test_materialize_upserts_existing_applicant(
    engine: Engine,
    feature_df: pd.DataFrame,
    test_applicant_id: int,
    clean_applicant_features: None,
) -> None:
    feature_name = get_numeric_feature()

    materialize_features(
        engine=engine,
        features=feature_df,
        feature_version="v1",
    )

    updated_df = feature_df.copy()
    updated_df[feature_name] = 42.0

    materialize_features(
        engine=engine,
        features=updated_df,
        feature_version="v2",
    )

    row = fetch_materialized_row(
        engine,
        test_applicant_id,
        feature_name,
    )

    assert count_applicant_rows(engine, test_applicant_id) == 1
    assert row["feature_version"] == "v2"
    assert row[feature_name] == pytest.approx(42.0)
