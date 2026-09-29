from collections.abc import Iterator

import pandas as pd
import pytest
from sqlalchemy import Engine, delete

from home_credit.db.connection import get_engine
from home_credit.db.tables import applicant_features, metadata
from home_credit.schema_features import (
    ACCEPTED_CATEGORICAL_FEATURES,
    ACCEPTED_FINAL_FEATURES,
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:

    engine = get_engine()
    metadata.create_all(engine)

    yield engine

    engine.dispose()


@pytest.fixture
def test_applicant_id() -> int:
    return 999_999_999


@pytest.fixture
def feature_df(test_applicant_id: int) -> pd.DataFrame:
    categorical_features = set(ACCEPTED_CATEGORICAL_FEATURES)

    row: dict[str, object] = {
        "SK_ID_CURR": test_applicant_id,
    }

    for feature in ACCEPTED_FINAL_FEATURES:
        if feature in categorical_features:
            row[feature] = "TEST"
        else:
            row[feature] = 1.0

    return pd.DataFrame([row])


@pytest.fixture
def clean_applicant_features(
    engine: Engine,
    test_applicant_id: int,
) -> Iterator[None]:
    def delete_test_row() -> None:
        with engine.begin() as connection:
            connection.execute(
                delete(applicant_features).where(
                    applicant_features.c.SK_ID_CURR == test_applicant_id
                )
            )

    # In case a previous interrupted test left data behind.
    delete_test_row()

    yield

    delete_test_row()
