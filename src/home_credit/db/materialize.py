from collections.abc import Iterator

import pandas as pd
from sqlalchemy import Engine, func
from sqlalchemy.dialects.postgresql import insert

from home_credit.db.tables import applicant_features
from home_credit.schema_features import ACCEPTED_FINAL_FEATURES


def _batches(
    records: list[dict],
    batch_size: int,
) -> Iterator[list[dict]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def materialize_features(
    engine: Engine,
    features: pd.DataFrame,
    feature_version: str,
    batch_size: int = 200,
) -> None:
    df = features.copy()

    if "SK_ID_CURR" not in df.columns:
        if df.index.name != "SK_ID_CURR":
            raise ValueError("SK_ID_CURR must be a column or named index")

        df = df.reset_index()

    expected_columns = ["SK_ID_CURR", *ACCEPTED_FINAL_FEATURES]

    missing = [column for column in expected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]}")

    df = df[expected_columns].copy()

    df["feature_version"] = feature_version

    # PostgreSQL expects NULL rather than pandas NaN/NA.
    df = df.astype(object).where(pd.notna(df), None)

    records = df.to_dict(orient="records")

    update_columns = {
        column.name: getattr(insert(applicant_features).excluded, column.name)
        for column in applicant_features.columns
        if column.name
        not in {
            "SK_ID_CURR",
            "updated_at",
        }
    }
    update_columns["updated_at"] = func.now()

    with engine.begin() as connection:
        for batch in _batches(records, batch_size):
            statement = insert(applicant_features).values(batch)

            statement = statement.on_conflict_do_update(
                index_elements=["SK_ID_CURR"],
                set_=update_columns,
            )

            connection.execute(statement)
