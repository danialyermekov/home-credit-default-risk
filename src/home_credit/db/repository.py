from sqlalchemy import Engine, select

from home_credit.db.tables import applicant_features
from home_credit.schema_features import ACCEPTED_FINAL_FEATURES


class FeatureRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_features(
        self,
        applicant_id: int,
    ) -> dict[str, object] | None:
        feature_columns = [
            applicant_features.c[feature] for feature in ACCEPTED_FINAL_FEATURES
        ]

        statement = select(
            *feature_columns,
            applicant_features.c.feature_version,
        ).where(applicant_features.c.SK_ID_CURR == applicant_id)

        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().one_or_none()

        if row is None:
            return None

        return dict(row)
