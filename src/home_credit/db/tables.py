from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    func,
)

from home_credit.schema_features import (
    ACCEPTED_CATEGORICAL_FEATURES,
    ACCEPTED_FINAL_FEATURES,
)

metadata = MetaData()

categorical_features = set(ACCEPTED_CATEGORICAL_FEATURES)


def build_feature_columns() -> list[Column]:
    columns = []

    for feature_name in ACCEPTED_FINAL_FEATURES:
        if feature_name in categorical_features:
            column = Column(feature_name, String, nullable=True)
        else:
            column = Column(feature_name, Float, nullable=True)

        columns.append(column)

    return columns


applicant_features = Table(
    "applicant_features",
    metadata,
    Column("SK_ID_CURR", Integer, primary_key=True),
    *build_feature_columns(),
    Column("feature_version", String(50), nullable=False),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)
