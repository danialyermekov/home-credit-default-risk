from pathlib import Path

import pandas as pd

from home_credit.core.config import get_settings
from home_credit.db.connection import get_engine
from home_credit.db.materialize import materialize_features
from home_credit.db.tables import metadata
from home_credit.features.assemble import build_feature_dataset


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = data_dir / "raw"

    return {
        "applications": pd.read_csv(raw_dir / "application_test.csv"),
        "bureau": pd.read_csv(raw_dir / "bureau.csv"),
        "previous_application": pd.read_csv(raw_dir / "previous_application.csv"),
        "installments_payments": pd.read_csv(raw_dir / "installments_payments.csv"),
        "credit_card_balance": pd.read_csv(raw_dir / "credit_card_balance.csv"),
        "pos_cash_balance": pd.read_csv(raw_dir / "POS_CASH_balance.csv"),
        "bureau_balance": pd.read_csv(raw_dir / "bureau_balance.csv"),
    }


def build_features(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return build_feature_dataset(
        applications=tables["applications"],
        bureau=tables["bureau"],
        previous_application=tables["previous_application"],
        installments_payments=tables["installments_payments"],
        credit_card_balance=tables["credit_card_balance"],
        pos_cash_balance=tables["pos_cash_balance"],
        bureau_balance=tables["bureau_balance"],
        include_id=True,
    )


def main() -> None:
    settings = get_settings()
    engine = get_engine()

    metadata.create_all(engine)

    tables = load_raw_tables(Path("data"))
    features = build_features(tables)

    materialize_features(
        engine=engine,
        features=features,
        feature_version=settings.feature_version,
    )

    print(f"Materialized {len(features):,} applicants")


if __name__ == "__main__":
    main()
