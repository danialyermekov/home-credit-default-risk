"""Feature assembly module.

Assembles the complete feature matrix for application_test (or application_train)
by orchestrating application-level preprocessing and all six historical-table
feature engineering modules:
1. bureau
2. previous_application
3. installments
4. credit_card
5. pos_cash
6. bureau_balance

Guarantees that the resulting feature matrix has the exact column names, order,
and compatible dtypes defined by ACCEPTED_FINAL_FEATURES from schema.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from home_credit.features.application import (
    HOUSING_COLUMN_TOKENS,
    build_application_features,
)
from home_credit.features.bureau import (
    BUREAU_ACCEPTED_FEATURES,
    BUREAU_COUNT_FEATURES,
    build_bureau_features,
)
from home_credit.features.bureau_balance import (
    BBX_ACCEPTED_FEATURES,
    build_bureau_balance_features,
)
from home_credit.features.credit_card import (
    CREDIT_CARD_ACCEPTED_FEATURES,
    build_credit_card_features,
)
from home_credit.features.installments import (
    INSTALLMENTS_ACCEPTED_FEATURES,
    build_installments_features,
)
from home_credit.features.pos_cash import (
    POS_ACCEPTED_FEATURES,
    build_pos_cash_features,
)
from home_credit.features.previous_application import (
    PREVIOUS_APPLICATION_ACCEPTED_FEATURES,
    build_previous_application_features,
)
from home_credit.schema import (
    ACCEPTED_CATEGORICAL_FEATURES,
    ACCEPTED_FINAL_FEATURES,
)


def load_accepted_final_features(
    model_path: Path | str | None = None,
) -> list[str]:
    """Load the frozen ACCEPTED_FINAL_FEATURES list.

    If model_path is explicitly provided, loads feature_names from model artifact;
    otherwise returns the frozen 166-feature list from schema.py.

    Parameters
    ----------
    model_path : Path | str | None, optional
        Path to model artifact file. If None, returns schema list.

    Returns
    -------
    list[str]
        The 166 accepted final feature names in exact model order.
    """
    if model_path is None:
        return list(ACCEPTED_FINAL_FEATURES)

    p = Path(model_path)
    if p.exists():
        if p.suffix == ".txt":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("feature_names="):
                        return line.strip().split("=")[1].split()
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data["learner"]["feature_names"])
        elif p.suffix == ".cbm":
            from catboost import CatBoostClassifier

            cb = CatBoostClassifier()
            cb.load_model(str(p))
            return list(cb.feature_names_)

    return list(ACCEPTED_FINAL_FEATURES)


def assemble_features(
    application: pd.DataFrame,
    bureau_features_df: pd.DataFrame | None = None,
    bureau_balance_features_df: pd.DataFrame | None = None,
    previous_application_features_df: pd.DataFrame | None = None,
    credit_card_features_df: pd.DataFrame | None = None,
    installments_features_df: pd.DataFrame | None = None,
    pos_cash_features_df: pd.DataFrame | None = None,
    accepted_features: Sequence[str] | None = None,
    include_id: bool = False,
) -> pd.DataFrame:
    """Assemble all historical features onto the application table.

    Joins historical feature tables using one-to-one left joins, applies
    established missing-value semantics (such as zero-filling bureau counts),
    and aligns the output schema exactly with ACCEPTED_FINAL_FEATURES.

    Parameters
    ----------
    application : pd.DataFrame
        Application dataframe with base features.
    bureau_features_df : pd.DataFrame | None, optional
        Precomputed bureau features from build_bureau_features.
    bureau_balance_features_df : pd.DataFrame | None, optional
        Precomputed bureau_balance features from build_bureau_balance_features.
    previous_application_features_df : pd.DataFrame | None, optional
        Precomputed previous application features from build_previous_application_features.
    credit_card_features_df : pd.DataFrame | None, optional
        Precomputed credit card features from build_credit_card_features.
    installments_features_df : pd.DataFrame | None, optional
        Precomputed installments features from build_installments_features.
    pos_cash_features_df : pd.DataFrame | None, optional
        Precomputed POS features from build_pos_cash_features.
    accepted_features : Sequence[str] | None, optional
        Explicit feature list. If None, defaults to ACCEPTED_FINAL_FEATURES.
    include_id : bool, default False
        If True, SK_ID_CURR is included as a regular column.
        If False, SK_ID_CURR is set as the DataFrame index.

    Returns
    -------
    pd.DataFrame
        Complete model input matrix.
    """
    if accepted_features is None:
        final_feature_names = list(ACCEPTED_FINAL_FEATURES)
    else:
        final_feature_names = list(accepted_features)

    # 1. Base application features
    df = build_application_features(application)

    # 2. Bureau features
    if bureau_features_df is not None:
        df = df.merge(
            bureau_features_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )
        for col in BUREAU_COUNT_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(0)

    # 3. Previous application features
    if previous_application_features_df is not None:
        df = df.merge(
            previous_application_features_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

    # 4. Credit card balance features
    if credit_card_features_df is not None:
        df = df.merge(
            credit_card_features_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

    # 5. Installments payments features
    if installments_features_df is not None:
        df = df.merge(
            installments_features_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

    # 6. POS cash balance dynamic features
    if pos_cash_features_df is not None:
        df = df.merge(
            pos_cash_features_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

    # 7. Bureau balance dynamic features
    if bureau_balance_features_df is not None:
        df = df.merge(
            bureau_balance_features_df,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

    # Verify all expected model features are present
    missing_features = [f for f in final_feature_names if f not in df.columns]
    if missing_features:
        raise KeyError(
            f"Missing {len(missing_features)} features required by ACCEPTED_FINAL_FEATURES: "
            f"{missing_features[:10]}"
        )

    # Ensure categorical dtypes
    for cat_col in ACCEPTED_CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    if include_id:
        output_cols = ["SK_ID_CURR"] + final_feature_names
        result = df[output_cols].copy()
    else:
        result = df[final_feature_names].copy()
        result.index = df["SK_ID_CURR"]
        result.index.name = "SK_ID_CURR"

    return result


def build_feature_dataset(
    applications: pd.DataFrame,
    bureau: pd.DataFrame,
    previous_application: pd.DataFrame,
    installments_payments: pd.DataFrame,
    credit_card_balance: pd.DataFrame,
    pos_cash_balance: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    accepted_features: Sequence[str] | None = None,
    include_id: bool = False,
) -> pd.DataFrame:
    """Build the complete feature matrix by computing and assembling features from all raw tables.

    Parameters
    ----------
    applications : pd.DataFrame
        Raw application dataframe.
    bureau : pd.DataFrame
        Raw bureau dataframe.
    previous_application : pd.DataFrame
        Raw previous_application dataframe.
    installments_payments : pd.DataFrame
        Raw installments_payments dataframe.
    credit_card_balance : pd.DataFrame
        Raw credit_card_balance dataframe.
    pos_cash_balance : pd.DataFrame
        Raw POS_CASH_balance dataframe.
    bureau_balance : pd.DataFrame
        Raw bureau_balance dataframe.
    accepted_features : Sequence[str] | None, optional
        Explicit feature list (defaults to ACCEPTED_FINAL_FEATURES).
    include_id : bool, default False
        Whether to retain SK_ID_CURR as a column or index.

    Returns
    -------
    pd.DataFrame
        Complete feature matrix aligned with ACCEPTED_FINAL_FEATURES.
    """
    bureau_feat = build_bureau_features(bureau)
    bureau_bal_feat = build_bureau_balance_features(
        bureau_balance,
        bureau=bureau[["SK_ID_BUREAU", "SK_ID_CURR"]],
    )
    prev_feat = build_previous_application_features(previous_application)
    cc_feat = build_credit_card_features(credit_card_balance)
    ip_feat = build_installments_features(installments_payments)
    pos_feat = build_pos_cash_features(pos_cash_balance)

    return assemble_features(
        application=applications,
        bureau_features_df=bureau_feat,
        bureau_balance_features_df=bureau_bal_feat,
        previous_application_features_df=prev_feat,
        credit_card_features_df=cc_feat,
        installments_features_df=ip_feat,
        pos_cash_features_df=pos_feat,
        accepted_features=accepted_features,
        include_id=include_id,
    )


def build_test_features(
    data_dir: Path | str = "data",
    output_path: Path | str | None = None,
    include_id: bool = False,
) -> pd.DataFrame:
    """Build the complete feature matrix for application_test from raw tables.

    Parameters
    ----------
    data_dir : Path | str, default 'data'
        Root data directory containing raw/ subdirectory.
    output_path : Path | str | None, optional
        Path to save the resulting parquet file. If None, file is not saved to disk.
    include_id : bool, default False
        Whether to keep SK_ID_CURR as a column or index.

    Returns
    -------
    pd.DataFrame
        Complete test feature matrix aligned with ACCEPTED_FINAL_FEATURES.
    """
    raw_dir = Path(data_dir) / "raw" if (Path(data_dir) / "raw").exists() else Path(data_dir)

    print("Loading application_test.csv...")
    application_test = pd.read_csv(raw_dir / "application_test.csv")

    print("Building bureau features...")
    bureau = pd.read_csv(raw_dir / "bureau.csv")
    bureau_features = build_bureau_features(bureau)

    print("Building bureau_balance features...")
    bureau_balance = pd.read_csv(raw_dir / "bureau_balance.csv")
    bureau_balance_features = build_bureau_balance_features(
        bureau_balance,
        bureau=bureau[["SK_ID_BUREAU", "SK_ID_CURR"]],
    )
    del bureau_balance

    print("Building previous_application features...")
    previous_application = pd.read_csv(raw_dir / "previous_application.csv")
    previous_application_features = build_previous_application_features(previous_application)
    del previous_application

    print("Building credit_card features...")
    credit_card_balance = pd.read_csv(raw_dir / "credit_card_balance.csv")
    credit_card_features = build_credit_card_features(credit_card_balance)
    del credit_card_balance

    print("Building installments features...")
    installments_payments = pd.read_csv(raw_dir / "installments_payments.csv")
    installments_features = build_installments_features(installments_payments)
    del installments_payments

    print("Building pos_cash features...")
    pos_cash_balance = pd.read_csv(raw_dir / "POS_CASH_balance.csv")
    pos_cash_features = build_pos_cash_features(pos_cash_balance)
    del pos_cash_balance
    del bureau

    print("Assembling final test feature matrix...")
    test_features = assemble_features(
        application=application_test,
        bureau_features_df=bureau_features,
        bureau_balance_features_df=bureau_balance_features,
        previous_application_features_df=previous_application_features,
        credit_card_features_df=credit_card_features,
        installments_features_df=installments_features,
        pos_cash_features_df=pos_cash_features,
        include_id=include_id,
    )

    print(f"Assembly complete. Feature matrix shape: {test_features.shape}")

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        test_features.to_parquet(out_file, index=not include_id)
        print(f"Saved feature matrix to: {out_file}")

    return test_features


if __name__ == "__main__":
    build_test_features(
        output_path="data/processed/application_test_features.parquet",
    )
