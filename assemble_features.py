"""Feature assembly module.

Assembles the complete feature matrix for application_test (or application_train)
by orchestrating the application-level preprocessing and all six historical-table
feature engineering modules:
1. bureau_features
2. previous_application_features
3. installments_features
4. credit_card_features
5. pos_cash_features
6. bureau_balance_features

Guarantees that the resulting feature matrix has the exact column names, order,
and compatible dtypes defined by ACCEPTED_FINAL_FEATURES from modeling.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bureau_balance_features import build_bureau_balance_features
from bureau_features import BUREAU_COUNT_FEATURES, build_bureau_features
from credit_card_features import build_credit_card_features
from installments_features import build_installments_features
from pos_cash_features import build_pos_cash_features
from previous_application_features import build_previous_application_features

# Tokens used to identify raw housing columns for computing HOUSING_INFO_MISSING_PCT
HOUSING_COLUMN_TOKENS: list[str] = [
    "APARTMENTS",
    "BASEMENTAREA",
    "COMMONAREA",
    "ELEVATORS",
    "ENTRANCES",
    "FLOORSMAX",
    "FLOORSMIN",
    "LANDAREA",
    "LIVINGAPARTMENTS",
    "LIVINGAREA",
    "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
    "YEARS_BUILD",
    "YEARS_BEGINEXPLUATATION",
]

# The 15 categorical features expected by final models
ACCEPTED_CATEGORICAL_FEATURES: list[str] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]


def load_accepted_final_features(
    model_path: Path | str = "artifacts/lightgbm_full_model.txt",
) -> list[str]:
    """Load the source-of-truth ACCEPTED_FINAL_FEATURES list from trained model artifacts.

    Parameters
    ----------
    model_path : Path | str, optional
        Path to lightgbm model artifact containing feature_names.
        Falls back to xgboost_full_model.json or catboost_full_model.cbm if not found.

    Returns
    -------
    list[str]
        The 166 accepted final feature names in exact model order.
    """
    lgb_path = Path(model_path)
    if lgb_path.exists():
        with open(lgb_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("feature_names="):
                    return line.strip().split("=")[1].split()

    xgb_path = lgb_path.parent / "xgboost_full_model.json"
    if xgb_path.exists():
        with open(xgb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return list(data["learner"]["feature_names"])

    cb_path = lgb_path.parent / "catboost_full_model.cbm"
    if cb_path.exists():
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier()
        cb.load_model(str(cb_path))
        return list(cb.feature_names_)

    raise FileNotFoundError(
        f"Could not load ACCEPTED_FINAL_FEATURES from {model_path} or alternate artifacts."
    )


def build_application_features(
    application: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare application-level features and transformations from raw application data.

    Preserves the exact preprocessing validated in EDA and modeling:
    - Replaces 365243 anomaly in DAYS_EMPLOYED with NaN
    - Computes AGE_YEARS and EMPLOYED_YEARS
    - Computes financial ratios: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, ANNUITY_CREDIT_RATIO
    - Computes HOUSING_INFO_MISSING_PCT across housing fields
    - Replaces missing values in categorical columns with '__MISSING__'
    - Sets categorical columns to category dtype

    Parameters
    ----------
    application : pd.DataFrame
        Raw application dataframe (e.g. application_test.csv or application_train.csv).

    Returns
    -------
    pd.DataFrame
        Cleaned application dataframe with engineered features.
    """
    df = application.copy()

    # Anomaly handling: 365243 in DAYS_EMPLOYED indicates missing/unemployed
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Demographic and employment duration
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    # Core financial ratios
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # Housing missingness share
    housing_cols = [
        col
        for col in application.columns
        if any(token in col for token in HOUSING_COLUMN_TOKENS)
    ]
    if housing_cols:
        df["HOUSING_INFO_MISSING_PCT"] = application[housing_cols].isna().mean(axis=1)

    # Categorical missingness representation
    for cat_col in ACCEPTED_CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna("__MISSING__").astype("category")

    return df


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
        Explicit feature list. If None, loaded from ACCEPTED_FINAL_FEATURES artifact.
    include_id : bool, default False
        If True, SK_ID_CURR is included as a regular column.
        If False, SK_ID_CURR is set as the DataFrame index.

    Returns
    -------
    pd.DataFrame
        Complete model input matrix.
    """
    if accepted_features is None:
        final_feature_names = load_accepted_final_features()
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
    raw_dir = Path(data_dir) / "raw"

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
