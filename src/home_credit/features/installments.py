"""Installments payments feature engineering module.

Extracts repayment discipline, payment shortfall, delay, and temporal recency
features from the Home Credit installments_payments table, preserving the exact
installment-level aggregation and recent-window semantics validated during
model development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The 19 accepted installments_payments features present in ACCEPTED_FINAL_FEATURES
INSTALLMENTS_ACCEPTED_FEATURES: list[str] = [
    "IP_CONTRACT_COUNT",
    "IP_PAYMENT_RECORD_COUNT",
    "IP_UNIQUE_INSTALLMENT_COUNT",
    "IP_HISTORY_AGE_DAYS",
    "IP_DAYS_SINCE_LAST_PAYMENT",
    "IP_LATE_INSTALLMENT_SHARE",
    "IP_MEAN_DELAY_DAYS",
    "IP_MAX_DELAY_DAYS",
    "IP_UNDERPAID_INSTALLMENT_SHARE",
    "IP_MEAN_PAYMENT_SHORTFALL",
    "IP_MEAN_PAYMENT_COVERAGE_RATIO",
    "IP_RECENT_6M_INSTALLMENT_COUNT",
    "IP_RECENT_6M_LATE_SHARE",
    "IP_RECENT_6M_MAX_DELAY_DAYS",
    "IP_RECENT_6M_UNDERPAID_SHARE",
    "IP_RECENT_12M_INSTALLMENT_COUNT",
    "IP_RECENT_12M_LATE_SHARE",
    "IP_RECENT_12M_MAX_DELAY_DAYS",
    "IP_RECENT_12M_UNDERPAID_SHARE",
]


def build_installments_features(
    installments_payments: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accepted installments features aggregated by applicant (SK_ID_CURR).

    Parameters
    ----------
    installments_payments : pd.DataFrame
        Raw or preloaded installments_payments table.

    Returns
    -------
    pd.DataFrame
        Aggregated features containing SK_ID_CURR and the 19 accepted
        installments_payments features.
    """
    ip = installments_payments

    # IP1: Contract and payment history structure
    ip1 = (
        ip.groupby("SK_ID_CURR")
        .agg(
            IP_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
            IP_PAYMENT_RECORD_COUNT=("SK_ID_PREV", "size"),
            IP_HISTORY_AGE_DAYS=("DAYS_ENTRY_PAYMENT", lambda x: -x.min()),
            IP_DAYS_SINCE_LAST_PAYMENT=("DAYS_ENTRY_PAYMENT", lambda x: -x.max()),
        )
        .reset_index()
    )

    unique_installments = (
        ip[["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]]
        .drop_duplicates()
        .groupby("SK_ID_CURR")
        .size()
        .rename("IP_UNIQUE_INSTALLMENT_COUNT")
        .reset_index()
    )
    ip1 = ip1.merge(unique_installments, on="SK_ID_CURR", how="left")

    # IP2: Aggregate multiple payment entries to single scheduled installment level
    installment_level = ip.groupby(
        [
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
        ],
        as_index=False,
    ).agg(
        DAYS_INSTALMENT=("DAYS_INSTALMENT", "first"),
        LAST_PAYMENT_DAY=("DAYS_ENTRY_PAYMENT", "max"),
        AMT_INSTALMENT=("AMT_INSTALMENT", "first"),
        TOTAL_PAYMENT=("AMT_PAYMENT", "sum"),
    )

    installment_level["IP_DELAY_DAYS"] = (
        installment_level["LAST_PAYMENT_DAY"] - installment_level["DAYS_INSTALMENT"]
    ).clip(lower=0)

    installment_level["IP_IS_LATE"] = (installment_level["IP_DELAY_DAYS"] > 0).astype(
        "int8"
    )

    installment_level["IP_PAYMENT_SHORTFALL"] = (
        installment_level["AMT_INSTALMENT"] - installment_level["TOTAL_PAYMENT"]
    ).clip(lower=0)

    installment_level["IP_IS_UNDERPAID"] = (
        installment_level["IP_PAYMENT_SHORTFALL"] > 0
    ).astype("int8")

    installment_level["IP_PAYMENT_COVERAGE_RATIO"] = (
        installment_level["TOTAL_PAYMENT"]
        / installment_level["AMT_INSTALMENT"].replace(0, np.nan)
    ).clip(upper=1)

    ip2 = (
        installment_level.groupby("SK_ID_CURR")
        .agg(
            IP_LATE_INSTALLMENT_SHARE=("IP_IS_LATE", "mean"),
            IP_MEAN_DELAY_DAYS=("IP_DELAY_DAYS", "mean"),
            IP_MAX_DELAY_DAYS=("IP_DELAY_DAYS", "max"),
            IP_UNDERPAID_INSTALLMENT_SHARE=("IP_IS_UNDERPAID", "mean"),
            IP_MEAN_PAYMENT_SHORTFALL=("IP_PAYMENT_SHORTFALL", "mean"),
            IP_MEAN_PAYMENT_COVERAGE_RATIO=("IP_PAYMENT_COVERAGE_RATIO", "mean"),
        )
        .reset_index()
    )

    # IP3: Recent repayment discipline (last 6 months and last 12 months)
    recent_6m = installment_level.loc[installment_level["DAYS_INSTALMENT"] >= -180]
    recent_12m = installment_level.loc[installment_level["DAYS_INSTALMENT"] >= -365]

    ip_recent_6m = (
        recent_6m.groupby("SK_ID_CURR")
        .agg(
            IP_RECENT_6M_INSTALLMENT_COUNT=("NUM_INSTALMENT_NUMBER", "size"),
            IP_RECENT_6M_LATE_SHARE=("IP_IS_LATE", "mean"),
            IP_RECENT_6M_MAX_DELAY_DAYS=("IP_DELAY_DAYS", "max"),
            IP_RECENT_6M_UNDERPAID_SHARE=("IP_IS_UNDERPAID", "mean"),
        )
        .reset_index()
    )

    ip_recent_12m = (
        recent_12m.groupby("SK_ID_CURR")
        .agg(
            IP_RECENT_12M_INSTALLMENT_COUNT=("NUM_INSTALMENT_NUMBER", "size"),
            IP_RECENT_12M_LATE_SHARE=("IP_IS_LATE", "mean"),
            IP_RECENT_12M_MAX_DELAY_DAYS=("IP_DELAY_DAYS", "max"),
            IP_RECENT_12M_UNDERPAID_SHARE=("IP_IS_UNDERPAID", "mean"),
        )
        .reset_index()
    )

    ip3_history = ip_recent_6m.merge(ip_recent_12m, on="SK_ID_CURR", how="outer")

    features = ip1.merge(ip2, on="SK_ID_CURR", how="left").merge(
        ip3_history, on="SK_ID_CURR", how="left"
    )

    output_cols = ["SK_ID_CURR"] + INSTALLMENTS_ACCEPTED_FEATURES
    return features[output_cols]
