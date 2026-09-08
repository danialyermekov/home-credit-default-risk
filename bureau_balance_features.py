"""Bureau balance feature engineering module.

Extracts dynamic delinquency and deterioration features from the Home Credit
bureau_balance table, mapped back to applicants via bureau records.
Preserves the exact feature semantics and temporal aggregations validated
in the accepted F9 modeling stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The 9 accepted bureau_balance dynamic features present in ACCEPTED_FINAL_FEATURES
BBX_ACCEPTED_FEATURES: list[str] = [
    "BBX_RECENT_DELINQUENT_ACCOUNT_SHARE",
    "BBX_MEAN_RECENT6_DELINQ_SHARE",
    "BBX_MAX_RECENT6_DELINQ_SHARE",
    "BBX_MEAN_RECENT12_DELINQ_SHARE",
    "BBX_MAX_SEVERITY",
    "BBX_MAX_RECENT6_SEVERITY",
    "BBX_MIN_MONTHS_SINCE_DELINQUENCY",
    "BBX_MEAN_RECENT_WORSENING",
    "BBX_MAX_RECENT_WORSENING",
]

SEVERITY_MAP: dict[str, float] = {
    "C": 0.0,
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "X": np.nan,
}


def build_bureau_balance_features(
    bureau_balance: pd.DataFrame,
    bureau: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute accepted dynamic bureau_balance features aggregated by applicant (SK_ID_CURR).

    Parameters
    ----------
    bureau_balance : pd.DataFrame
        Raw or preloaded bureau_balance table.
    bureau : pd.DataFrame | None, optional
        Bureau table containing SK_ID_BUREAU and SK_ID_CURR. Required if
        SK_ID_CURR is not already present in bureau_balance.

    Returns
    -------
    pd.DataFrame
        Aggregated features containing SK_ID_CURR and the 9 accepted
        bureau_balance dynamic features.
    """
    if "SK_ID_CURR" in bureau_balance.columns:
        x = bureau_balance.copy()
    else:
        if bureau is None:
            raise ValueError(
                "bureau DataFrame containing ['SK_ID_BUREAU', 'SK_ID_CURR'] "
                "must be provided when bureau_balance lacks SK_ID_CURR."
            )
        x = bureau_balance.merge(
            bureau[["SK_ID_BUREAU", "SK_ID_CURR"]],
            on="SK_ID_BUREAU",
            how="inner",
            validate="many_to_one",
        )

    # Point-in-time check: all monthly observations must precede cutoff
    if not (x["MONTHS_BALANCE"] <= 0).all():
        x = x.loc[x["MONTHS_BALANCE"] <= 0].copy()

    x["BBX_SEVERITY"] = x["STATUS"].map(SEVERITY_MAP)
    x["BBX_DELINQUENT"] = (x["BBX_SEVERITY"] > 0).astype(float)

    # Recent 6-month aggregations per account
    recent6 = (
        x.loc[x["MONTHS_BALANCE"] >= -6]
        .groupby(["SK_ID_CURR", "SK_ID_BUREAU"])
        .agg(
            BBX_RECENT6_DELINQ_SHARE=("BBX_DELINQUENT", "mean"),
            BBX_RECENT6_MAX_SEVERITY=("BBX_SEVERITY", "max"),
        )
    )

    # Recent 12-month aggregations per account
    recent12 = (
        x.loc[x["MONTHS_BALANCE"] >= -12]
        .groupby(["SK_ID_CURR", "SK_ID_BUREAU"])
        .agg(
            BBX_RECENT12_DELINQ_SHARE=("BBX_DELINQUENT", "mean"),
        )
    )

    # Account-level summaries
    account = (
        x.groupby(["SK_ID_CURR", "SK_ID_BUREAU"])
        .agg(
            BBX_MONTHS_OBSERVED=("MONTHS_BALANCE", "nunique"),
            BBX_ALL_DELINQ_SHARE=("BBX_DELINQUENT", "mean"),
            BBX_MAX_SEVERITY=("BBX_SEVERITY", "max"),
        )
        .join(recent6)
        .join(recent12)
        .reset_index()
    )

    # Delinquency recency
    delinquent = x.loc[x["BBX_DELINQUENT"] == 1]
    last_delinquency = (
        delinquent
        .groupby(["SK_ID_CURR", "SK_ID_BUREAU"])["MONTHS_BALANCE"]
        .max()
        .rename("BBX_LAST_DELINQUENCY_MONTH")
        .reset_index()
    )

    account = account.merge(
        last_delinquency,
        how="left",
        on=["SK_ID_CURR", "SK_ID_BUREAU"],
    )

    account["BBX_MONTHS_SINCE_DELINQUENCY"] = -account["BBX_LAST_DELINQUENCY_MONTH"]
    account["BBX_RECENT_WORSENING"] = (
        account["BBX_RECENT6_DELINQ_SHARE"] - account["BBX_ALL_DELINQ_SHARE"]
    )
    account["BBX_RECENT_DELINQUENT_ACCOUNT"] = (
        account["BBX_RECENT6_DELINQ_SHARE"] > 0
    ).astype(float)

    # Client-level aggregation
    client = (
        account.groupby("SK_ID_CURR")
        .agg(
            BBX_RECENT_DELINQUENT_ACCOUNT_SHARE=("BBX_RECENT_DELINQUENT_ACCOUNT", "mean"),
            BBX_MEAN_RECENT6_DELINQ_SHARE=("BBX_RECENT6_DELINQ_SHARE", "mean"),
            BBX_MAX_RECENT6_DELINQ_SHARE=("BBX_RECENT6_DELINQ_SHARE", "max"),
            BBX_MEAN_RECENT12_DELINQ_SHARE=("BBX_RECENT12_DELINQ_SHARE", "mean"),
            BBX_MAX_SEVERITY=("BBX_MAX_SEVERITY", "max"),
            BBX_MAX_RECENT6_SEVERITY=("BBX_RECENT6_MAX_SEVERITY", "max"),
            BBX_MIN_MONTHS_SINCE_DELINQUENCY=("BBX_MONTHS_SINCE_DELINQUENCY", "min"),
            BBX_MEAN_RECENT_WORSENING=("BBX_RECENT_WORSENING", "mean"),
            BBX_MAX_RECENT_WORSENING=("BBX_RECENT_WORSENING", "max"),
        )
        .reset_index()
    )

    output_cols = ["SK_ID_CURR"] + BBX_ACCEPTED_FEATURES
    return client[output_cols]
