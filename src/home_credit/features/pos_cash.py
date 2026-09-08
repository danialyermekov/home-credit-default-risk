"""POS and cash balance feature engineering module.

Extracts dynamic delinquency, contract progress, and deterioration features
from the Home Credit POS_CASH_balance table, preserving the exact contract-level
tracking and recent-window semantics validated in the accepted F9 modeling stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The 9 accepted POS_CASH_balance dynamic features present in ACCEPTED_FINAL_FEATURES
POS_ACCEPTED_FEATURES: list[str] = [
    "POSX_MEAN_LATEST_DPD",
    "POSX_MAX_LATEST_DPD",
    "POSX_BAD_LATEST_SHARE",
    "POSX_MEAN_RECENT6_DPD_SHARE",
    "POSX_MAX_RECENT6_DPD_SHARE",
    "POSX_MEAN_RECENT_WORSENING",
    "POSX_MAX_RECENT_WORSENING",
    "POSX_MEAN_PROGRESS",
    "POSX_MIN_PROGRESS",
]


def build_pos_cash_features(
    pos_cash_balance: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accepted dynamic POS features aggregated by applicant (SK_ID_CURR).

    Parameters
    ----------
    pos_cash_balance : pd.DataFrame
        Raw or preloaded POS_CASH_balance table.

    Returns
    -------
    pd.DataFrame
        Aggregated features containing SK_ID_CURR and the 9 accepted
        POS_CASH_balance dynamic features.
    """
    cols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]
    x = pos_cash_balance[cols].copy()

    # Point-in-time check: all monthly observations must precede cutoff
    if not (x["MONTHS_BALANCE"] <= 0).all():
        x = x.loc[x["MONTHS_BALANCE"] <= 0].copy()

    x["POSX_REMAINING_RATIO"] = (
        x["CNT_INSTALMENT_FUTURE"]
        / x["CNT_INSTALMENT"].replace(0, np.nan)
    )

    x["POSX_DPD_FLAG"] = (x["SK_DPD"] > 0).astype(float)
    x["POSX_DPD30_FLAG"] = (x["SK_DPD"] > 30).astype(float)

    # Recent 6-month contract behavior
    recent6 = (
        x.loc[x["MONTHS_BALANCE"] >= -6]
        .groupby(["SK_ID_CURR", "SK_ID_PREV"])
        .agg(
            POSX_RECENT6_DPD_MEAN=("SK_DPD", "mean"),
            POSX_RECENT6_DPD_MAX=("SK_DPD", "max"),
            POSX_RECENT6_DPD_SHARE=("POSX_DPD_FLAG", "mean"),
            POSX_RECENT6_DPD30_SHARE=("POSX_DPD30_FLAG", "mean"),
        )
    )

    # Full contract behavior
    contract = (
        x.groupby(["SK_ID_CURR", "SK_ID_PREV"])
        .agg(
            POSX_MONTHS_OBSERVED=("MONTHS_BALANCE", "nunique"),
            POSX_MEAN_DPD=("SK_DPD", "mean"),
            POSX_MAX_DPD=("SK_DPD", "max"),
            POSX_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
            POSX_DPD_SHARE=("POSX_DPD_FLAG", "mean"),
        )
        .join(recent6)
        .reset_index()
    )

    ordered = x.sort_values(["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"])

    oldest = (
        ordered
        .drop_duplicates(["SK_ID_CURR", "SK_ID_PREV"], keep="first")
        [["SK_ID_CURR", "SK_ID_PREV", "POSX_REMAINING_RATIO"]]
        .rename(columns={"POSX_REMAINING_RATIO": "POSX_OLDEST_REMAINING_RATIO"})
    )

    latest = (
        ordered
        .drop_duplicates(["SK_ID_CURR", "SK_ID_PREV"], keep="last")
        [
            [
                "SK_ID_CURR",
                "SK_ID_PREV",
                "SK_DPD",
                "SK_DPD_DEF",
                "POSX_REMAINING_RATIO",
            ]
        ]
        .rename(
            columns={
                "SK_DPD": "POSX_LATEST_DPD",
                "SK_DPD_DEF": "POSX_LATEST_DPD_DEF",
                "POSX_REMAINING_RATIO": "POSX_LATEST_REMAINING_RATIO",
            }
        )
    )

    contract = contract.merge(oldest).merge(latest)

    contract["POSX_PROGRESS"] = (
        contract["POSX_OLDEST_REMAINING_RATIO"]
        - contract["POSX_LATEST_REMAINING_RATIO"]
    )

    contract["POSX_RECENT_WORSENING"] = (
        contract["POSX_RECENT6_DPD_SHARE"]
        - contract["POSX_DPD_SHARE"]
    )

    contract["POSX_BAD_LATEST"] = (contract["POSX_LATEST_DPD"] > 0).astype(float)

    # Client-level aggregation
    client = (
        contract.groupby("SK_ID_CURR")
        .agg(
            POSX_MEAN_LATEST_DPD=("POSX_LATEST_DPD", "mean"),
            POSX_MAX_LATEST_DPD=("POSX_LATEST_DPD", "max"),
            POSX_BAD_LATEST_SHARE=("POSX_BAD_LATEST", "mean"),
            POSX_MEAN_RECENT6_DPD_SHARE=("POSX_RECENT6_DPD_SHARE", "mean"),
            POSX_MAX_RECENT6_DPD_SHARE=("POSX_RECENT6_DPD_SHARE", "max"),
            POSX_MEAN_RECENT_WORSENING=("POSX_RECENT_WORSENING", "mean"),
            POSX_MAX_RECENT_WORSENING=("POSX_RECENT_WORSENING", "max"),
            POSX_MEAN_PROGRESS=("POSX_PROGRESS", "mean"),
            POSX_MIN_PROGRESS=("POSX_PROGRESS", "min"),
        )
        .reset_index()
    )

    output_cols = ["SK_ID_CURR"] + POS_ACCEPTED_FEATURES
    return client[output_cols]
