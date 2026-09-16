"""Credit card balance feature engineering module.

Extracts activity, balance, and utilization features from the Home Credit
credit_card_balance table, preserving the exact feature semantics and
latest-month contract tracking validated during model development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The 12 accepted credit_card_balance features present in ACCEPTED_FINAL_FEATURES
CREDIT_CARD_ACCEPTED_FEATURES: list[str] = [
    "CC_MONTHS_OBSERVED",
    "CC_HISTORY_AGE_MONTHS",
    "CC_ACTIVE_BALANCE_MONTH_SHARE",
    "CC_DRAWING_MONTH_SHARE",
    "CC_MEAN_BALANCE",
    "CC_MAX_BALANCE",
    "CC_LATEST_BALANCE",
    "CC_MEAN_CREDIT_LIMIT",
    "CC_LATEST_CREDIT_LIMIT",
    "CC_MEAN_UTILIZATION",
    "CC_MAX_UTILIZATION",
    "CC_LATEST_UTILIZATION",
]


def build_credit_card_features(
    credit_card_balance: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accepted credit card features aggregated by applicant (SK_ID_CURR).

    Parameters
    ----------
    credit_card_balance : pd.DataFrame
        Raw or preloaded credit_card_balance table.

    Returns
    -------
    pd.DataFrame
        Aggregated features containing SK_ID_CURR and the 12 accepted
        credit_card_balance features.
    """
    cc = credit_card_balance.copy()

    # Activity flags and utilization
    cc["CC_HAS_BALANCE"] = cc["AMT_BALANCE"] != 0
    cc["CC_HAS_DRAWINGS"] = cc["AMT_DRAWINGS_CURRENT"] != 0
    cc["CC_UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(
        0, np.nan
    )

    # Historical aggregations across all observed months
    cc_hist = (
        cc.groupby("SK_ID_CURR")
        .agg(
            CC_MONTHS_OBSERVED=("MONTHS_BALANCE", "count"),
            CC_HISTORY_AGE_MONTHS=("MONTHS_BALANCE", lambda x: -x.min()),
            CC_ACTIVE_BALANCE_MONTH_SHARE=("CC_HAS_BALANCE", "mean"),
            CC_DRAWING_MONTH_SHARE=("CC_HAS_DRAWINGS", "mean"),
            CC_MEAN_BALANCE=("AMT_BALANCE", "mean"),
            CC_MAX_BALANCE=("AMT_BALANCE", "max"),
            CC_MEAN_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            CC_MEAN_UTILIZATION=("CC_UTILIZATION", "mean"),
            CC_MAX_UTILIZATION=("CC_UTILIZATION", "max"),
        )
        .reset_index()
    )

    # Latest observed month per card contract
    cc_latest = (
        cc.sort_values(["SK_ID_PREV", "MONTHS_BALANCE"]).groupby("SK_ID_PREV").tail(1)
    )

    cc_latest_agg = (
        cc_latest.groupby("SK_ID_CURR")
        .agg(
            CC_LATEST_BALANCE=("AMT_BALANCE", "mean"),
            CC_LATEST_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            CC_LATEST_UTILIZATION=("CC_UTILIZATION", "mean"),
        )
        .reset_index()
    )

    features = cc_hist.merge(cc_latest_agg, on="SK_ID_CURR", how="left")

    output_cols = ["SK_ID_CURR"] + CREDIT_CARD_ACCEPTED_FEATURES
    return features[output_cols]
