"""Bureau feature engineering module.

Extracts applicant-level bureau features from the Home Credit bureau table,
preserving the exact feature semantics, cutoff filtering, and aggregations
validated during model development.
"""

from __future__ import annotations

import pandas as pd

# The 16 accepted bureau features present in ACCEPTED_FINAL_FEATURES
BUREAU_ACCEPTED_FEATURES: list[str] = [
    "MAX_CREDIT_OVERDUE_AMT",
    "BUREAU_CREDIT_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CLOSED_COUNT",
    "BUREAU_SOLD_COUNT",
    "BUREAU_ACTIVE_SHARE",
    "BUREAU_DAYS_SINCE_LATEST_CREDIT",
    "BUREAU_HISTORY_AGE_DAYS",
    "BUREAU_CREDITS_LAST_180D",
    "BUREAU_CREDITS_LAST_365D",
    "BUREAU_CREDITS_LAST_730D",
    "BUREAU_TOTAL_CREDIT_SUM",
    "BUREAU_MAX_CREDIT_SUM",
    "BUREAU_TOTAL_CREDIT_LIMIT",
    "BUREAU_MEAN_CREDIT_SUM",
    "BUREAU_TOTAL_CREDIT_DEBT",
]

BUREAU_COUNT_FEATURES: list[str] = [
    "BUREAU_CREDIT_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CLOSED_COUNT",
    "BUREAU_SOLD_COUNT",
]


def build_bureau_features(
    bureau: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accepted bureau features aggregated by applicant (SK_ID_CURR).

    Parameters
    ----------
    bureau : pd.DataFrame
        Raw or preloaded bureau table.

    Returns
    -------
    pd.DataFrame
        Aggregated features containing SK_ID_CURR and the 16 accepted
        bureau features.
    """
    # Cutoff audit: records updated after application date (DAYS_CREDIT_UPDATE > 0)
    # are excluded to prevent post-application leakage.
    bureau_safe = bureau.loc[bureau["DAYS_CREDIT_UPDATE"] <= 0].copy()

    # Credit status flags
    bureau_safe["BUREAU_IS_ACTIVE"] = bureau_safe["CREDIT_ACTIVE"].eq("Active").astype("int8")
    bureau_safe["BUREAU_IS_CLOSED"] = bureau_safe["CREDIT_ACTIVE"].eq("Closed").astype("int8")
    bureau_safe["BUREAU_IS_SOLD"] = bureau_safe["CREDIT_ACTIVE"].eq("Sold").astype("int8")

    # Recency window flags
    bureau_safe["BUREAU_CREDIT_LAST_180D"] = bureau_safe["DAYS_CREDIT"].ge(-180).astype("int8")
    bureau_safe["BUREAU_CREDIT_LAST_365D"] = bureau_safe["DAYS_CREDIT"].ge(-365).astype("int8")
    bureau_safe["BUREAU_CREDIT_LAST_730D"] = bureau_safe["DAYS_CREDIT"].ge(-730).astype("int8")

    features = (
        bureau_safe.groupby("SK_ID_CURR")
        .agg(
            BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
            BUREAU_ACTIVE_COUNT=("BUREAU_IS_ACTIVE", "sum"),
            BUREAU_CLOSED_COUNT=("BUREAU_IS_CLOSED", "sum"),
            BUREAU_SOLD_COUNT=("BUREAU_IS_SOLD", "sum"),
            BUREAU_DAYS_SINCE_LATEST_CREDIT=("DAYS_CREDIT", lambda x: -x.max()),
            BUREAU_HISTORY_AGE_DAYS=("DAYS_CREDIT", lambda x: -x.min()),
            BUREAU_CREDITS_LAST_180D=("BUREAU_CREDIT_LAST_180D", "sum"),
            BUREAU_CREDITS_LAST_365D=("BUREAU_CREDIT_LAST_365D", "sum"),
            BUREAU_CREDITS_LAST_730D=("BUREAU_CREDIT_LAST_730D", "sum"),
            BUREAU_TOTAL_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
            BUREAU_MAX_CREDIT_SUM=("AMT_CREDIT_SUM", "max"),
            BUREAU_MEAN_CREDIT_SUM=("AMT_CREDIT_SUM", "mean"),
            BUREAU_TOTAL_CREDIT_LIMIT=("AMT_CREDIT_SUM_LIMIT", "sum"),
            BUREAU_TOTAL_CREDIT_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
            MAX_CREDIT_OVERDUE_AMT=("AMT_CREDIT_MAX_OVERDUE", "max"),
        )
        .reset_index()
    )

    # Active share calculation
    features["BUREAU_ACTIVE_SHARE"] = (
        features["BUREAU_ACTIVE_COUNT"] / features["BUREAU_CREDIT_COUNT"]
    )

    output_cols = ["SK_ID_CURR"] + BUREAU_ACCEPTED_FEATURES
    return features[output_cols]
