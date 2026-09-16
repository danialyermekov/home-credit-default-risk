"""Previous application feature engineering module.

Extracts financial and temporal history features from the Home Credit
previous_application table, preserving the exact feature semantics,
sentinel handling, and aggregations validated during model development.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The 34 accepted previous_application features present in ACCEPTED_FINAL_FEATURES
PREVIOUS_APPLICATION_ACCEPTED_FEATURES: tuple[str, ...] = (
    "TOTAL_AMT_PREV_APPLICATION",
    "MEAN_AMT_PREV_APPLICATION",
    "MAX_AMT_PREV_APPLICATION",
    "TOTAL_AMT_PREV_CREDIT",
    "MEAN_AMT_PREV_CREDIT",
    "MAX_AMT_PREV_CREDIT",
    "TOTAL_AMT_PREV_GOODS_PRICE",
    "MEAN_AMT_PREV_GOODS_PRICE",
    "MAX_AMT_PREV_GOODS_PRICE",
    "TOTAL_AMT_PREV_ANNUITY",
    "MEAN_AMT_PREV_ANNUITY",
    "MAX_AMT_PREV_ANNUITY",
    "TOTAL_AMT_PREV_DOWN_PAYMENT",
    "MEAN_AMT_PREV_DOWN_PAYMENT",
    "MAX_AMT_PREV_DOWN_PAYMENT",
    "TOTAL_CNT_PREV_PAYMENT",
    "MEAN_CNT_PREV_PAYMENT",
    "MAX_CNT_PREV_PAYMENT",
    "TOTAL_CREDIT_APPLICATION_DIFF",
    "MEAN_CREDIT_APPLICATION_DIFF",
    "MAX_CREDIT_APPLICATION_DIFF",
    "MIN_CREDIT_APPLICATION_DIFF",
    "MEAN_CREDIT_APPLICATION_RATIO",
    "MAX_CREDIT_APPLICATION_RATIO",
    "MIN_CREDIT_APPLICATION_RATIO",
    "PREV_DAYS_SINCE_LAST_DECISION",
    "PREV_HISTORY_AGE_DAYS",
    "PREV_MEAN_DAYS_SINCE_DECISION",
    "MEAN_PREV_PLANNED_DURATION_DAYS",
    "MAX_PREV_PLANNED_DURATION_DAYS",
    "PREV_FUTURE_PLANNED_END_COUNT",
    "PREV_FUTURE_PLANNED_END_SHARE",
    "MAX_PREV_PLANNED_DAYS_REMAINING",
    "PREV_DAYS_SINCE_LAST_TERMINATION",
)

TEMPORAL_SENTINEL_COLUMNS: tuple[str, ...] = (
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
)


def build_previous_application_features(
    previous_application: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accepted previous application
        features aggregated by applicant (SK_ID_CURR).

    Parameters
    ----------
    previous_application : pd.DataFrame
        Raw or preloaded previous_application table.

    Returns
    -------
    pd.DataFrame
        Aggregated features containing SK_ID_CURR and the 34 accepted
        previous_application features.
    """
    prev = previous_application.copy()

    # Credit-to-application relationships
    prev["CREDIT_APPLICATION_DIFF"] = prev["AMT_CREDIT"] - prev["AMT_APPLICATION"]
    prev["CREDIT_APPLICATION_RATIO"] = prev["AMT_CREDIT"] / prev[
        "AMT_APPLICATION"
    ].replace(0, np.nan)

    # Sentinel value audit: replace 365243 anomaly with NaN
    sentinel_cols = [c for c in TEMPORAL_SENTINEL_COLUMNS if c in prev.columns]
    prev[sentinel_cols] = prev[sentinel_cols].replace(365243, np.nan)

    # Planned contract duration and future termination
    prev["PREV_PLANNED_DURATION_DAYS"] = (
        prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_FIRST_DUE"]
    )
    prev["PREV_PLANNED_ENDS_IN_FUTURE"] = (
        prev["DAYS_LAST_DUE_1ST_VERSION"] > 0
    ).astype("int8")

    prev["PREV_PLANNED_DAYS_REMAINING"] = prev["DAYS_LAST_DUE_1ST_VERSION"].clip(
        lower=0
    )

    features = (
        prev.groupby("SK_ID_CURR")
        .agg(
            TOTAL_AMT_PREV_APPLICATION=("AMT_APPLICATION", "sum"),
            MEAN_AMT_PREV_APPLICATION=("AMT_APPLICATION", "mean"),
            MAX_AMT_PREV_APPLICATION=("AMT_APPLICATION", "max"),
            TOTAL_AMT_PREV_CREDIT=("AMT_CREDIT", "sum"),
            MEAN_AMT_PREV_CREDIT=("AMT_CREDIT", "mean"),
            MAX_AMT_PREV_CREDIT=("AMT_CREDIT", "max"),
            TOTAL_AMT_PREV_GOODS_PRICE=("AMT_GOODS_PRICE", "sum"),
            MEAN_AMT_PREV_GOODS_PRICE=("AMT_GOODS_PRICE", "mean"),
            MAX_AMT_PREV_GOODS_PRICE=("AMT_GOODS_PRICE", "max"),
            TOTAL_AMT_PREV_ANNUITY=("AMT_ANNUITY", "sum"),
            MEAN_AMT_PREV_ANNUITY=("AMT_ANNUITY", "mean"),
            MAX_AMT_PREV_ANNUITY=("AMT_ANNUITY", "max"),
            TOTAL_AMT_PREV_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "sum"),
            MEAN_AMT_PREV_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "mean"),
            MAX_AMT_PREV_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "max"),
            TOTAL_CNT_PREV_PAYMENT=("CNT_PAYMENT", "sum"),
            MEAN_CNT_PREV_PAYMENT=("CNT_PAYMENT", "mean"),
            MAX_CNT_PREV_PAYMENT=("CNT_PAYMENT", "max"),
            TOTAL_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "sum"),
            MEAN_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "mean"),
            MIN_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "min"),
            MAX_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "max"),
            MEAN_CREDIT_APPLICATION_RATIO=("CREDIT_APPLICATION_RATIO", "mean"),
            MIN_CREDIT_APPLICATION_RATIO=("CREDIT_APPLICATION_RATIO", "min"),
            MAX_CREDIT_APPLICATION_RATIO=("CREDIT_APPLICATION_RATIO", "max"),
            PREV_DAYS_SINCE_LAST_DECISION=("DAYS_DECISION", lambda x: -x.max()),
            PREV_HISTORY_AGE_DAYS=("DAYS_DECISION", lambda x: -x.min()),
            PREV_MEAN_DAYS_SINCE_DECISION=("DAYS_DECISION", lambda x: -x.mean()),
            MEAN_PREV_PLANNED_DURATION_DAYS=("PREV_PLANNED_DURATION_DAYS", "mean"),
            MAX_PREV_PLANNED_DURATION_DAYS=("PREV_PLANNED_DURATION_DAYS", "max"),
            PREV_FUTURE_PLANNED_END_COUNT=("PREV_PLANNED_ENDS_IN_FUTURE", "sum"),
            PREV_FUTURE_PLANNED_END_SHARE=("PREV_PLANNED_ENDS_IN_FUTURE", "mean"),
            MAX_PREV_PLANNED_DAYS_REMAINING=("PREV_PLANNED_DAYS_REMAINING", "max"),
            PREV_DAYS_SINCE_LAST_TERMINATION=("DAYS_TERMINATION", lambda x: -x.max()),
        )
        .reset_index()
    )

    output_cols = ["SK_ID_CURR"] + list(PREVIOUS_APPLICATION_ACCEPTED_FEATURES)
    return features[output_cols]
