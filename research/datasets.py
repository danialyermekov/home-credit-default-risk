import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import pandas as pd
    import numpy as np

    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"

    APPLICATION_PATH = RAW_DATA_DIR / "application_train.csv"
    BUREAU_PATH = RAW_DATA_DIR / "bureau.csv"
    PREVIOUS_APPLICATION_PATH = RAW_DATA_DIR / "previous_application.csv"
    INSTALLMENTS_PAYMENTS_PATH = RAW_DATA_DIR / "installments_payments.csv"
    CREDIT_CARD_BALANCE_PATH = RAW_DATA_DIR / "credit_card_balance.csv"
    POS_CASH_BALANCE_PATH = RAW_DATA_DIR / "POS_CASH_balance.csv"
    BUREAU_BALANCE_PATH = RAW_DATA_DIR / "bureau_balance.csv"
    return (
        BUREAU_BALANCE_PATH,
        BUREAU_PATH,
        CREDIT_CARD_BALANCE_PATH,
        INSTALLMENTS_PAYMENTS_PATH,
        POS_CASH_BALANCE_PATH,
        PREVIOUS_APPLICATION_PATH,
        np,
        pd,
    )


@app.cell
def _(
    BUREAU_BALANCE_PATH,
    BUREAU_PATH,
    CREDIT_CARD_BALANCE_PATH,
    INSTALLMENTS_PAYMENTS_PATH,
    POS_CASH_BALANCE_PATH,
    PREVIOUS_APPLICATION_PATH,
    pd,
):
    modeling_df = pd.read_parquet("data/processed/modeling_df.parquet")
    bureau = pd.read_csv(BUREAU_PATH)
    previous_application = pd.read_csv(PREVIOUS_APPLICATION_PATH)
    credit_card_balance = pd.read_csv(CREDIT_CARD_BALANCE_PATH)
    installments_payments = pd.read_csv(INSTALLMENTS_PAYMENTS_PATH)
    pos_cash_balance = pd.read_csv(POS_CASH_BALANCE_PATH)
    bureau_balance = pd.read_csv(BUREAU_BALANCE_PATH)
    return (
        bureau,
        bureau_balance,
        credit_card_balance,
        installments_payments,
        pos_cash_balance,
        previous_application,
    )


@app.cell
def _():
    # tables = {
    #     "applications": applications,
    #     "bureau": bureau,
    #     "previous_application": previous_application,
    #     "installments_payments": installments_payments,
    #     "credit_card_balance": credit_card_balance,
    #     "POS_CASH_balance": pos_cash_balance,
    #     "bureau_balance": bureau_balance,
    # }

    # for table_name, df in tables.items():
    #     print(f"\n{'=' * 20} {table_name} {'=' * 20}")
    #     print("shape:", df.shape)

    #     id_columns = [
    #         col for col in df.columns
    #         if "SK_ID" in col.upper() or "ID_" in col.upper()
    #     ]

    #     time_columns = [
    #         col for col in df.columns
    #         if any(
    #             token in col.upper()
    #             for token in ("DAY", "DATE", "TIME", "MONTH")
    #         )
    #     ]

    #     print("ID columns:")
    #     print(id_columns)

    #     print("Potential time columns:")
    #     print(time_columns)
    return


@app.cell
def _():
    # history_tables = {
    #     "bureau": bureau,
    #     "previous_application": previous_application,
    #     "installments_payments": installments_payments,
    #     "credit_card_balance": credit_card_balance,
    #     "POS_CASH_balance": pos_cash_balance,
    # }

    # application_ids = set(applications["SK_ID_CURR"])

    # for name, table in history_tables.items():
    #     history_ids = set(table["SK_ID_CURR"])

    #     covered_ids = application_ids & history_ids

    #     print(
    #         name,
    #         f"{len(covered_ids):,}/{len(application_ids):,}",
    #         f"({len(covered_ids) / len(application_ids):.2%})",
    #     )
    return


@app.cell
def _():
    # age_years = -applications["DAYS_BIRTH"] / 365.25

    # print(age_years.describe())
    # print("Age < 18:", (age_years < 18).sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset structure and validation implications

    The project uses one application-level table and several historical tables:

    - `applications`: one row per current loan application (`SK_ID_CURR`).
    - `bureau`: previous credits reported by the credit bureau.
    - `previous_application`: previous Home Credit applications.
    - `installments_payments`: installment payment history.
    - `credit_card_balance`: monthly credit card history.
    - `POS_CASH_balance`: monthly POS/cash loan history.
    - `bureau_balance`: monthly history for individual bureau credits.

    ## Entity structure

    `SK_ID_CURR` is the prediction-row identifier and is unique in `applications`.

    No persistent customer identifier is available in `applications` that would allow multiple current applications to be linked to the same physical customer.

    Therefore:

    - repeated-customer overlap between folds cannot be directly detected;
    - customer-level `GroupKFold` / `StratifiedGroupKFold` cannot be constructed;
    - this limitation must be treated as part of the dataset contract rather than approximated with an artificial grouping key.

    Historical tables contain additional lower-level identifiers:

    - `SK_ID_BUREAU` identifies an individual bureau credit;
    - `SK_ID_PREV` identifies a previous Home Credit application.

    These identifiers describe historical records associated with a current `SK_ID_CURR`; they are not substitutes for a persistent customer ID across current applications.

    ## Time structure

    `applications` does not contain an absolute timestamp or calendar date for the current application.

    Columns such as:

    - `DAYS_BIRTH`
    - `DAYS_EMPLOYED`
    - `DAYS_REGISTRATION`
    - `DAYS_ID_PUBLISH`
    - `DAYS_LAST_PHONE_CHANGE`

    are relative attributes measured with respect to the current application and cannot be used to order current applications chronologically.

    Therefore a genuine application-level temporal split cannot be constructed from the available data.

    Historical tables do contain relative temporal information with respect to the current application:

    - `bureau`: `DAYS_CREDIT`, `DAYS_CREDIT_ENDDATE`, `DAYS_ENDDATE_FACT`, `DAYS_CREDIT_UPDATE`
    - `previous_application`: `DAYS_DECISION`, due/drawing/termination dates
    - `installments_payments`: `DAYS_INSTALMENT`, `DAYS_ENTRY_PAYMENT`
    - `credit_card_balance`: `MONTHS_BALANCE`
    - `POS_CASH_balance`: `MONTHS_BALANCE`
    - `bureau_balance`: `MONTHS_BALANCE`

    These columns will later be used for point-in-time leakage audits and recency/window-based feature engineering.

    However, the public dataset does not provide a complete `available_at` lineage for every historical field. Point-in-time correctness can therefore only be enforced within the temporal information exposed by the dataset.

    ## Historical source coverage

    Coverage is defined as the proportion of current labeled applications whose `SK_ID_CURR` appears at least once in a given historical table.

    | Historical source | Applications with history | Coverage |
    |---|---:|---:|
    | `bureau` | 263,491 / 307,511 | 85.69% |
    | `previous_application` | 291,057 / 307,511 | 94.65% |
    | `installments_payments` | 291,643 / 307,511 | 94.84% |
    | `credit_card_balance` | 86,905 / 307,511 | 28.26% |
    | `POS_CASH_balance` | 289,444 / 307,511 | 94.12% |

    The historical sources have substantially different coverage.

    `previous_application`, `installments_payments`, and `POS_CASH_balance` cover more than 94% of current applications, while `bureau` covers about 86%.

    `credit_card_balance` is fundamentally different: only 28.26% of applications have credit-card history.

    Therefore missing historical aggregates must not automatically be interpreted as ordinary missing values. In many cases they represent a meaningful state such as **no available history for this source**.

    This motivates explicit source-level cohort features such as:

    - `HAS_BUREAU_HISTORY`
    - `HAS_PREVIOUS_APPLICATION`
    - `HAS_INSTALLMENT_HISTORY`
    - `HAS_CREDIT_CARD_HISTORY`
    - `HAS_POS_CASH_HISTORY`

    Their usefulness as model features will still require an explicit feature-bundle experiment; their immediate value is to preserve the semantics of historical-source availability.

    ## Multi-table aggregation contract

    Historical tables are not independently divided into training and validation folds.

    The validation unit is the current application (`SK_ID_CURR`).

    The intended pipeline is:

    1. identify historical records belonging to a current `SK_ID_CURR`;
    2. enforce the available temporal/cutoff constraints;
    3. aggregate historical records to one row per `SK_ID_CURR`;
    4. left-join the resulting feature table to `applications`;
    5. preserve exactly one prediction row per current application.

    For nested history such as `bureau_balance`, aggregation is hierarchical:

    `bureau_balance`
    → aggregate by `SK_ID_BUREAU`
    → join to `bureau`
    → aggregate by `SK_ID_CURR`
    → join to `applications`.

    Every join must preserve the number and uniqueness of application-level prediction rows.

    ## Validation implication

    Given the available public data:

    - group-aware validation by physical customer is not possible;
    - application-level temporal validation is not possible;
    - the feasible offline assumption is that future applications are drawn from approximately the same population as the labeled application dataset.

    Therefore the main development validation scheme will use a fixed stratified split at the application level.

    This is a limitation of the available dataset, not evidence that the real production process is IID.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Key findings

    - The prediction unit is one current application (`SK_ID_CURR`).
    - No persistent customer ID is available, so customer-level group validation cannot be constructed.
    - No current-application timestamp is available, so genuine temporal validation is not possible.
    - Historical tables contain relative event timing that can be used for cutoff-aware feature engineering and leakage checks.
    - Historical source coverage is heterogeneous; absence of history is a meaningful cohort rather than generic missingness.
    - `credit_card_balance` is especially sparse at the application level (28.26% coverage).
    - Historical tables will be aggregated to one row per `SK_ID_CURR`; folds belong to application rows, not individual historical events.
    - The main offline validation assumption is therefore approximately IID future applications, using fixed stratified application-level folds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bureau
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grain
    """)
    return


@app.cell
def _(bureau):
    print("Shape:", bureau.shape)

    print(
        "Unique SK_ID_CURR:",
        bureau["SK_ID_CURR"].nunique(),
    )

    print(
        "Unique SK_ID_BUREAU:",
        bureau["SK_ID_BUREAU"].nunique(),
    )

    print(
        "Duplicated SK_ID_BUREAU:",
        bureau["SK_ID_BUREAU"].duplicated().sum(),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Structure
    """)
    return


@app.cell
def _(bureau):
    bureau.head()
    return


@app.cell
def _(bureau):
    bureau.dtypes
    return


@app.cell
def _(bureau):
    bureau.columns.tolist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loan count per application
    """)
    return


@app.cell
def _(bureau):
    bureau_records_per_application = (
        bureau
        .groupby("SK_ID_CURR")
        .size()
    )

    bureau_records_per_application.describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    )
    return


@app.cell
def _():
    # bureau_records_per_application.sort_values(
    #     ascending=False
    # ).head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Missingness
    """)
    return


@app.cell
def _(bureau):
    bureau_missing = (
        bureau
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .to_frame()
    )

    bureau_missing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical
    """)
    return


@app.cell
def _(bureau):
    bureau_categorical = bureau.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    bureau_categorical
    return (bureau_categorical,)


@app.cell
def _(bureau, bureau_categorical):
    for col in bureau_categorical:
        print(f"\n=== {col} ===")

        print(
            bureau[col]
            .value_counts(dropna=False)
            .head(20)
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time-related
    """)
    return


@app.cell
def _(bureau):
    TIME_COLUMNS = [
        "DAYS_CREDIT",
        "DAYS_CREDIT_ENDDATE",
        "DAYS_ENDDATE_FACT",
        "DAYS_CREDIT_UPDATE",
    ]

    bureau[TIME_COLUMNS].describe()
    return (TIME_COLUMNS,)


@app.cell
def _(TIME_COLUMNS, bureau):
    for time_col in TIME_COLUMNS:
        print(
            time_col,
            "positive:",
            (bureau[time_col] > 0).sum(),
            "zero:",
            (bureau[time_col] == 0).sum(),
            "negative:",
            (bureau[time_col] < 0).sum(),
            "missing:",
            bureau[time_col].isna().sum(),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `DAYS_CREDIT_UPDATE` > 0
    """)
    return


@app.cell
def _(bureau):
    future_credit_update = (
        bureau.loc[
            bureau["DAYS_CREDIT_UPDATE"] > 0,
            [
                "SK_ID_CURR",
                "SK_ID_BUREAU",
                "CREDIT_ACTIVE",
                "CREDIT_TYPE",
                "DAYS_CREDIT",
                "DAYS_CREDIT_ENDDATE",
                "DAYS_ENDDATE_FACT",
                "DAYS_CREDIT_UPDATE",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE",
            ],
        ]
        .sort_values("DAYS_CREDIT_UPDATE", ascending=False)
    )

    future_credit_update
    return (future_credit_update,)


@app.cell
def _(future_credit_update):
    future_credit_update["DAYS_CREDIT_UPDATE"].describe()
    return


@app.cell
def _(future_credit_update):
    future_credit_update["CREDIT_ACTIVE"].value_counts(dropna=False)
    return


@app.cell
def _(future_credit_update):
    future_credit_update["CREDIT_TYPE"].value_counts(dropna=False)
    return


@app.cell
def _(bureau):
    bureau_safe = (
        bureau
        .loc[bureau["DAYS_CREDIT_UPDATE"] <= 0]
        .copy()
    )

    print("Before:", len(bureau))
    print("After:", len(bureau_safe))
    print("Removed:", len(bureau) - len(bureau_safe))
    return (bureau_safe,)


@app.cell
def _(bureau_safe):
    assert (bureau_safe["DAYS_CREDIT_UPDATE"] > 0).sum() == 0
    assert (bureau_safe["DAYS_CREDIT"] > 0).sum() == 0
    assert (bureau_safe["DAYS_ENDDATE_FACT"] > 0).sum() == 0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bureau temporal / cutoff audit

    The temporal columns were checked relative to the current application cutoff.

    - `DAYS_CREDIT`: no positive values were found. All bureau credits were opened no later than the current application date.
    - `DAYS_ENDDATE_FACT`: no positive values were found. Recorded actual credit closures therefore do not occur after the application cutoff.
    - `DAYS_CREDIT_ENDDATE`: positive values are allowed because this represents a scheduled credit end date that may already be known at prediction time.
    - `DAYS_CREDIT_UPDATE`: 17 records had positive values, ranging from 10 to 372 days after the current application.

    All 17 post-cutoff update records corresponded to active credits (16 car loans and 1 mortgage).

    Because a bureau record updated after the application cutoff may contain other fields reflecting post-cutoff state, removing only `DAYS_CREDIT_UPDATE` would not be sufficient to guarantee point-in-time consistency.

    The 17 affected bureau records were therefore conservatively excluded from feature construction.

    This removes approximately 0.001% of bureau records and has negligible impact on dataset coverage while eliminating an identifiable source of potential temporal leakage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Money-related
    """)
    return


@app.cell
def _(bureau):
    AMOUNT_COLUMNS = [
        col
        for col in bureau.columns
        if col.startswith("AMT_")
    ]

    bureau[AMOUNT_COLUMNS].describe().T
    return (AMOUNT_COLUMNS,)


@app.cell
def _(AMOUNT_COLUMNS, bureau):
    bureau[AMOUNT_COLUMNS].isna().mean().sort_values(
        ascending=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Active and closed loans per client
    """)
    return


@app.cell
def _(bureau_safe):
    bureau_status_counts = (
        bureau_safe
        .groupby(["SK_ID_CURR", "CREDIT_ACTIVE"])
        .size()
        .unstack(fill_value=0)
    )

    bureau_status_counts.head()
    return (bureau_status_counts,)


@app.cell
def _(bureau_status_counts):
    status_summary = bureau_status_counts.describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    ).T

    status_summary
    return


@app.cell
def _(bureau_status_counts):
    for status in bureau_status_counts.columns:
        n_clients = (bureau_status_counts[status] > 0).sum()
        share = n_clients / len(bureau_status_counts)

        print(
            f"{status}: "
            f"{n_clients:,} clients "
            f"({share:.2%})"
        )
    return


@app.cell
def _(bureau_status_counts):
    bureau_status_profile = bureau_status_counts.copy()

    bureau_status_profile["TOTAL"] = (
        bureau_status_profile.sum(axis=1)
    )

    bureau_status_profile["ACTIVE_SHARE"] = (
        bureau_status_profile.get("Active", 0)
        / bureau_status_profile["TOTAL"]
    )

    bureau_status_profile[
        ["TOTAL", "Active", "Closed", "ACTIVE_SHARE"]
    ].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B1 - Bureau basic credit history
    """)
    return


@app.cell
def _():
    BUREAU_BASIC_FEATURES = [
        "BUREAU_CREDIT_COUNT",
        "BUREAU_ACTIVE_COUNT",
        "BUREAU_CLOSED_COUNT",
        "BUREAU_SOLD_COUNT",
        "BUREAU_ACTIVE_SHARE",
    ]
    return


@app.cell
def _():
    # bureau_basic_source = bureau_safe.assign(
    #     BUREAU_IS_ACTIVE=bureau_safe["CREDIT_ACTIVE"].eq("Active").astype("int8"),
    #     BUREAU_IS_CLOSED=bureau_safe["CREDIT_ACTIVE"].eq("Closed").astype("int8"),
    #     BUREAU_IS_SOLD=bureau_safe["CREDIT_ACTIVE"].eq("Sold").astype("int8"),
    # )

    # bureau_basic = (
    #     bureau_basic_source
    #     .groupby("SK_ID_CURR")
    #     .agg(
    #         BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
    #         BUREAU_ACTIVE_COUNT=("BUREAU_IS_ACTIVE", "sum"),
    #         BUREAU_CLOSED_COUNT=("BUREAU_IS_CLOSED", "sum"),
    #         BUREAU_SOLD_COUNT=("BUREAU_IS_SOLD", "sum"),
    #     )
    #     .reset_index()
    # )

    # bureau_basic["BUREAU_ACTIVE_SHARE"] = (
    #     bureau_basic["BUREAU_ACTIVE_COUNT"]
    #     / bureau_basic["BUREAU_CREDIT_COUNT"]
    # )
    return


@app.cell
def _():
    # print(bureau_basic.shape)

    # print(
    #     "Unique SK_ID_CURR:",
    #     bureau_basic["SK_ID_CURR"].nunique(),
    # )

    # print(
    #     "Duplicated SK_ID_CURR:",
    #     bureau_basic["SK_ID_CURR"].duplicated().sum(),
    # )

    # bureau_basic.describe().T
    return


@app.cell
def _():
    # modeling_bureau = modeling_df.merge(
    #     bureau_basic,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )

    # BUREAU_COUNT_FEATURES = [
    #     "BUREAU_CREDIT_COUNT",
    #     "BUREAU_ACTIVE_COUNT",
    #     "BUREAU_CLOSED_COUNT",
    #     "BUREAU_SOLD_COUNT",
    # ]

    # modeling_bureau[BUREAU_COUNT_FEATURES] = (
    #     modeling_bureau[BUREAU_COUNT_FEATURES]
    #     .fillna(0)
    # )
    return


@app.cell
def _():
    # modeling_bureau.to_parquet("data/processed/modeling_bureau_b1.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B2 - Bureau recency / credit activity
    """)
    return


@app.cell
def _():
    # bureau_recency_source = bureau_safe.assign(
    #     BUREAU_CREDIT_LAST_180D=bureau_safe["DAYS_CREDIT"].ge(-180).astype("int8"),
    #     BUREAU_CREDIT_LAST_365D=bureau_safe["DAYS_CREDIT"].ge(-365).astype("int8"),
    #     BUREAU_CREDIT_LAST_730D=bureau_safe["DAYS_CREDIT"].ge(-730).astype("int8"),
    # )

    # bureau_recency = (
    #     bureau_recency_source
    #     .groupby("SK_ID_CURR")
    #     .agg(
    #         BUREAU_DAYS_SINCE_LATEST_CREDIT=(
    #             "DAYS_CREDIT",
    #             lambda x: -x.max(),
    #         ),
    #         BUREAU_HISTORY_AGE_DAYS=(
    #             "DAYS_CREDIT",
    #             lambda x: -x.min(),
    #         ),
    #         BUREAU_CREDITS_LAST_180D=(
    #             "BUREAU_CREDIT_LAST_180D",
    #             "sum",
    #         ),
    #         BUREAU_CREDITS_LAST_365D=(
    #             "BUREAU_CREDIT_LAST_365D",
    #             "sum",
    #         ),
    #         BUREAU_CREDITS_LAST_730D=(
    #             "BUREAU_CREDIT_LAST_730D",
    #             "sum",
    #         ),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    BUREAU_RECENCY_FEATURES = [
        "BUREAU_DAYS_SINCE_LATEST_CREDIT",
        "BUREAU_HISTORY_AGE_DAYS",
        "BUREAU_CREDITS_LAST_180D",
        "BUREAU_CREDITS_LAST_365D",
        "BUREAU_CREDITS_LAST_730D",
    ]
    return


@app.cell
def _():
    # bureau_recency[
    #     BUREAU_RECENCY_FEATURES
    # ].describe().T
    return


@app.cell
def _():
    # for col_rec in [
    #     "BUREAU_CREDITS_LAST_180D",
    #     "BUREAU_CREDITS_LAST_365D",
    #     "BUREAU_CREDITS_LAST_730D",
    # ]:
    #     print(
    #         col_rec,
    #         "zero share:",
    #         (bureau_recency[col_rec] == 0).mean(),
    #         "mean:",
    #         bureau_recency[col_rec].mean(),
    #         "max:",
    #         bureau_recency[col_rec].max(),
    #     )
    return


@app.cell
def _():
    # modeling_bureau_b2 = modeling_bureau.merge(
    #     bureau_recency,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_bureau_b2.to_parquet("data/processed/modeling_bureau_b2.parquet", index=False)
    return


@app.cell
def _():
    # modeling_bureau_b2[BUREAU_RECENCY_FEATURES]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B3 - Bureau loan amount
    """)
    return


@app.cell
def _(bureau_safe):
    bureau_safe[
        [
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT",
            "AMT_CREDIT_SUM_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE",
            "AMT_ANNUITY"
        ]
    ].describe().T
    return


@app.cell
def _(bureau_safe):
    bureau_safe[
        [
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT",
            "AMT_CREDIT_SUM_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE",
            "AMT_ANNUITY"
        ]
    ].isna().mean().sort_values(ascending=False)
    return


@app.cell
def _(bureau_safe):
    for col_credit in [
        "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_LIMIT",
        "AMT_CREDIT_SUM_OVERDUE",
        "AMT_CREDIT_MAX_OVERDUE",
        "AMT_ANNUITY"

    ]:
        print(
            f"\n{col_credit}")
        print("negative:", (bureau_safe[col_credit] < 0).sum())
        print("zero:", (bureau_safe[col_credit] == 0).sum())
    return


@app.cell
def _(bureau_safe):
    negative_debt = bureau_safe.loc[
        bureau_safe["AMT_CREDIT_SUM_DEBT"] < 0,
        [
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "CREDIT_ACTIVE",
            "CREDIT_TYPE",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT",
            "AMT_CREDIT_SUM_OVERDUE",
        ],
    ].sort_values(
        "AMT_CREDIT_SUM_DEBT"
    )

    negative_debt.head(10)
    return (negative_debt,)


@app.cell
def _(negative_debt):
    negative_debt["CREDIT_ACTIVE"].value_counts(dropna=False)
    return


@app.cell
def _(negative_debt):
    negative_debt["AMT_CREDIT_SUM_DEBT"].describe()
    return


@app.cell
def _(bureau_safe):
    debt_limit = bureau_safe[
        [
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT",
            "CREDIT_TYPE",
            "AMT_CREDIT_SUM"
        ]
    ].dropna(
        subset=[
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT",
            "AMT_CREDIT_SUM"
        ]
    )

    negative_debt_mask = debt_limit["AMT_CREDIT_SUM_DEBT"] < 0

    (
        debt_limit.loc[negative_debt_mask]
        .assign(
            DEBT_PLUS_LIMIT=lambda x:
                x["AMT_CREDIT_SUM_DEBT"]
                + x["AMT_CREDIT_SUM_LIMIT"]
        )
        [["DEBT_PLUS_LIMIT", "AMT_CREDIT_SUM"]]
        .describe()
    )
    return debt_limit, negative_debt_mask


@app.cell
def _(debt_limit, negative_debt_mask):
    debt_limit.loc[
        negative_debt_mask,
        "CREDIT_TYPE"
    ].value_counts()
    return


@app.cell
def _():
    # bureau_credit = (
    #     bureau_safe.groupby("SK_ID_CURR").
    #     agg(
    #         BUREAU_TOTAL_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
    #         BUREAU_MAX_CREDIT_SUM=("AMT_CREDIT_SUM", "max"),
    #         BUREAU_MEAN_CREDIT_SUM=("AMT_CREDIT_SUM", "mean"),
    #         BUREAU_TOTAL_CREDIT_LIMIT=("AMT_CREDIT_SUM_LIMIT", "sum"),
    #         BUREAU_TOTAL_CREDIT_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
    #         BUREAU_TOTAL_CREDIT_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    #     ).reset_index()
    # )
    return


@app.cell
def _():
    BUREAU_CREDIT = [
        "BUREAU_TOTAL_CREDIT_SUM",
        "BUREAU_MAX_CREDIT_SUM",
        "BUREAU_TOTAL_CREDIT_LIMIT",
        "BUREAU_MEAN_CREDIT_SUM",
        "BUREAU_TOTAL_CREDIT_DEBT",
        "BUREAU_TOTAL_CREDIT_OVERDUE"
    ]
    return


@app.cell
def _():
    # bureau_credit[BUREAU_CREDIT].describe().T
    return


@app.cell
def _():
    # bureau_credit['SK_ID_CURR'].duplicated().sum()
    return


@app.cell
def _():
    # modeling_bureau_b3 = modeling_bureau_b2.merge(
    #     bureau_credit,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_bureau_b3.to_parquet("data/processed/modeling_bureau_b3.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B4 - Delinquency / credit stress
    """)
    return


@app.cell
def _():
    STRESS_FEATURES = [
        "TOTAL_CREDIT_OVERDUE",
        "HAS_OVERDUE_CREDIT",
        "MAX_CREDIT_DAY_OVERDUE",
        "MAX_CREDIT_OVERDUE_AMT",
        "TOTAL_ACTIVE_CREDIT_OVERDUE",
        "OVERDUE_CREDIT_SHARE"
    ]
    return


@app.cell
def _():
    # bureau_stress =(
    #     bureau_safe.groupby("SK_ID_CURR").
    #     agg(
    #         TOTAL_CREDIT_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    #         HAS_OVERDUE_CREDIT=("AMT_CREDIT_SUM_OVERDUE", lambda x: (x > 0).any().astype("int8")),
    #         MAX_CREDIT_DAY_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
    #         MAX_CREDIT_OVERDUE_AMT=("AMT_CREDIT_MAX_OVERDUE", "max"),
    #         TOTAL_ACTIVE_CREDIT_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", lambda x: x[bureau_safe.loc[x.index, "CREDIT_ACTIVE"] == "Active"].sum()),
    #         OVERDUE_CREDIT_SHARE=("AMT_CREDIT_SUM_OVERDUE", lambda x: x[x > 0].sum() / x.sum() if x.sum() > 0 else 0),
    #     ).reset_index()
    # )
    return


@app.cell
def _():
    # bureau_stress.describe()
    return


@app.cell
def _():
    # bureau_stress['SK_ID_CURR'].duplicated().sum()
    return


@app.cell
def _():
    # modeling_bureau_b4 = modeling_bureau_b3.merge(
    #     bureau_stress,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_bureau_b4.to_parquet("data/processed/modeling_bureau_b4.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Previous applications
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Structure
    """)
    return


@app.cell
def _(previous_application):
    previous_application.head()
    return


@app.cell
def _(previous_application):
    print(f"Shape: {previous_application.shape}")
    print(f"Unique application IDs: {previous_application['SK_ID_CURR'].nunique()}")
    print(f"Unique previous application IDs: {previous_application['SK_ID_PREV'].nunique()}")
    print(f"Non-unique previous application IDs: {previous_application['SK_ID_PREV'].duplicated().sum()}")
    return


@app.cell
def _(previous_application):
    previous_application.info()
    return


@app.cell
def _(previous_application):
    numeric_columns = previous_application.select_dtypes(
        include=["number"]
    ).columns.tolist()

    cat_columns = previous_application.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    return cat_columns, numeric_columns


@app.cell
def _(numeric_columns):
    numeric_columns
    return


@app.cell
def _(cat_columns):
    cat_columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Previous application per SK_ID_CURR
    """)
    return


@app.cell
def _(previous_application):
    previous_application_per_id = (
        previous_application
        .groupby("SK_ID_CURR")
        .size()
        .rename("PREVIOUS_APPLICATION_COUNT")
        .to_frame()
    )
    previous_application_per_id.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Missingness
    """)
    return


@app.cell
def _(previous_application):
    missing_applications = (previous_application.isna().sum() / previous_application.shape[0]) * 100
    missing_applications.sort_values(ascending=False)
    return


@app.cell
def _(previous_application):
    previous_application[["RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED"]].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time-related
    """)
    return


@app.cell
def _(numeric_columns, previous_application):
    day_cols = []

    for day_col in numeric_columns:
        if "DAY" in day_col.upper():
            day_cols.append(day_col)
            print(
                day_col,
                "positive:",
                (previous_application[day_col] > 0).sum(),
                "zero:",
                (previous_application[day_col] == 0).sum(),
                "negative:",
                (previous_application[day_col] < 0).sum(),
                "missing:",
                previous_application[day_col].isna().sum(),
                end="\n\n"
            )
    return (day_cols,)


@app.cell
def _(day_cols, previous_application):
    previous_application[day_cols].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### P1 - Application history
    """)
    return


@app.cell
def _():
    APPLICATION_HISTORY = [
        "PREV_APP_COUNT",
        "PREV_APP_APPROVED_COUNT",
        "PREV_APP_REFUSED_COUNT",
        "PREV_APP_CANCELED_COUNT",
        "PREV_APP_UNUSED_COUNT",
        "PREV_APP_APPROVAL_RATE",
        "PREV_APP_REFUSAL_RATE",
        "PREV_APP_DAYS_SINCE_LAST"
    ]
    return


@app.cell
def _():
    # application_history = (
    #     previous_application.groupby("SK_ID_CURR").
    #     agg(
    #         PREV_APP_COUNT=("SK_ID_PREV", "count"),
    #         PREV_APP_APPROVED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
    #         PREV_APP_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
    #         PREV_APP_CANCELED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Canceled").sum()),
    #         PREV_APP_UNUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Unused offer").sum()),
    #         PREV_APP_APPROVAL_RATE=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum() / len(x)),
    #         PREV_APP_REFUSAL_RATE=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum() / len(x)),
    #         PREV_APP_DAYS_SINCE_LAST=("DAYS_DECISION", lambda x: -x.min())
    #     ).reset_index()
    # )
    return


@app.cell
def _():
    # application_history.describe()
    return


@app.cell
def _():
    # modeling_bureau_4 = pd.read_parquet("data/processed/modeling_bureau_b4.parquet")
    return


@app.cell
def _():
    # application_history_1 = modeling_bureau_4.merge(
    #     application_history,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # application_history_1.to_parquet("data/processed/modeling_application_history_1.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P2 - Financial history
    """)
    return


@app.cell
def _():
    financial_history_features = [
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_GOODS_PRICE",
        "AMT_ANNUITY",
        "AMT_DOWN_PAYMENT",
        "CNT_PAYMENT"
    ]
    return (financial_history_features,)


@app.cell
def _(financial_history_features, previous_application):
    previous_application[financial_history_features].describe()
    return


@app.cell
def _(pd, previous_application):
    previous_application["CREDIT_APPLICATION_DIFF"] = (
        previous_application["AMT_CREDIT"]
        - previous_application["AMT_APPLICATION"]
    )

    previous_application["CREDIT_APPLICATION_RATIO"] = (
        previous_application["AMT_CREDIT"]
        / previous_application["AMT_APPLICATION"].replace(0, pd.NA)
    )
    return


@app.cell
def _():
    # financial_history = (
    #     previous_application.groupby("SK_ID_CURR")
    #     .agg(
    #         TOTAL_AMT_PREV_APPLICATION=("AMT_APPLICATION", "sum"),
    #         MEAN_AMT_PREV_APPLICATION=("AMT_APPLICATION", "mean"),
    #         MAX_AMT_PREV_APPLICATION=("AMT_APPLICATION", "max"),

    #         TOTAL_AMT_PREV_CREDIT=("AMT_CREDIT", "sum"),
    #         MEAN_AMT_PREV_CREDIT=("AMT_CREDIT", "mean"),
    #         MAX_AMT_PREV_CREDIT=("AMT_CREDIT", "max"),

    #         TOTAL_AMT_PREV_GOODS_PRICE=("AMT_GOODS_PRICE", "sum"),
    #         MEAN_AMT_PREV_GOODS_PRICE=("AMT_GOODS_PRICE", "mean"),
    #         MAX_AMT_PREV_GOODS_PRICE=("AMT_GOODS_PRICE", "max"),

    #         TOTAL_AMT_PREV_ANNUITY=("AMT_ANNUITY", "sum"),
    #         MEAN_AMT_PREV_ANNUITY=("AMT_ANNUITY", "mean"),
    #         MAX_AMT_PREV_ANNUITY=("AMT_ANNUITY", "max"),

    #         TOTAL_AMT_PREV_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "sum"),
    #         MEAN_AMT_PREV_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "mean"),
    #         MAX_AMT_PREV_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "max"),

    #         TOTAL_CNT_PREV_PAYMENT=("CNT_PAYMENT", "sum"),
    #         MEAN_CNT_PREV_PAYMENT=("CNT_PAYMENT", "mean"),
    #         MAX_CNT_PREV_PAYMENT=("CNT_PAYMENT", "max"),

    #         TOTAL_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "sum"),
    #         MEAN_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "mean"),
    #         MIN_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "min"),
    #         MAX_CREDIT_APPLICATION_DIFF=("CREDIT_APPLICATION_DIFF", "max"),

    #         MEAN_CREDIT_APPLICATION_RATIO=("CREDIT_APPLICATION_RATIO", "mean"),
    #         MIN_CREDIT_APPLICATION_RATIO=("CREDIT_APPLICATION_RATIO", "min"),
    #         MAX_CREDIT_APPLICATION_RATIO=("CREDIT_APPLICATION_RATIO", "max"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # application_history_2 = application_history_1.merge(
    #     financial_history,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # application_history_2.to_parquet("data/processed/modeling_application_history_2.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P3 - Categorical
    """)
    return


@app.cell
def _(cat_columns, previous_application):
    previous_application[cat_columns].describe()
    return


@app.cell
def _():
    # categorical = (
    #     previous_application.groupby("SK_ID_CURR")
    #     .agg(
    #         MODE_NAME_CONTRACT_TYPE=(
    #             "NAME_CONTRACT_TYPE",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #         SHARE_FLAG_LAST_APPL_PER_CONTRACT=(
    #             "FLAG_LAST_APPL_PER_CONTRACT",
    #             lambda x: (x == "Y").mean(),
    #         ),
    #         MODE_NAME_PAYMENT_TYPE=(
    #             "NAME_PAYMENT_TYPE",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #         MODE_CODE_REJECT_REASON=(
    #             "CODE_REJECT_REASON",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #         MODE_NAME_TYPE_SUITE=(
    #             "NAME_TYPE_SUITE",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #         MODE_NAME_CLIENT_TYPE=(
    #             "NAME_CLIENT_TYPE",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #         MODE_NAME_CASH_LOAN_PURPOSE=(
    #             "NAME_CASH_LOAN_PURPOSE",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #         MODE_NAME_PORTFOLIO=(
    #             "NAME_PORTFOLIO",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #         MODE_CHANNEL_TYPE=(
    #             "CHANNEL_TYPE",
    #             lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    #         ),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # application_history_p2 = pd.read_parquet("data\processed\modeling_application_history_2.parquet")
    return


@app.cell
def _():
    # application_history_3 = application_history_p2.merge(
    #     categorical,
    #     on='SK_ID_CURR',
    #     how='left',
    #     validate='one_to_one'
    # )

    # categorical_columns = [
    #     "MODE_NAME_CONTRACT_TYPE",
    #     "MODE_NAME_PAYMENT_TYPE",
    #     "MODE_CODE_REJECT_REASON",
    #     "MODE_NAME_TYPE_SUITE",
    #     "MODE_NAME_CLIENT_TYPE",
    #     "MODE_NAME_CASH_LOAN_PURPOSE",
    #     "MODE_NAME_PORTFOLIO",
    #     "MODE_CHANNEL_TYPE",
    # ]

    # for cat_col in categorical_columns:
    #     application_history_3[cat_col] = (
    #         application_history_3[cat_col]
    #         .fillna("__MISSING__")
    #         .astype("category")
    #     )
    return


@app.cell
def _():
    # application_history_3.to_parquet("data/processed/modeling_application_history_3.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P4 - Temporal history
    """)
    return


@app.cell
def _(np, previous_application):
    TEMPORAL_SENTINEL_COLUMNS = [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]

    previous_application[TEMPORAL_SENTINEL_COLUMNS] = (
        previous_application[TEMPORAL_SENTINEL_COLUMNS]
        .replace(365243, np.nan)
    )
    return (TEMPORAL_SENTINEL_COLUMNS,)


@app.cell
def _(TEMPORAL_SENTINEL_COLUMNS, previous_application):
    previous_application[
        ["DAYS_DECISION"] + TEMPORAL_SENTINEL_COLUMNS
    ].agg(["count", "min", "max"])
    return


@app.cell
def _(TEMPORAL_SENTINEL_COLUMNS, previous_application):
    (previous_application[
        ["DAYS_DECISION"] + TEMPORAL_SENTINEL_COLUMNS
    ] > 0).sum()
    return


@app.cell
def _():
    # previous_application["PREV_PLANNED_DURATION_DAYS"] = (
    #     previous_application["DAYS_LAST_DUE_1ST_VERSION"]
    #     - previous_application["DAYS_FIRST_DUE"]
    # )

    # previous_application["PREV_PLANNED_ENDS_IN_FUTURE"] = (
    #     previous_application["DAYS_LAST_DUE_1ST_VERSION"] > 0
    # ).astype("int8")

    # previous_application["PREV_PLANNED_DAYS_REMAINING"] = (
    #     previous_application["DAYS_LAST_DUE_1ST_VERSION"]
    #     .clip(lower=0)
    # )
    return


@app.cell
def _():
    # temporal_history = (
    #     previous_application.groupby("SK_ID_CURR")
    #     .agg(
    #         PREV_DAYS_SINCE_LAST_DECISION=(
    #             "DAYS_DECISION",
    #             lambda x: -x.max(),
    #         ),
    #         PREV_HISTORY_AGE_DAYS=(
    #             "DAYS_DECISION",
    #             lambda x: -x.min(),
    #         ),
    #         PREV_MEAN_DAYS_SINCE_DECISION=(
    #             "DAYS_DECISION",
    #             lambda x: -x.mean(),
    #         ),

    #         MEAN_PREV_PLANNED_DURATION_DAYS=(
    #             "PREV_PLANNED_DURATION_DAYS",
    #             "mean",
    #         ),
    #         MAX_PREV_PLANNED_DURATION_DAYS=(
    #             "PREV_PLANNED_DURATION_DAYS",
    #             "max",
    #         ),

    #         PREV_FUTURE_PLANNED_END_COUNT=(
    #             "PREV_PLANNED_ENDS_IN_FUTURE",
    #             "sum",
    #         ),
    #         PREV_FUTURE_PLANNED_END_SHARE=(
    #             "PREV_PLANNED_ENDS_IN_FUTURE",
    #             "mean",
    #         ),
    #         MAX_PREV_PLANNED_DAYS_REMAINING=(
    #             "PREV_PLANNED_DAYS_REMAINING",
    #             "max",
    #         ),

    #         PREV_DAYS_SINCE_LAST_TERMINATION=(
    #             "DAYS_TERMINATION",
    #             lambda x: -x.max(),
    #         ),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # modeling_application_history_3 = pd.read_parquet("data/processed/modeling_application_history_3.parquet")
    return


@app.cell
def _():
    # applicton_history_p4 = modeling_application_history_3.merge(
    #     temporal_history,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # applicton_history_p4['SK_ID_CURR'].duplicated().sum()
    return


@app.cell
def _():
    # applicton_history_p4.to_parquet("data/processed/modeling_application_history_4.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Credit card balance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grain
    """)
    return


@app.cell
def _(credit_card_balance):
    print(credit_card_balance.shape)

    print(
        credit_card_balance[
            ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"]
        ].nunique()
    )

    print(
        credit_card_balance.groupby("SK_ID_CURR")["SK_ID_PREV"]
        .nunique()
        .describe()
    )
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance.groupby('SK_ID_PREV')['MONTHS_BALANCE'].nunique().describe()
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance[
        ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"]
    ].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Missingness
    """)
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance.isna().mean().sort_values(ascending=False)
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC1 - Activity
    """)
    return


@app.cell
def _():
    # credit_card_balance["CC_HAS_BALANCE"] = (
    #     credit_card_balance["AMT_BALANCE"] != 0
    # ).astype("int8")

    # credit_card_balance["CC_HAS_DRAWINGS"] = (
    #     credit_card_balance["AMT_DRAWINGS_CURRENT"] != 0
    # ).astype("int8")

    # cc_activity = (
    #     credit_card_balance.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
    #         CC_MONTHS_OBSERVED=("MONTHS_BALANCE", "count"),
    #         CC_HISTORY_AGE_MONTHS=("MONTHS_BALANCE", lambda x: -x.min()),
    #         CC_MONTHS_SINCE_LATEST=("MONTHS_BALANCE", lambda x: -x.max()),
    #         CC_ACTIVE_BALANCE_MONTH_SHARE=("CC_HAS_BALANCE", "mean"),
    #         CC_DRAWING_MONTH_SHARE=("CC_HAS_DRAWINGS", "mean"),
    #     )
    #     .reset_index()
    # )

    # cc_activity["HAS_CREDIT_CARD_HISTORY"] = 1
    return


@app.cell
def _():
    # modeling_previous_application_4 = pd.read_parquet("data/processed/modeling_application_history_4.parquet")
    return


@app.cell
def _():
    # modeling_cc1 = modeling_previous_application_4.merge(
    #     cc_activity,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )

    # modeling_cc1['HAS_CREDIT_CARD_HISTORY'] = modeling_cc1['HAS_CREDIT_CARD_HISTORY'].fillna(0).astype('int8')
    return


@app.cell
def _():
    # modeling_cc1['SK_ID_CURR'].duplicated().sum()
    return


@app.cell
def _():
    # modeling_cc1.to_parquet("data/processed/modeling_cc1.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC2 - balance and utilization
    """)
    return


@app.cell
def _(credit_card_balance):
    (credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"] == 0).mean()
    return


@app.cell
def _(credit_card_balance, np):
    credit_card_balance["CC_UTILIZATION"] = (
        credit_card_balance["AMT_BALANCE"]
        / credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    )
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance["CC_UTILIZATION"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    )
    return


@app.cell
def _(credit_card_balance):
    cc_latest = (
        credit_card_balance
        .sort_values(["SK_ID_PREV", "MONTHS_BALANCE"])
        .groupby("SK_ID_PREV")
        .tail(1)
    )
    return


@app.cell
def _():
    # cc_balance_history = (
    #     credit_card_balance.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_MEAN_BALANCE=("AMT_BALANCE", "mean"),
    #         CC_MAX_BALANCE=("AMT_BALANCE", "max"),

    #         CC_MEAN_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
    #         CC_MAX_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "max"),

    #         CC_MEAN_UTILIZATION=("CC_UTILIZATION", "mean"),
    #         CC_MAX_UTILIZATION=("CC_UTILIZATION", "max"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # cc_latest_history = (
    #     cc_latest.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_LATEST_BALANCE=("AMT_BALANCE", "mean"),
    #         CC_LATEST_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
    #         CC_LATEST_UTILIZATION=("CC_UTILIZATION", "mean"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # cc_2 = cc_balance_history.merge(
    #     cc_latest_history,
    #     on="SK_ID_CURR",
    #     how="left",
    # )
    return


@app.cell
def _():
    # modeling_cc2 = modeling_cc1.merge(
    #     cc_2,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_cc2.to_parquet("data/processed/modeling_cc2.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC3 - drawings and payments
    """)
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance
    return


@app.cell
def _(credit_card_balance):
    DRAWINGS_PAYMENT_COLUMNS = [
        column for column in credit_card_balance.columns
        if (column.startswith("AMT_DRAWINGS_") or column.startswith("CNT_DRAWINGS_")) or
        (column.startswith("AMT_PAYMENT_") or column.startswith("CNT_PAYMENT_"))
    ]
    return (DRAWINGS_PAYMENT_COLUMNS,)


@app.cell
def _(DRAWINGS_PAYMENT_COLUMNS, credit_card_balance):
    credit_card_balance[DRAWINGS_PAYMENT_COLUMNS].describe()
    return


@app.cell
def _(credit_card_balance, np):
    credit_card_balance["CC_NET_DRAWINGS"] = (
        credit_card_balance["AMT_DRAWINGS_CURRENT"]
        - credit_card_balance["AMT_PAYMENT_TOTAL_CURRENT"]
    )

    credit_card_balance["CC_ATM_DRAWING_SHARE"] = (
        credit_card_balance["AMT_DRAWINGS_ATM_CURRENT"]
        / credit_card_balance["AMT_DRAWINGS_CURRENT"].replace(0, np.nan)
    )
    return


@app.cell
def _():
    # cc_drawings_payments = (
    #     credit_card_balance.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_MEAN_DRAWINGS=("AMT_DRAWINGS_CURRENT", "mean"),
    #         CC_MAX_DRAWINGS=("AMT_DRAWINGS_CURRENT", "max"),

    #         CC_MEAN_ATM_DRAWINGS =("AMT_DRAWINGS_ATM_CURRENT", "mean"),
    #         CC_MAX_ATM_DRAWINGS =("AMT_DRAWINGS_ATM_CURRENT", "max"),

    #         CC_MEAN_DRAWING_COUNT=("CNT_DRAWINGS_CURRENT", "mean"),
    #         CC_MAX_DRAWING_COUNT=("CNT_DRAWINGS_CURRENT", "max"),

    #         CC_MEAN_PAYMENTS=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
    #         CC_MAX_PAYMENTS=("AMT_PAYMENT_TOTAL_CURRENT", "max"),

    #         CC_MEAN_NET_DRAWINGS=("CC_NET_DRAWINGS", "mean"),
    #         CC_MAX_NET_DRAWINGS=("CC_NET_DRAWINGS", "max"),

    #         CC_MEAN_ATM_DRAWING_SHARE=("CC_ATM_DRAWING_SHARE", "mean"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # cc_drawings_payments_latest = (
    #     cc_latest.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_LATEST_DRAWINGS=("AMT_DRAWINGS_CURRENT", "mean"),
    #         CC_LATEST_PAYMENTS=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # cc3 = cc_drawings_payments.merge(
    #     cc_drawings_payments_latest,
    #     on="SK_ID_CURR",
    #     how="left",
    # )
    return


@app.cell
def _():
    # modeling_cc3 = modeling_cc2.merge(
    #     cc3,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_cc3.to_parquet("data/processed/modeling_cc3.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC4 - delinquency and stress
    """)
    return


@app.cell
def _(credit_card_balance):
    credit_card_balance["CC_HAS_DPD"] = (
        credit_card_balance["SK_DPD"] > 0
    ).astype("int8")

    credit_card_balance["CC_HAS_DPD_DEF"] = (
        credit_card_balance["SK_DPD_DEF"] > 0
    ).astype("int8")

    credit_card_balance["CC_DPD_30_PLUS"] = (
        credit_card_balance["SK_DPD"] >= 30
    ).astype("int8")

    credit_card_balance["CC_DPD_90_PLUS"] = (
        credit_card_balance["SK_DPD"] >= 90
    ).astype("int8")
    return


@app.cell
def _():
    # cc_delinquency = (
    #     credit_card_balance.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_MAX_DPD=("SK_DPD", "max"),
    #         CC_MEAN_DPD=("SK_DPD", "mean"),
    #         CC_DPD_MONTH_SHARE=("CC_HAS_DPD", "mean"),

    #         CC_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
    #         CC_DPD_DEF_MONTH_SHARE=("CC_HAS_DPD_DEF", "mean"),

    #         CC_DPD_30_PLUS_MONTH_SHARE=("CC_DPD_30_PLUS", "mean"),
    #         CC_DPD_90_PLUS_MONTH_SHARE=("CC_DPD_90_PLUS", "mean"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # cc_recent_6m = credit_card_balance[
    #     credit_card_balance["MONTHS_BALANCE"] >= -6
    # ]

    # cc_recent_dpd = (
    #     cc_recent_6m.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_RECENT_6M_MAX_DPD=("SK_DPD", "max"),
    #         CC_RECENT_6M_DPD_MONTH_SHARE=("CC_HAS_DPD", "mean"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # cc_latest_dpd = (
    #     cc_latest.groupby("SK_ID_CURR")
    #     .agg(
    #         CC_LATEST_MAX_DPD=("SK_DPD", "max"),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # cc_4 = (
    #     cc_delinquency
    #     .merge(cc_recent_dpd, on="SK_ID_CURR", how="left")
    #     .merge(cc_latest_dpd, on="SK_ID_CURR", how="left")
    # )
    return


@app.cell
def _():
    # modeling_cc4 = modeling_cc2.merge(
    #     cc_4,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_cc4.to_parquet("data/processed/modeling_cc4.parquet", index=False)
    return


@app.cell
def _(pd):
    cc4_df = pd.read_parquet("data/processed/modeling_cc4.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Installments payments
    """)
    return


@app.cell
def _(installments_payments):
    installments_payments.head()
    return


@app.cell
def _(installments_payments):
    installments_payments.shape
    return


@app.cell
def _(installments_payments):
    installments_payments[["SK_ID_CURR", "SK_ID_PREV"]].nunique()
    return


@app.cell
def _(installments_payments):
    installments_payments[
            [
                "NUM_INSTALMENT_VERSION",
                "NUM_INSTALMENT_NUMBER",
                "DAYS_INSTALMENT",
                "DAYS_ENTRY_PAYMENT",
                "AMT_INSTALMENT",
                "AMT_PAYMENT",
            ]
        ].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Missingness
    """)
    return


@app.cell
def _(installments_payments):
    installments_payments.isna().mean().sort_values(ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grain
    """)
    return


@app.cell
def _(installments_payments):
    installments_payments.groupby(
        ["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]
    ).size().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time-related
    """)
    return


@app.cell
def _(installments_payments):
    print(
        (installments_payments[
            ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"]
        ] > 0).sum()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IP1 - structure and history
    """)
    return


@app.cell
def _():
    # installments_history = (
    #     installments_payments.groupby("SK_ID_CURR")
    #     .agg(
    #         IP_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
    #         IP_PAYMENT_RECORD_COUNT=("SK_ID_PREV", "size"),

    #         IP_HISTORY_AGE_DAYS=(
    #             "DAYS_ENTRY_PAYMENT",
    #             lambda x: -x.min(),
    #         ),
    #         IP_DAYS_SINCE_LAST_PAYMENT=(
    #             "DAYS_ENTRY_PAYMENT",
    #             lambda x: -x.max(),
    #         ),
    #     )
    #     .reset_index()
    # )

    # unique_installments = (
    #     installments_payments[
    #         ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]
    #     ]
    #     .drop_duplicates()
    #     .groupby("SK_ID_CURR")
    #     .size()
    #     .rename("IP_UNIQUE_INSTALLMENT_COUNT")
    #     .reset_index()
    # )

    # installments_history = installments_history.merge(
    #     unique_installments,
    #     on="SK_ID_CURR",
    #     how="left",
    # )
    return


@app.cell
def _():
    # modeling_ip1 = cc4_df.merge(
    #     installments_history,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_ip1['SK_ID_CURR'].duplicated().sum()
    return


@app.cell
def _():
    # modeling_ip1.to_parquet("data/processed/modeling_ip1.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IP2 - repayment discipline
    """)
    return


@app.cell
def _():
    # installments_payments.groupby(
    #     [
    #         "SK_ID_PREV",
    #         "NUM_INSTALMENT_VERSION",
    #         "NUM_INSTALMENT_NUMBER",
    #     ]
    # ).size().describe()
    return


@app.cell
def _():
    # installment_level = (
    #     installments_payments
    #     .groupby(
    #         [
    #             "SK_ID_CURR",
    #             "SK_ID_PREV",
    #             "NUM_INSTALMENT_VERSION",
    #             "NUM_INSTALMENT_NUMBER",
    #         ],
    #         as_index=False,
    #     )
    #     .agg(
    #         DAYS_INSTALMENT=("DAYS_INSTALMENT", "first"),
    #         LAST_PAYMENT_DAY=("DAYS_ENTRY_PAYMENT", "max"),

    #         AMT_INSTALMENT=("AMT_INSTALMENT", "first"),
    #         TOTAL_PAYMENT=("AMT_PAYMENT", "sum"),
    #     )
    # )
    return


@app.cell
def _():
    # installment_level["IP_DELAY_DAYS"] = (
    #     installment_level["LAST_PAYMENT_DAY"]
    #     - installment_level["DAYS_INSTALMENT"]
    # ).clip(lower=0)

    # installment_level["IP_IS_LATE"] = (
    #     installment_level["IP_DELAY_DAYS"] > 0
    # ).astype("int8")

    # installment_level["IP_IS_30_PLUS_LATE"] = (
    #     installment_level["IP_DELAY_DAYS"] >= 30
    # ).astype("int8")

    # installment_level["IP_PAYMENT_SHORTFALL"] = (
    #     installment_level["AMT_INSTALMENT"]
    #     - installment_level["TOTAL_PAYMENT"]
    # ).clip(lower=0)

    # installment_level["IP_IS_UNDERPAID"] = (
    #     installment_level["IP_PAYMENT_SHORTFALL"] > 0
    # ).astype("int8")

    # installment_level["IP_PAYMENT_COVERAGE_RATIO"] = (
    #     installment_level["TOTAL_PAYMENT"]
    #     / installment_level["AMT_INSTALMENT"].replace(0, np.nan)).clip(upper=1)
    return


@app.cell
def _():
    # installment_level[
    #     [
    #         "IP_DELAY_DAYS",
    #         "IP_PAYMENT_SHORTFALL",
    #         "IP_PAYMENT_COVERAGE_RATIO",
    #     ]
    # ].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    return


@app.cell
def _():
    # installment_level[
    #     [
    #         "IP_IS_LATE",
    #         "IP_IS_30_PLUS_LATE",
    #         "IP_IS_UNDERPAID",
    #     ]
    # ].mean()
    return


@app.cell
def _():
    # ip2_history = (
    #     installment_level.groupby("SK_ID_CURR")
    #     .agg(
    #         IP_LATE_INSTALLMENT_SHARE=(
    #             "IP_IS_LATE",
    #             "mean",
    #         ),
    #         IP_MEAN_DELAY_DAYS=(
    #             "IP_DELAY_DAYS",
    #             "mean",
    #         ),
    #         IP_MAX_DELAY_DAYS=(
    #             "IP_DELAY_DAYS",
    #             "max",
    #         ),
    #         IP_30_PLUS_LATE_SHARE=(
    #             "IP_IS_30_PLUS_LATE",
    #             "mean",
    #         ),

    #         IP_UNDERPAID_INSTALLMENT_SHARE=(
    #             "IP_IS_UNDERPAID",
    #             "mean",
    #         ),
    #         IP_MEAN_PAYMENT_SHORTFALL=(
    #             "IP_PAYMENT_SHORTFALL",
    #             "mean",
    #         ),
    #         IP_MAX_PAYMENT_SHORTFALL=(
    #             "IP_PAYMENT_SHORTFALL",
    #             "max",
    #         ),

    #         IP_MEAN_PAYMENT_COVERAGE_RATIO=(
    #             "IP_PAYMENT_COVERAGE_RATIO",
    #             "mean",
    #         ),
    #         IP_MIN_PAYMENT_COVERAGE_RATIO=(
    #             "IP_PAYMENT_COVERAGE_RATIO",
    #             "min",
    #         ),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # modeling_ip2 = modeling_ip1.merge(
    #     ip2_history,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_ip2.to_parquet("data/processed/modeling_ip2.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IP3 - Recent repayment discipline
    """)
    return


@app.cell
def _():
    # recent_6m = installment_level[
    #     installment_level["DAYS_INSTALMENT"] >= -180
    # ]

    # recent_12m = installment_level[
    #     installment_level["DAYS_INSTALMENT"] >= -365
    # ]
    return


@app.cell
def _():
    # ip_recent_6m = (
    #     recent_6m.groupby("SK_ID_CURR")
    #     .agg(
    #         IP_RECENT_6M_INSTALLMENT_COUNT=(
    #             "NUM_INSTALMENT_NUMBER",
    #             "size",
    #         ),
    #         IP_RECENT_6M_LATE_SHARE=(
    #             "IP_IS_LATE",
    #             "mean",
    #         ),
    #         IP_RECENT_6M_30_PLUS_LATE_SHARE=(
    #             "IP_IS_30_PLUS_LATE",
    #             "mean",
    #         ),
    #         IP_RECENT_6M_MAX_DELAY_DAYS=(
    #             "IP_DELAY_DAYS",
    #             "max",
    #         ),
    #         IP_RECENT_6M_UNDERPAID_SHARE=(
    #             "IP_IS_UNDERPAID",
    #             "mean",
    #         ),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # ip_recent_12m = (
    #     recent_12m.groupby("SK_ID_CURR")
    #     .agg(
    #         IP_RECENT_12M_INSTALLMENT_COUNT=(
    #             "NUM_INSTALMENT_NUMBER",
    #             "size",
    #         ),
    #         IP_RECENT_12M_LATE_SHARE=(
    #             "IP_IS_LATE",
    #             "mean",
    #         ),
    #         IP_RECENT_12M_30_PLUS_LATE_SHARE=(
    #             "IP_IS_30_PLUS_LATE",
    #             "mean",
    #         ),
    #         IP_RECENT_12M_MAX_DELAY_DAYS=(
    #             "IP_DELAY_DAYS",
    #             "max",
    #         ),
    #         IP_RECENT_12M_UNDERPAID_SHARE=(
    #             "IP_IS_UNDERPAID",
    #             "mean",
    #         ),
    #     )
    #     .reset_index()
    # )
    return


@app.cell
def _():
    # ip3_history = ip_recent_6m.merge(
    #     ip_recent_12m,
    #     on="SK_ID_CURR",
    #     how="outer",
    # )
    return


@app.cell
def _():
    # modeling_ip3 = modeling_ip2.merge(
    #     ip3_history,
    #     on="SK_ID_CURR",
    #     how="left",
    #     validate="one_to_one",
    # )
    return


@app.cell
def _():
    # modeling_ip3.to_parquet("data/processed/modeling_ip3.parquet", index=False)
    return


@app.cell
def _(pd):
    ip3_df = pd.read_parquet("data/processed/modeling_ip3.parquet")
    return (ip3_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F8 Candidate — Installments Contract-Level Dynamic Features

    ### Objective / Hypothesis
    Test whether a two-stage aggregation (`installment -> previous contract -> applicant`) capturing contract-level repayment heterogeneity (worst-contract lateness, severe delinquency share, underpayment, latest contract behavior, and recency-weighted patterns) exposes risk signals averaged away by client-level aggregations.
    """)
    return


@app.cell
def _(np):
    def build_installments_contract_features(installments):
        cols = [
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ]

        x = installments[cols].copy()

        # Only installments whose scheduled due date is before the current application.
        x = x.loc[
            x["DAYS_INSTALMENT"] <= 0
        ].copy()

        # Actual payments after the current application are unavailable at cutoff.
        known_payment = (
            x["DAYS_ENTRY_PAYMENT"].notna()
            & (x["DAYS_ENTRY_PAYMENT"] <= 0)
        )

        x["IPX_DAYS_LATE"] = np.where(
            known_payment,
            (
                x["DAYS_ENTRY_PAYMENT"]
                - x["DAYS_INSTALMENT"]
            ).clip(lower=0),
            np.nan,
        )

        x["IPX_PAYMENT_RATIO"] = np.where(
            known_payment,
            x["AMT_PAYMENT"]
            / x["AMT_INSTALMENT"].replace(0, np.nan),
            np.nan,
        )

        # Limit extreme partial/overpayment values.
        x["IPX_PAYMENT_RATIO"] = (
            x["IPX_PAYMENT_RATIO"]
            .clip(0, 3)
        )

        x["IPX_LATE"] = (
            x["IPX_DAYS_LATE"] > 0
        ).astype(float)

        x["IPX_LATE_7D"] = (
            x["IPX_DAYS_LATE"] > 7
        ).astype(float)

        x["IPX_LATE_30D"] = (
            x["IPX_DAYS_LATE"] > 30
        ).astype(float)

        x["IPX_UNDERPAID"] = (
            x["IPX_PAYMENT_RATIO"] < 0.95
        ).astype(float)

        # Make unavailable actual-payment records missing rather than "good".
        for col in [
            "IPX_LATE",
            "IPX_LATE_7D",
            "IPX_LATE_30D",
            "IPX_UNDERPAID",
        ]:
            x.loc[~known_payment, col] = np.nan

        contract = (
            x.groupby(
                ["SK_ID_CURR", "SK_ID_PREV"],
                observed=True,
            )
            .agg(
                IPX_CONTRACT_N_INSTALLMENTS=(
                    "NUM_INSTALMENT_NUMBER",
                    "nunique",
                ),
                IPX_CONTRACT_LAST_DUE_DAY=(
                    "DAYS_INSTALMENT",
                    "max",
                ),

                IPX_CONTRACT_LATE_SHARE=(
                    "IPX_LATE",
                    "mean",
                ),
                IPX_CONTRACT_LATE_7D_SHARE=(
                    "IPX_LATE_7D",
                    "mean",
                ),
                IPX_CONTRACT_LATE_30D_SHARE=(
                    "IPX_LATE_30D",
                    "mean",
                ),

                IPX_CONTRACT_MEAN_DAYS_LATE=(
                    "IPX_DAYS_LATE",
                    "mean",
                ),
                IPX_CONTRACT_MAX_DAYS_LATE=(
                    "IPX_DAYS_LATE",
                    "max",
                ),

                IPX_CONTRACT_MEAN_PAYMENT_RATIO=(
                    "IPX_PAYMENT_RATIO",
                    "mean",
                ),
                IPX_CONTRACT_MIN_PAYMENT_RATIO=(
                    "IPX_PAYMENT_RATIO",
                    "min",
                ),
                IPX_CONTRACT_UNDERPAID_SHARE=(
                    "IPX_UNDERPAID",
                    "mean",
                ),
            )
            .reset_index()
        )

        # Contract-level flags.
        contract["IPX_BAD_CONTRACT"] = (
            (contract["IPX_CONTRACT_LATE_30D_SHARE"] > 0)
            | (contract["IPX_CONTRACT_UNDERPAID_SHARE"] > 0.10)
        ).astype(float)

        contract["IPX_RECENCY_WEIGHT"] = np.exp(
            contract[
                "IPX_CONTRACT_LAST_DUE_DAY"
            ].clip(lower=-3650)
            / 365.0
        )

        client = (
            contract.groupby(
                "SK_ID_CURR",
                observed=True,
            )
            .agg(
                IPX_N_CONTRACTS=(
                    "SK_ID_PREV",
                    "nunique",
                ),

                IPX_MEAN_CONTRACT_LATE_SHARE=(
                    "IPX_CONTRACT_LATE_SHARE",
                    "mean",
                ),
                IPX_MAX_CONTRACT_LATE_SHARE=(
                    "IPX_CONTRACT_LATE_SHARE",
                    "max",
                ),
                IPX_STD_CONTRACT_LATE_SHARE=(
                    "IPX_CONTRACT_LATE_SHARE",
                    "std",
                ),

                IPX_MEAN_CONTRACT_LATE30_SHARE=(
                    "IPX_CONTRACT_LATE_30D_SHARE",
                    "mean",
                ),
                IPX_MAX_CONTRACT_LATE30_SHARE=(
                    "IPX_CONTRACT_LATE_30D_SHARE",
                    "max",
                ),

                IPX_MEAN_CONTRACT_MAX_DAYS_LATE=(
                    "IPX_CONTRACT_MAX_DAYS_LATE",
                    "mean",
                ),
                IPX_WORST_CONTRACT_DAYS_LATE=(
                    "IPX_CONTRACT_MAX_DAYS_LATE",
                    "max",
                ),

                IPX_MEAN_CONTRACT_PAYMENT_RATIO=(
                    "IPX_CONTRACT_MEAN_PAYMENT_RATIO",
                    "mean",
                ),
                IPX_MIN_CONTRACT_PAYMENT_RATIO=(
                    "IPX_CONTRACT_MIN_PAYMENT_RATIO",
                    "min",
                ),

                IPX_MEAN_CONTRACT_UNDERPAID_SHARE=(
                    "IPX_CONTRACT_UNDERPAID_SHARE",
                    "mean",
                ),
                IPX_MAX_CONTRACT_UNDERPAID_SHARE=(
                    "IPX_CONTRACT_UNDERPAID_SHARE",
                    "max",
                ),

                IPX_BAD_CONTRACT_SHARE=(
                    "IPX_BAD_CONTRACT",
                    "mean",
                ),
            )
        )

        # Latest previous contract.
        latest_idx = (
            contract.groupby("SK_ID_CURR")[
                "IPX_CONTRACT_LAST_DUE_DAY"
            ]
            .idxmax()
        )

        latest = (
            contract.loc[
                latest_idx,
                [
                    "SK_ID_CURR",
                    "IPX_CONTRACT_LATE_SHARE",
                    "IPX_CONTRACT_LATE_30D_SHARE",
                    "IPX_CONTRACT_MAX_DAYS_LATE",
                    "IPX_CONTRACT_UNDERPAID_SHARE",
                ],
            ]
            .set_index("SK_ID_CURR")
            .add_prefix("IPX_LATEST_")
        )

        client = client.join(
            latest,
            how="left",
        )

        # Recency-weighted contract behavior.
        weighted_features = [
            "IPX_CONTRACT_LATE_SHARE",
            "IPX_CONTRACT_LATE_30D_SHARE",
            "IPX_CONTRACT_UNDERPAID_SHARE",
        ]

        for feature in weighted_features:
            valid = contract[feature].notna()

            numerator = (
                (
                    contract.loc[valid, feature]
                    * contract.loc[
                        valid,
                        "IPX_RECENCY_WEIGHT",
                    ]
                )
                .groupby(
                    contract.loc[
                        valid,
                        "SK_ID_CURR",
                    ]
                )
                .sum()
            )

            denominator = (
                contract.loc[
                    valid,
                    "IPX_RECENCY_WEIGHT",
                ]
                .groupby(
                    contract.loc[
                        valid,
                        "SK_ID_CURR",
                    ]
                )
                .sum()
            )

            client[
                f"IPX_WEIGHTED_{feature}"
            ] = numerator / denominator

        return client.reset_index()

    return (build_installments_contract_features,)


@app.cell
def _(build_installments_contract_features, installments_payments):
    ipx_features = build_installments_contract_features(
        installments_payments
    )

    print(ipx_features.shape)
    return (ipx_features,)


@app.cell
def _(ipx_features):
    ipx_cols = [
        c for c in ipx_features.columns
        if c.startswith("IPX_")
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # POS / cash balance
    """)
    return


@app.cell
def _(pos_cash_balance):
    pos_cash_balance.head()
    return


@app.cell
def _(pos_cash_balance):
    pos_cash_balance[
            ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"]
        ].nunique()
    return


@app.cell
def _(pos_cash_balance):
    pos_cash_balance[
            [
                "MONTHS_BALANCE",
                "CNT_INSTALMENT",
                "CNT_INSTALMENT_FUTURE",
                "SK_DPD",
                "SK_DPD_DEF",
            ]
        ].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Missingness
    """)
    return


@app.cell
def _(pos_cash_balance):
    pos_cash_balance.isna().mean().sort_values(ascending=False)
    return


@app.cell
def _(pos_cash_balance):
    pos_cash_balance.groupby(
        ["SK_ID_PREV", "MONTHS_BALANCE"]
    ).size().describe()
    return


@app.cell
def _(pos_cash_balance):
    (
        pos_cash_balance["CNT_INSTALMENT_FUTURE"]
        > pos_cash_balance["CNT_INSTALMENT"]
    ).sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## POS1 - Contract history and repayment progress
    """)
    return


@app.cell
def _(pos_cash_balance):
    pos_cash_balance["POS_COMPLETION_RATIO"] = (
        (
            pos_cash_balance["CNT_INSTALMENT"]
            - pos_cash_balance["CNT_INSTALMENT_FUTURE"]
        )
        / pos_cash_balance["CNT_INSTALMENT"]
    )
    return


@app.cell
def _(pos_cash_balance):
    pos_history = (
        pos_cash_balance.groupby("SK_ID_CURR")
        .agg(
            POS_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
            POS_MONTHS_OBSERVED=("MONTHS_BALANCE", "size"),

            POS_HISTORY_AGE_MONTHS=(
                "MONTHS_BALANCE",
                lambda x: -x.min(),
            ),
            POS_MONTHS_SINCE_LATEST=(
                "MONTHS_BALANCE",
                lambda x: -x.max(),
            ),

            POS_MEAN_INSTALMENT_COUNT=(
                "CNT_INSTALMENT",
                "mean",
            ),
            POS_MAX_INSTALMENT_COUNT=(
                "CNT_INSTALMENT",
                "max",
            ),

            POS_MEAN_INSTALMENTS_FUTURE=(
                "CNT_INSTALMENT_FUTURE",
                "mean",
            ),

            POS_MEAN_COMPLETION_RATIO=(
                "POS_COMPLETION_RATIO",
                "mean",
            ),
        )
        .reset_index()
    )
    return (pos_history,)


@app.cell
def _(pos_cash_balance):
    pos_latest = (
        pos_cash_balance
        .sort_values(["SK_ID_PREV", "MONTHS_BALANCE"])
        .groupby("SK_ID_PREV")
        .tail(1)
    )
    return (pos_latest,)


@app.cell
def _(pos_latest):
    pos_latest_history = (
        pos_latest.groupby("SK_ID_CURR")
        .agg(
            POS_LATEST_MEAN_INSTALMENTS_FUTURE=(
                "CNT_INSTALMENT_FUTURE",
                "mean",
            ),
            POS_LATEST_MAX_INSTALMENTS_FUTURE=(
                "CNT_INSTALMENT_FUTURE",
                "max",
            ),
            POS_LATEST_MEAN_COMPLETION_RATIO=(
                "POS_COMPLETION_RATIO",
                "mean",
            ),
        )
        .reset_index()
    )
    return (pos_latest_history,)


@app.cell
def _(pos_history, pos_latest_history):
    pos1 = pos_history.merge(
        pos_latest_history,
        on="SK_ID_CURR",
        how="left",
    )
    return (pos1,)


@app.cell
def _(ip3_df, pos1):
    modeling_pos1 = ip3_df.merge(
        pos1,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (modeling_pos1,)


@app.cell
def _(modeling_pos1):
    modeling_pos1.to_parquet("data/processed/modeling_pos1.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## POS2 - Delinquency history
    """)
    return


@app.cell
def _(pos_cash_balance):
    pos_cash_balance["POS_HAS_DPD"] = (
        pos_cash_balance["SK_DPD"] > 0
    ).astype("int8")

    pos_cash_balance["POS_HAS_DPD_DEF"] = (
        pos_cash_balance["SK_DPD_DEF"] > 0
    ).astype("int8")

    pos_cash_balance["POS_DPD_30_PLUS"] = (
        pos_cash_balance["SK_DPD"] >= 30
    ).astype("int8")
    return


@app.cell
def _(pos_cash_balance):
    pos_delinquency = (
        pos_cash_balance.groupby("SK_ID_CURR")
        .agg(
            POS_MAX_DPD=(
                "SK_DPD",
                "max",
            ),
            POS_MEAN_DPD=(
                "SK_DPD",
                "mean",
            ),
            POS_DPD_MONTH_SHARE=(
                "POS_HAS_DPD",
                "mean",
            ),
            POS_DPD_30_PLUS_MONTH_SHARE=(
                "POS_DPD_30_PLUS",
                "mean",
            ),

            POS_MAX_DPD_DEF=(
                "SK_DPD_DEF",
                "max",
            ),
            POS_DPD_DEF_MONTH_SHARE=(
                "POS_HAS_DPD_DEF",
                "mean",
            ),
        )
        .reset_index()
    )
    return (pos_delinquency,)


@app.cell
def _(pos_cash_balance):
    pos_recent_6m = pos_cash_balance[
        pos_cash_balance["MONTHS_BALANCE"] >= -6
    ]

    pos_recent = (
        pos_recent_6m.groupby("SK_ID_CURR")
        .agg(
            POS_RECENT_6M_MAX_DPD=(
                "SK_DPD",
                "max",
            ),
            POS_RECENT_6M_DPD_MONTH_SHARE=(
                "POS_HAS_DPD",
                "mean",
            ),
        )
        .reset_index()
    )
    return (pos_recent,)


@app.cell
def _(pos_latest):
    pos_latest_delinquency = (
        pos_latest.groupby("SK_ID_CURR")
        .agg(
            POS_LATEST_MAX_DPD=(
                "SK_DPD",
                "max",
            ),
            POS_LATEST_MAX_DPD_DEF=(
                "SK_DPD_DEF",
                "max",
            ),
        )
        .reset_index()
    )
    return (pos_latest_delinquency,)


@app.cell
def _(pos_delinquency, pos_latest_delinquency, pos_recent):
    pos2_history = (
        pos_delinquency
        .merge(
            pos_recent,
            on="SK_ID_CURR",
            how="left",
        )
        .merge(
            pos_latest_delinquency,
            on="SK_ID_CURR",
            how="left",
        )
    )
    return (pos2_history,)


@app.cell
def _(ip3_df, pos2_history):
    modeling_pos2 = ip3_df.merge(
        pos2_history,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (modeling_pos2,)


@app.cell
def _(modeling_pos2):
    modeling_pos2.to_parquet("data/processed/modeling_pos2.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F9 Candidate — POS/Cash Contract Progress and Dynamic Delinquency

    ### Objective / Hypothesis
    Construct trajectory-aware POS dynamic features separating recent 6-month behavior from full contract history, capturing remaining installment ratios, latest DPD states, and delinquency worsening.
    """)
    return


@app.cell
def _(np):
    def build_pos_dynamic_features(pos):
        x = pos[
            [
                "SK_ID_CURR",
                "SK_ID_PREV",
                "MONTHS_BALANCE",
                "CNT_INSTALMENT",
                "CNT_INSTALMENT_FUTURE",
                "SK_DPD",
                "SK_DPD_DEF",
            ]
        ].copy()

        assert (x["MONTHS_BALANCE"] <= 0).all()

        x["POSX_REMAINING_RATIO"] = (
            x["CNT_INSTALMENT_FUTURE"]
            / x["CNT_INSTALMENT"].replace(0, np.nan)
        )

        x["POSX_DPD_FLAG"] = (
            x["SK_DPD"] > 0
        ).astype(float)

        x["POSX_DPD30_FLAG"] = (
            x["SK_DPD"] > 30
        ).astype(float)

        recent6 = (
            x.loc[x["MONTHS_BALANCE"] >= -6]
            .groupby(
                ["SK_ID_CURR", "SK_ID_PREV"]
            )
            .agg(
                POSX_RECENT6_DPD_MEAN=("SK_DPD", "mean"),
                POSX_RECENT6_DPD_MAX=("SK_DPD", "max"),
                POSX_RECENT6_DPD_SHARE=("POSX_DPD_FLAG", "mean"),
                POSX_RECENT6_DPD30_SHARE=("POSX_DPD30_FLAG", "mean"),
            )
        )

        contract = (
            x.groupby(
                ["SK_ID_CURR", "SK_ID_PREV"]
            )
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

        ordered = x.sort_values(
            ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"]
        )

        oldest = (
            ordered
            .drop_duplicates(
                ["SK_ID_CURR", "SK_ID_PREV"],
                keep="first",
            )
            [
                [
                    "SK_ID_CURR",
                    "SK_ID_PREV",
                    "POSX_REMAINING_RATIO",
                ]
            ]
            .rename(
                columns={
                    "POSX_REMAINING_RATIO":
                        "POSX_OLDEST_REMAINING_RATIO"
                }
            )
        )

        latest = (
            ordered
            .drop_duplicates(
                ["SK_ID_CURR", "SK_ID_PREV"],
                keep="last",
            )
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
                    "POSX_REMAINING_RATIO":
                        "POSX_LATEST_REMAINING_RATIO",
                }
            )
        )

        contract = (
            contract
            .merge(oldest)
            .merge(latest)
        )

        contract["POSX_PROGRESS"] = (
            contract["POSX_OLDEST_REMAINING_RATIO"]
            - contract["POSX_LATEST_REMAINING_RATIO"]
        )

        contract["POSX_RECENT_WORSENING"] = (
            contract["POSX_RECENT6_DPD_SHARE"]
            - contract["POSX_DPD_SHARE"]
        )

        contract["POSX_BAD_LATEST"] = (
            contract["POSX_LATEST_DPD"] > 0
        ).astype(float)

        client = (
            contract.groupby("SK_ID_CURR")
            .agg(
                POSX_MEAN_LATEST_DPD=("POSX_LATEST_DPD", "mean"),
                POSX_MAX_LATEST_DPD=("POSX_LATEST_DPD", "max"),
                POSX_BAD_LATEST_SHARE=("POSX_BAD_LATEST", "mean"),

                POSX_MEAN_RECENT6_DPD_SHARE=(
                    "POSX_RECENT6_DPD_SHARE",
                    "mean",
                ),
                POSX_MAX_RECENT6_DPD_SHARE=(
                    "POSX_RECENT6_DPD_SHARE",
                    "max",
                ),

                POSX_MEAN_RECENT_WORSENING=(
                    "POSX_RECENT_WORSENING",
                    "mean",
                ),
                POSX_MAX_RECENT_WORSENING=(
                    "POSX_RECENT_WORSENING",
                    "max",
                ),

                POSX_MEAN_PROGRESS=(
                    "POSX_PROGRESS",
                    "mean",
                ),
                POSX_MIN_PROGRESS=(
                    "POSX_PROGRESS",
                    "min",
                ),
            )
            .reset_index()
        )

        return client

    return (build_pos_dynamic_features,)


@app.cell
def _(build_pos_dynamic_features, pos_cash_balance):
    posx_features = build_pos_dynamic_features(
        pos_cash_balance
    )
    return (posx_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bureau balance
    """)
    return


@app.cell
def _(bureau_balance):
    bureau_balance.head()
    return


@app.cell
def _(bureau_balance):
    print(bureau_balance["MONTHS_BALANCE"].min(), bureau_balance["MONTHS_BALANCE"].max())
    return


@app.cell
def _(bureau_balance):
    bureau_balance["STATUS"].value_counts(dropna=False).sort_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BB1 - monthly bureau delinquency history
    """)
    return


@app.cell
def _(bureau_balance):
    DPD_STATUSES = ["1", "2", "3", "4", "5"]
    SEVERE_DPD_STATUSES = ["3", "4", "5"]
    KNOWN_PAYMENT_STATUSES = ["0", "1", "2", "3", "4", "5"]

    severity_map = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
    }

    bureau_balance["BB_STATUS_SEVERITY"] = (
        bureau_balance["STATUS"].map(severity_map)
    )

    bureau_balance["BB_HAS_DPD"] = (
        bureau_balance["STATUS"].isin(DPD_STATUSES)
    ).astype("int8")

    bureau_balance["BB_HAS_SEVERE_DPD"] = (
        bureau_balance["STATUS"].isin(SEVERE_DPD_STATUSES)
    ).astype("int8")

    bureau_balance["BB_IS_KNOWN_STATUS"] = (
        bureau_balance["STATUS"].isin(KNOWN_PAYMENT_STATUSES)
    ).astype("int8")
    return DPD_STATUSES, KNOWN_PAYMENT_STATUSES


@app.cell
def _(bureau_balance):
    bb_base = (
        bureau_balance.groupby("SK_ID_BUREAU")
        .agg(
            BB_MONTHS_OBSERVED=("MONTHS_BALANCE", "size"),

            BB_HISTORY_AGE_MONTHS=(
                "MONTHS_BALANCE",
                lambda x: -x.min(),
            ),

            BB_MAX_STATUS_SEVERITY=(
                "BB_STATUS_SEVERITY",
                "max",
            ),

            BB_DPD_MONTH_COUNT=(
                "BB_HAS_DPD",
                "sum",
            ),

            BB_SEVERE_DPD_MONTH_COUNT=(
                "BB_HAS_SEVERE_DPD",
                "sum",
            ),

            BB_KNOWN_STATUS_MONTH_COUNT=(
                "BB_IS_KNOWN_STATUS",
                "sum",
            ),
        )
        .reset_index()
    )
    return (bb_base,)


@app.cell
def _(bb_base, np):
    bb_base["BB_DPD_MONTH_SHARE"] = (
        bb_base["BB_DPD_MONTH_COUNT"]
        / bb_base["BB_KNOWN_STATUS_MONTH_COUNT"].replace(0, np.nan)
    )

    bb_base["BB_SEVERE_DPD_MONTH_SHARE"] = (
        bb_base["BB_SEVERE_DPD_MONTH_COUNT"]
        / bb_base["BB_KNOWN_STATUS_MONTH_COUNT"].replace(0, np.nan)
    )
    return


@app.cell
def _(DPD_STATUSES, bureau_balance):
    bb_dpd = bureau_balance[
        bureau_balance["STATUS"].isin(DPD_STATUSES)
    ]

    bb_last_dpd = (
        bb_dpd.groupby("SK_ID_BUREAU")
        .agg(
            BB_MONTHS_SINCE_LAST_DPD=(
                "MONTHS_BALANCE",
                lambda x: -x.max(),
            )
        )
        .reset_index()
    )
    return (bb_last_dpd,)


@app.cell
def _(KNOWN_PAYMENT_STATUSES, bureau_balance, np):
    bb_recent = bureau_balance[
        bureau_balance["MONTHS_BALANCE"] >= -12
    ].copy()

    bb_recent["BB_RECENT_KNOWN"] = (
        bb_recent["STATUS"].isin(KNOWN_PAYMENT_STATUSES)
    ).astype("int8")

    bb_recent_agg = (
        bb_recent.groupby("SK_ID_BUREAU")
        .agg(
            BB_RECENT_12M_MAX_SEVERITY=(
                "BB_STATUS_SEVERITY",
                "max",
            ),
            BB_RECENT_12M_DPD_COUNT=(
                "BB_HAS_DPD",
                "sum",
            ),
            BB_RECENT_12M_KNOWN_COUNT=(
                "BB_RECENT_KNOWN",
                "sum",
            ),
        )
        .reset_index()
    )

    bb_recent_agg["BB_RECENT_12M_DPD_SHARE"] = (
        bb_recent_agg["BB_RECENT_12M_DPD_COUNT"]
        / bb_recent_agg["BB_RECENT_12M_KNOWN_COUNT"].replace(0, np.nan)
    )
    return (bb_recent_agg,)


@app.cell
def _(bb_base, bb_last_dpd, bb_recent_agg, bureau):
    bb_credit_features = (
        bb_base
        .merge(bb_last_dpd, on="SK_ID_BUREAU", how="left")
        .merge(bb_recent_agg, on="SK_ID_BUREAU", how="left")
    )

    bb_credit_features = bb_credit_features.merge(
        bureau[["SK_ID_BUREAU", "SK_ID_CURR"]],
        on="SK_ID_BUREAU",
        how="inner",
    )
    return (bb_credit_features,)


@app.cell
def _(bb_credit_features):
    bb_history = (
        bb_credit_features.groupby("SK_ID_CURR")
        .agg(
            BB_TOTAL_MONTHS_OBSERVED=("BB_MONTHS_OBSERVED", "sum"),
            BB_MAX_HISTORY_AGE_MONTHS=("BB_HISTORY_AGE_MONTHS", "max"),

            BB_MAX_STATUS_SEVERITY=("BB_MAX_STATUS_SEVERITY", "max"),
            BB_MEAN_DPD_MONTH_SHARE=("BB_DPD_MONTH_SHARE", "mean"),
            BB_MAX_DPD_MONTH_SHARE=("BB_DPD_MONTH_SHARE", "max"),
            BB_MEAN_SEVERE_DPD_MONTH_SHARE=("BB_SEVERE_DPD_MONTH_SHARE", "mean"),

            BB_MONTHS_SINCE_LAST_DPD=("BB_MONTHS_SINCE_LAST_DPD", "min"),

            BB_RECENT_12M_MAX_SEVERITY=("BB_RECENT_12M_MAX_SEVERITY", "max"),
            BB_RECENT_12M_MEAN_DPD_SHARE=("BB_RECENT_12M_DPD_SHARE", "mean"),
        )
        .reset_index()
    )
    return (bb_history,)


@app.cell
def _(bb_history, modeling_pos2):
    modeling_bb1 = modeling_pos2.merge(
        bb_history,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (modeling_bb1,)


@app.cell
def _(modeling_bb1):
    modeling_bb1.to_parquet("data/processed/modeling_bb1.parquet", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F9 Candidate — Bureau Balance Dynamic Delinquency Features

    ### Objective / Hypothesis
    Construct temporal bureau balance features from monthly status records, mapping status codes to ordinal severity levels and tracking recent 6-month and 12-month delinquency rates, maximum severity, and recent status worsening.
    """)
    return


@app.cell
def _(np):
    def build_bb_dynamic_features(
        bureau_balance,
        bureau,
    ):
        x = bureau_balance.merge(
            bureau[
                ["SK_ID_BUREAU", "SK_ID_CURR"]
            ],
            on="SK_ID_BUREAU",
            how="inner",
            validate="many_to_one",
        )

        assert (x["MONTHS_BALANCE"] <= 0).all()

        severity_map = {
            "C": 0,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "X": np.nan,
        }

        x["BBX_SEVERITY"] = (
            x["STATUS"].map(severity_map)
        )

        x["BBX_DELINQUENT"] = (
            x["BBX_SEVERITY"] > 0
        ).astype(float)

        recent6 = (
            x.loc[x["MONTHS_BALANCE"] >= -6]
            .groupby(
                ["SK_ID_CURR", "SK_ID_BUREAU"]
            )
            .agg(
                BBX_RECENT6_DELINQ_SHARE=(
                    "BBX_DELINQUENT",
                    "mean",
                ),
                BBX_RECENT6_MAX_SEVERITY=(
                    "BBX_SEVERITY",
                    "max",
                ),
            )
        )

        recent12 = (
            x.loc[x["MONTHS_BALANCE"] >= -12]
            .groupby(
                ["SK_ID_CURR", "SK_ID_BUREAU"]
            )
            .agg(
                BBX_RECENT12_DELINQ_SHARE=(
                    "BBX_DELINQUENT",
                    "mean",
                ),
            )
        )

        account = (
            x.groupby(
                ["SK_ID_CURR", "SK_ID_BUREAU"]
            )
            .agg(
                BBX_MONTHS_OBSERVED=(
                    "MONTHS_BALANCE",
                    "nunique",
                ),
                BBX_ALL_DELINQ_SHARE=(
                    "BBX_DELINQUENT",
                    "mean",
                ),
                BBX_MAX_SEVERITY=(
                    "BBX_SEVERITY",
                    "max",
                ),
            )
            .join(recent6)
            .join(recent12)
            .reset_index()
        )

        delinquent = x.loc[
            x["BBX_DELINQUENT"] == 1
        ]

        last_delinquency = (
            delinquent
            .groupby(
                ["SK_ID_CURR", "SK_ID_BUREAU"]
            )["MONTHS_BALANCE"]
            .max()
            .rename("BBX_LAST_DELINQUENCY_MONTH")
            .reset_index()
        )

        account = account.merge(
            last_delinquency,
            how="left",
            on=["SK_ID_CURR", "SK_ID_BUREAU"],
        )

        account["BBX_MONTHS_SINCE_DELINQUENCY"] = (
            -account["BBX_LAST_DELINQUENCY_MONTH"]
        )

        account["BBX_RECENT_WORSENING"] = (
            account["BBX_RECENT6_DELINQ_SHARE"]
            - account["BBX_ALL_DELINQ_SHARE"]
        )

        account["BBX_RECENT_DELINQUENT_ACCOUNT"] = (
            account["BBX_RECENT6_DELINQ_SHARE"] > 0
        ).astype(float)

        client = (
            account.groupby("SK_ID_CURR")
            .agg(
                BBX_RECENT_DELINQUENT_ACCOUNT_SHARE=(
                    "BBX_RECENT_DELINQUENT_ACCOUNT",
                    "mean",
                ),

                BBX_MEAN_RECENT6_DELINQ_SHARE=(
                    "BBX_RECENT6_DELINQ_SHARE",
                    "mean",
                ),
                BBX_MAX_RECENT6_DELINQ_SHARE=(
                    "BBX_RECENT6_DELINQ_SHARE",
                    "max",
                ),

                BBX_MEAN_RECENT12_DELINQ_SHARE=(
                    "BBX_RECENT12_DELINQ_SHARE",
                    "mean",
                ),

                BBX_MAX_SEVERITY=(
                    "BBX_MAX_SEVERITY",
                    "max",
                ),
                BBX_MAX_RECENT6_SEVERITY=(
                    "BBX_RECENT6_MAX_SEVERITY",
                    "max",
                ),

                BBX_MIN_MONTHS_SINCE_DELINQUENCY=(
                    "BBX_MONTHS_SINCE_DELINQUENCY",
                    "min",
                ),

                BBX_MEAN_RECENT_WORSENING=(
                    "BBX_RECENT_WORSENING",
                    "mean",
                ),
                BBX_MAX_RECENT_WORSENING=(
                    "BBX_RECENT_WORSENING",
                    "max",
                ),
            )
            .reset_index()
        )

        return client

    return (build_bb_dynamic_features,)


@app.cell
def _(build_bb_dynamic_features, bureau, bureau_balance):
    bbx_features = build_bb_dynamic_features(
        bureau_balance,
        bureau,
    )
    return (bbx_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final Research Feature Superset Assembly

    Merge all historical and candidate dynamic feature tables with the base application dataset to produce the final comprehensive research dataset (`data/processed/modeling_final.parquet`).
    """)
    return


@app.cell
def _(bbx_features, ipx_features, pd, posx_features):
    training_dataset = pd.read_parquet("data/processed/modeling_bb1.parquet")

    training_dataset = training_dataset.merge(
        ipx_features,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )

    training_dataset = training_dataset.merge(
        posx_features,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )

    training_dataset = training_dataset.merge(
        bbx_features,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (training_dataset,)


@app.cell
def _(ipx_features):
    ipx_features.columns
    return


@app.cell
def _(posx_features):
    posx_features.columns
    return


@app.cell
def _(bbx_features):
    bbx_features.columns
    return


@app.cell
def _(training_dataset):
    training_dataset.shape
    return


@app.cell
def _(training_dataset):
    training_dataset.to_parquet(
        "data/processed/modeling_final.parquet",
        index=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset freeze

    `modeling_final.parquet` is the final research superset and intentionally contains candidate features required for late-stage experiments, including the rejected F8/IPX bundle.

    It must not be interpreted as the production model schema.

    The final frozen production representation is defined exclusively by `ACCEPTED_FINAL_FEATURES` in `modeling.py`:

    - RFE2 core: 148 features
    - POSX: 9 features
    - BBX: 9 features
    - Total: 166 features

    F8/IPX and all other rejected experimental features are excluded from production inference.

    **DATASET RESEARCH STAGE CLOSED**
    """)
    return


if __name__ == "__main__":
    app.run()
