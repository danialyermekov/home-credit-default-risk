import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Home Credit Default Risk - Exploratory Data Analysis

    ## Objective

    Explore `application_train.csv` to understand:

    - dataset structure and data quality;
    - target class imbalance;
    - distributions and anomalies in features;
    - relationships between fatures and credit default risk;
    - candidate features and preprocessing decision for subsequent modeling.

    ## Scope

    This notebook covers only `application_train.csv`.
    External Home Credit tables are intentionally excluded from this stage.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contents

    0. Environment and data preparation
    1. Dataset structure
    2. Target analysis
    3. Univariate analysis
    4. Bivariate analysis
    5. Hypothesis testing
    6. Feature engineering
    7. Final dataset preparation
    8. EDA summary
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment and data preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports and settings
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px

    from scipy import stats

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )

    from scipy.stats import mannwhitneyu, chi2_contingency

    return (
        Path,
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        chi2_contingency,
        f1_score,
        mannwhitneyu,
        np,
        pd,
        plt,
        precision_score,
        px,
        recall_score,
        roc_auc_score,
        sns,
    )


@app.cell
def _(Path, pd, plt, sns):
    pd.set_option("display.max_columns", 150)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    sns.set_theme(style="whitegrid")

    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11

    RANDOM_STATE = 67

    DATA_DIR = Path("data")
    TRAIN_PATH = DATA_DIR / "raw" / "application_train.csv"
    PARQUET_PATH = DATA_DIR / "processed" / "application_train.parquet"
    return DATA_DIR, PARQUET_PATH, TRAIN_PATH


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Primary look at data
    """)
    return


@app.cell
def _(TRAIN_PATH, pd):
    df = pd.read_csv(TRAIN_PATH)
    return (df,)


@app.cell
def _(df):
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    return


@app.cell
def _(df):
    df.head(5)
    return


@app.cell
def _(df):
    df.dtypes.value_counts()
    return


@app.cell
def _(df):
    df.info(memory_usage="deep")
    return


@app.cell
def _(df, pd):
    overview = pd.Series(
        {
            "Number of rows": f"{df.shape[0]}",
            "Number of columns": f"{df.shape[1]}",
            "Number of missing values": f"{df.isnull().sum().sum()}",
            "Percentage of missing values": f"{df.isnull().mean().mean() * 100:.2f}%",
            "Number of duplicate rows": f"{df.duplicated().sum()}",
            "Percentage of duplicate rows": f"{df.duplicated().mean() * 100:.2f}%",
            "Memory usage (MB)": f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f}"
        }
    )
    overview
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** The dataset contains 307,511 rows and 122 columns, with a mix of numeric and categorical features.
    - **Evidence:** 65 float, 41 integer, and 16 string columns were detected in the original CSV.
    - **Implication:** Different feature groups will require different EDA and preprocessing strategies.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data types optimization
    """)
    return


@app.cell
def _(df):
    df_opt = df.copy()
    return (df_opt,)


@app.cell
def _(df):
    integer_columns = df.select_dtypes(include=["int64"]).columns.tolist()
    float_columns = df.select_dtypes(include=["float64"]).columns.tolist()
    object_columns = df.select_dtypes(include=["object", "str"]).columns.tolist()
    return float_columns, integer_columns, object_columns


@app.cell
def _(df, object_columns):
    df[object_columns].nunique().sort_values(ascending=False)
    return


@app.cell
def _(df_opt, float_columns, integer_columns, object_columns, pd):
    # Optimization of memory usage

    # Category data type for object columns
    for object_col in object_columns:
        df_opt[object_col] = df_opt[object_col].astype("category")

    # Downcast integer columns
    for int_col in integer_columns:
        df_opt[int_col] = pd.to_numeric(df_opt[int_col], downcast="integer")

    # Downcast float columns
    for float_col in float_columns:
        df_opt[float_col] = pd.to_numeric(df_opt[float_col], downcast="float")
    return


@app.cell
def _(df, df_opt):
    memory_before = df.memory_usage(deep=True).sum() / 1024**2
    memory_after = df_opt.memory_usage(deep=True).sum() / 1024**2

    reduction = (1 - memory_after / memory_before) * 100

    print(f"Before:    {memory_before:.2f} MB")
    print(f"After:     {memory_after:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")
    return memory_after, memory_before


@app.cell
def _(
    df_opt,
    float_columns,
    integer_columns,
    memory_after,
    memory_before,
    object_columns,
    pd,
):
    optimization_summary = pd.DataFrame(
        index=["memory_mb", "int64_count", "float64_count", "object_count", "category_count"],
        data={
            "before": [
                memory_before,
                len(integer_columns),
                len(float_columns),
                len(object_columns),
                0,
            ],
            "after": [
                memory_after,
                len(df_opt.select_dtypes(include=["int64"]).columns),
                len(df_opt.select_dtypes(include=["float64"]).columns),
                0,
                len(df_opt.select_dtypes(include=["category"]).columns),
            ],
        }
    )
    optimization_summary
    return


@app.cell
def _(df_opt):
    print(f"Float64 columns: {df_opt.select_dtypes(include=['float64']).columns.tolist()}")
    print(f"Maximum value of float64 columns: {df_opt.select_dtypes(include=['float64']).max().max()}")
    print(f"Minimum value of float64 columns: {df_opt.select_dtypes(include=['float64']).min().min()}")
    return


@app.cell
def _(df, df_opt, float_columns, integer_columns, np):
    # Checking if data was corupted during optimization

    int_cols_corupted = []
    float_cols_corupted = []

    for integer_col in integer_columns:
        if not np.array_equal(df[integer_col].to_numpy(), df_opt[integer_col].to_numpy(), equal_nan=True):
            int_cols_corupted.append(integer_col)
            print(f"Integer column {integer_col} has been corrupted during optimization.")


    for f_col in float_columns:
        if not np.allclose(df[f_col], df_opt[f_col], rtol=1e-04, atol=1e-04, equal_nan=True):
            float_cols_corupted.append(f_col)
            print(f"Float column {f_col} has been corrupted during optimization.")


    print(f"Number of corrupted integer columns: {len(int_cols_corupted)}")
    print(f"Number of corrupted float columns: {len(float_cols_corupted)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Data type optimization substantially reduced memory usage.
    - **Evidence:** Memory usage decreased from about 505 MB to 96.5 MB, a reduction of roughly 81%, while validation checks found no corrupted numeric columns.
    - **Implication:** The optimized dataset can be used safely for subsequent EDA with much lower memory overhead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save and load in parquet
    """)
    return


@app.cell
def _(PARQUET_PATH, df_opt):
    df_opt.to_parquet(
        PARQUET_PATH,
        engine="pyarrow",
        index=False,
        compression="snappy"
    )
    return


@app.cell
def _(PARQUET_PATH, pd):
    df_parquet = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
    return (df_parquet,)


@app.cell
def _(df, df_opt):
    del df
    del df_opt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Parquet is substantially faster to load than the original CSV in this workflow.
    - **Evidence:** Parquet loaded in roughly 0.5 seconds versus about 5 seconds for CSV.
    - **Implication:** Parquet will be used as the working format for the remaining EDA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset structure
    """)
    return


@app.cell
def _(df_parquet, pd):
    feature_summary = pd.DataFrame({
        "dtype": df_parquet.dtypes.astype(str),
        "nunique": df_parquet.nunique(dropna=False),
        "missing_pct": df_parquet.isnull().mean() * 100
    }).sort_values(by="nunique", ascending=False)

    feature_summary.head(50)
    return


@app.cell
def _(df_parquet):
    numeric_cardinality = (
        df_parquet.select_dtypes(include='number')
        .nunique().sort_values())

    numeric_cardinality.head(30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Features classification
    """)
    return


@app.cell
def _(pd):
    def get_semantic_type(col_name: str, series: pd.Series) -> str | None:
        if col_name == 'TARGET':
            return 'target'
        elif col_name.endswith('_ID') or col_name.startswith("SK_ID"):
            return 'identifier'
        elif col_name.startswith('DAYS_'):
            return 'duration/date-like'
        elif series.nunique() == 2:
            return 'binary'
        elif series.nunique() <= 200:
            return 'discrete'
        elif series.nunique() > 200:
            return 'continuous'

    return (get_semantic_type,)


@app.cell
def _(df_parquet, get_semantic_type, pd):
    numeric_columns = df_parquet.select_dtypes(include='number').columns
    feature_types = []
    for num_col in numeric_columns:
        semantic_type = get_semantic_type(num_col, df_parquet[num_col])
        feature_types.append({'feature': num_col, 'semantic_type': semantic_type})

    num_schema = pd.DataFrame(feature_types).sort_values(by='semantic_type', ascending=False).reset_index()
    num_schema
    return


@app.cell
def _(df_parquet):
    cat_columns = df_parquet.select_dtypes(include='category').columns
    df_parquet[cat_columns]
    return


@app.cell
def _(pd):
    cat_dict = {
        "NAME_CONTRACT_TYPE": "nominal",
        "CODE_GENDER": "binary",
        "FLAG_OWN_CAR": "binary",
        "FLAG_OWN_REALTY": "binary",
        "NAME_TYPE_SUITE": "nominal",
        "NAME_INCOME_TYPE": "nominal",
        "NAME_EDUCATION_TYPE": "ordinal",
        "NAME_FAMILY_STATUS": "nominal",
        "NAME_HOUSING_TYPE": "nominal",
        "OCCUPATION_TYPE": "nominal",
        "WEEKDAY_APPR_PROCESS_START": "day-of-week",
        "ORGANIZATION_TYPE": "nominal",
        "FONDKAPREMONT_MODE": "nominal",
        "HOUSETYPE_MODE": "nominal",
        "WALLSMATERIAL_MODE": "nominal",
        "EMERGENCYSTATE_MODE": "nominal"
    }

    cat_schema = (
        pd.Series(cat_dict, name="categorical_type")
          .rename_axis("feature")
          .reset_index()
    )
    cat_schema
    return (cat_schema,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Pandas dtypes do not fully represent the semantic role of features.
    - **Evidence:** The dataset contains identifiers, binary flags, categorical variables, count variables, continuous numeric features, and date-like duration features.
    - **Implication:** Subsequent EDA should select visualizations and statistical tests based on semantic feature type rather than dtype alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis of null values
    """)
    return


@app.cell
def _(df_parquet, pd):
    missing_summary = pd.DataFrame({
        "missing_count": df_parquet.isnull().sum(),
        "missing_pct": df_parquet.isnull().mean() * 100
    })

    missing_summary.sort_values(by="missing_pct", ascending=False).head(50)
    return (missing_summary,)


@app.cell
def _(missing_summary, pd):
    missing_summary['missing_group'] = pd.cut(
        missing_summary['missing_pct'],
        bins=[0, 5, 20, 40, 60, 80, 100],
        labels=["0-5%", "5-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        include_lowest=True
    )

    missing_summary['missing_group'].value_counts().sort_index()
    return


@app.cell
def _(missing_summary, px):
    top_missing = (
        missing_summary.sort_values('missing_pct', ascending=False)
        .head(30).reset_index(names='feature')
    )

    fig = px.bar(
        top_missing,
        x='missing_pct',
        y='feature',
        orientation='h',
        title='Top 20 Features with Highest Missing Value Percentage',
        labels={'missing_pct': 'Missing Value Percentage', 'feature': 'Feature'}
    )

    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    fig.show()
    return


@app.cell
def _(df_parquet, pd):
    def missing_analysis(missing_feature: str, groupby_feature: str ='FLAG_OWN_REALTY') -> pd.DataFrame:
        missing_by_group = (
        df_parquet
        .assign(is_missing=df_parquet[missing_feature].isna())
        .groupby(groupby_feature, observed=True)["is_missing"]
        .mean()
        .mul(100)
    )
        return missing_by_group 

    return (missing_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Features associated with houses/appartments
    """)
    return


@app.cell
def _(missing_analysis):
    missing_analysis('COMMONAREA_MODE')
    return


@app.cell
def _(missing_analysis):
    missing_analysis('COMMONAREA_MODE', 'NAME_HOUSING_TYPE')
    return


@app.cell
def _(missing_analysis):
    missing_analysis('LIVINGAPARTMENTS_AVG')
    return


@app.cell
def _(missing_analysis):
    missing_analysis('LIVINGAPARTMENTS_AVG', 'NAME_HOUSING_TYPE')
    return


@app.cell
def _(missing_analysis):
    missing_analysis('ELEVATORS_AVG')
    return


@app.cell
def _(missing_analysis):
    missing_analysis('ELEVATORS_AVG', 'NAME_HOUSING_TYPE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Other features
    """)
    return


@app.cell
def _(missing_analysis):
    missing_analysis('OWN_CAR_AGE', 'FLAG_OWN_CAR')
    return


@app.cell
def _(missing_analysis):
    missing_analysis('OCCUPATION_TYPE', 'NAME_INCOME_TYPE')
    return


@app.cell
def _(missing_analysis):
    missing_analysis(
        'EXT_SOURCE_1', 'NAME_EDUCATION_TYPE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Missing values are structured rather that uniformly distributed.
    - **Evidence:** House related features for Rented apartment have 10-20% more missing values comparing to other `NAME_HOUSING_TYPE`. `OWN_CAR_AGE` has 100% missing rate for clients without a car. `OCCUPATION_TYPE` is nearly always missing for unemployed and pensionners. Missing values of `EXT_SOURCE_1` also varies across education types with spikes for Lower secondary and Secondary/secondary special.
    -  **Implication:** Missingness is unlike to be completely random. Missing values should not be removed blindly and can be useful for modeling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Integrity check
    """)
    return


@app.cell
def _(df_parquet):
    nunique_ids = df_parquet['SK_ID_CURR'].nunique()
    n_rows = df_parquet.shape[0]

    print(f'Number of rows: {n_rows}')
    print(f'Number of unique IDs: {nunique_ids}')
    print(f'Number of duplicate ids: {n_rows - nunique_ids}')
    return


@app.cell
def _(df_parquet):
    duplicated_rows = df_parquet.duplicated().sum()
    print(f'Number of duplicated rows: {duplicated_rows}')
    return


@app.cell
def _(df_parquet):
    nunique_rows = df_parquet.nunique(dropna=False)
    constant_columns = nunique_rows[nunique_rows==1].index.tolist()
    constant_columns
    return (nunique_rows,)


@app.cell
def _(nunique_rows):
    nunique_rows.sort_values(ascending=True).head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Dataset has no duplicated rows, ids, constant columns.
    - **Evidence:** `Number of duplicated ids` = 0, `Number of duplicated rows` = 0, `constant_columns` = []
    - **Implication**: Each row represents a unique loan application identified by `SK_ID_CURR`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Target analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Target variable distribution
    """)
    return


@app.cell
def _(df_parquet):
    target_counts = df_parquet['TARGET'].value_counts().sort_index()
    print(target_counts)
    return (target_counts,)


@app.cell
def _(df_parquet):
    target_percentages = df_parquet['TARGET'].value_counts(normalize=True).sort_index()
    print(target_percentages)
    return (target_percentages,)


@app.cell
def _(pd, target_counts, target_percentages):
    target_summary = pd.DataFrame({
        "count": target_counts,
        "percentage": target_percentages * 100
    })
    target_summary
    return (target_summary,)


@app.cell
def _(target_counts):
    imbalance_ratio =target_counts[0] / target_counts[1]
    print(imbalance_ratio)
    return


@app.cell
def _(px, target_summary):
    plot_df = (target_summary.reset_index(names='TARGET'))

    target_bar_plot = px.bar(
        plot_df,
        x='TARGET',
        y='percentage',
        text='percentage',
        title='Distribution of TARGET variable',
        labels={'TARGET': 'Target', 'percentage': 'Share (%)'}
    )

    target_bar_plot.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

    target_bar_plot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Target variable is highly imbalanced, positive class is underrepresented in compare to negative one.
    - **Evidence:** `Number of positive class objects` = 24825, `number of negative class objects` = 282686. `Imbalance ratio` = 11.387.
    - **Implication:** Imbalance requires accurate choice of metric. Accuracy of dummy model that predicts negative class for all objects would be 92% for train set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline
    """)
    return


@app.cell
def _(df_parquet, np):
    y_true = df_parquet['TARGET']
    y_pred_majority = np.zeros_like(y_true, dtype=int)
    y_score_majority = np.zeros_like(y_true)
    return y_pred_majority, y_score_majority, y_true


@app.cell
def _(
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
    y_pred_majority,
    y_score_majority,
    y_true,
):
    dummy_metrics = pd.DataFrame({
        "metric": ["accuracy", "balanced_accuracy", "recall", "precision", "f1", "roc_auc", "pr"],
        "value": [
            accuracy_score(y_true, y_pred_majority),
            balanced_accuracy_score(y_true, y_pred_majority),
            recall_score(y_true, y_pred_majority),
            precision_score(y_true, y_pred_majority, zero_division=0),
            f1_score(y_true, y_pred_majority),
            roc_auc_score(y_true, y_score_majority),
            average_precision_score(y_true, y_score_majority)
        ]
    })
    dummy_metrics 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Accuracy is misleading because of the strong class imbalance.
    - **Evidence:** A classifier predicting only the negative class achieves ~91.9% accuracy, while positive-class recall and F1 are 0, balanced accuracy and ROC-AUC are 0.5, and Average Precision is approximately equal to the positive-class prevalence (~8.1%).
    - **Implication:** Model ranking should be evaluated using ROC-AUC and Average Precision, while precision, recall and F1 should be examined at an appropriately selected decision threshold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Univariative analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numerical features
    """)
    return


@app.cell
def _(df_parquet, np, pd):
    numeric_cols = [col for col in df_parquet.select_dtypes(include='number').columns.tolist() if col not in ['TARGET', "SK_ID_CURR"]]

    numeric_profile = pd.DataFrame({
        "feature": numeric_cols,
        "dtype": df_parquet[numeric_cols].dtypes.astype(str),
        "nunique": df_parquet[numeric_cols].nunique(dropna=False),
        "missing_pct": df_parquet[numeric_cols].isnull().mean() * 100,
        "mean": df_parquet[numeric_cols].mean(),
        "std": df_parquet[numeric_cols].std(),
        "min": df_parquet[numeric_cols].min(),
        "25%": df_parquet[numeric_cols].quantile(0.25),
        "median": df_parquet[numeric_cols].median(),
        "75%": df_parquet[numeric_cols].quantile(0.75),
        "max": df_parquet[numeric_cols].max(),
        "skewness": df_parquet[numeric_cols].skew(),
        "kurtosis": df_parquet[numeric_cols].kurtosis(),
        "abs_skewness": df_parquet[numeric_cols].skew().abs(),
        "max_to_median": df_parquet[numeric_cols].max() / df_parquet[numeric_cols].median().replace(0, np.nan),
    }).sort_values(by="skewness").reset_index(drop=True)

    numeric_profile
    return numeric_cols, numeric_profile


@app.cell
def _(numeric_profile):
    top15_skewness = numeric_profile[[
        'feature', 'missing_pct', 'nunique', 'median', 'mean', 'max', 'min', 'abs_skewness']]\
        .sort_values(by='abs_skewness', ascending=False).head(15)
    top15_skewness
    return


@app.cell
def _(numeric_profile):
    top15_max_to_median = numeric_profile[[
        'feature', 'missing_pct', 'nunique', 'median', 'mean', 'max', 'min', 'abs_skewness', 'max_to_median']]\
        .sort_values(by='max_to_median', ascending=False).head(15)
    top15_max_to_median
    return


@app.cell
def _(numeric_profile):
    numeric_profile[numeric_profile['feature'].isin(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT'])]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    AMT_INCOME_TOTAL - extreme right skew
    FLAG_MOBIL, FLAG_CONT_MOBILE - mean around 1 for binary feature
    FLAG_DOCUMENT_12,10,2,4,7,17,21,20,19,15 - mean around 0 for binary feature
    AMT_REQ_CREDIT_BUREAU_QRT - high skewness, max 261 for median 0

    House-related aggregated binary features: mean and median around zero:
    NONLIVINGAREA_MODE, NONLIVINGAREA_MEDI, NONLIVINGAREA_AVG - high missing pct, extreme right tail
    COMMONAREA_MODE, COMMONAREA_MEDI, COMMONAREA_AVG - 70% of missing values
    LANDAREA_MODE, LANDAREA_AVG, LANDAREA_MEDI - high missing percentage
    TOTALAREA_MODE, TOTALAREA_AVG, TOTALAREA_MEDI - high missing percentage
    LIVINGAREA_MODE, LIVINGAREA_AVG, LIVINGAREA_MEDI - high missing percentage
    BASEMANTAREA_MODE, BASEMANTAREA_AVG, BASEMANTAREA_MEDI - high_missing_percantage

    DAYS_BIRTH - negative min and max
    DAYS_EMPLOYED - negative min and imposible max
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `FLAG_MOBIL` features
    """)
    return


@app.cell
def _(df_parquet):
    df_parquet["FLAG_MOBIL"].value_counts(dropna=False)
    return


@app.cell
def _(df_parquet):
    df_parquet["FLAG_CONT_MOBILE"].value_counts(dropna=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These features are potentially useless for modeling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `FLAG_DOCUMENT` features
    """)
    return


@app.cell
def _(df_parquet):
    df_parquet["FLAG_DOCUMENT_12"].value_counts(dropna=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Several document features are extremely sparse and may have limited predictive value.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `DAYS` features
    """)
    return


@app.cell
def _(df_parquet):
    df_parquet[[ 'DAYS_BIRTH', 'DAYS_EMPLOYED']].describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This features represent number of days until loan application.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `AMT_INCOME_TOTAL` - highest skew among continuos features
    """)
    return


@app.cell
def _(px):
    def plot_distribution(df, feature, nbins=60):
        fig = px.histogram(
            df,
            x=feature,
            nbins=nbins,
            title=f'Distribution of {feature}',
            labels={feature: feature},
            marginal="box"
        )
        fig.show()

    return (plot_distribution,)


@app.cell
def _(df_parquet, plot_distribution):
    # Histogram for AMT_INCOME_TOTAL
    plot_distribution(df_parquet, 'AMT_INCOME_TOTAL', nbins=60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Extreme skew, max of 117M for median of 147150
    """)
    return


@app.cell
def _(df_parquet, plot_distribution):
    upper_amt_income = df_parquet['AMT_INCOME_TOTAL'].quantile(0.99)
    plot_distribution(df_parquet[df_parquet['AMT_INCOME_TOTAL'] <= upper_amt_income], 'AMT_INCOME_TOTAL', nbins=60)
    return (upper_amt_income,)


@app.cell
def _(df_parquet, upper_amt_income):
    print(f"Number of rows with AMT_INCOME_TOTAL > {upper_amt_income}: {len(df_parquet[df_parquet['AMT_INCOME_TOTAL'] > upper_amt_income])}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cropped to 99th quantile `AMT_INCOME_TOTAL` is still moderately skewed
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `AMT_REQ_CREDIT_BUREAU_QRT` with high missing percentage and extreme maximum value
    """)
    return


@app.cell
def _(df_parquet):
    df_parquet['AMT_REQ_CREDIT_BUREAU_QRT'].value_counts(dropna=False).sort_index(ascending=False).head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Extreme value appears once, between it and regular ones lies no values
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `DAYS_EMPLOYED` with impossible maximum value
    """)
    return


@app.cell
def _(df_parquet):
    df_parquet['DAYS_EMPLOYED'].value_counts(dropna=False).sort_index(ascending=False).head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Seems to be a flag for missing experience.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### House-related features with high missing percentage and skewness
    """)
    return


@app.cell
def _(df_parquet, plot_distribution):
    plot_distribution(df_parquet, 'NONLIVINGAREA_AVG')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Highly skewed distribution with median around 0 and large number of outliers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Several numeric features contain highly skewed distributions and suspicious extreme values.
    - **Evidence:** DAYS_EMPLOYED=365243 occurs in 55,374 rows and is physically impossible as employment duration; AMT_REQ_CREDIT_BUREAU_QRT=261 occurs once with no intermediate values; income and non-living-area features have strong right tails.
    - **Implication:** Extreme values require feature-specific treatment rather than automatic IQR-based removal. DAYS_EMPLOYED likely contains a sentinel value, while the isolated credit-bureau value requires separate handling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical features
    """)
    return


@app.cell
def _(cat_schema, df_parquet, pd):
    categorical_cols = cat_schema['feature'].tolist()

    categorical_profile = pd.DataFrame({
        "feature": categorical_cols,
        "nunique": df_parquet[categorical_cols].nunique(dropna=False),
        "missing_pct": df_parquet[categorical_cols].isnull().mean() * 100,
        "top_category": df_parquet[categorical_cols].mode().iloc[0],
        "top_category_pct": df_parquet[categorical_cols].apply(lambda x: x.value_counts(normalize=True, dropna=True).max() * 100),
        "rare_category": df_parquet[categorical_cols].apply(lambda x: x.value_counts(normalize=True, dropna=False)).apply(lambda x: x[x < 0.01].index.tolist()),
        "n_rara_categories": df_parquet[categorical_cols].apply(lambda x: (x.value_counts(normalize=True, dropna=False) < 0.01).sum())
    }).sort_values(by="nunique", ascending=False).reset_index(drop=True)

    categorical_profile
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ORGANIZATION_TYPE - 58 unique categories, 41 rare categories
    OCCUPATION_TYPE - 19 unique categories, 7 rare categories, high missing percentage
    NAME_TYPE_SUITE - top category of 80 percent, unrepresentative rare categories: ["Group of people","Other_A","Other_B",null]
    NAME_INCOME_TYPE - 4 rare categories
    """)
    return


@app.function
def category_freq_table(df, feature, top_n=10):
    freq_table = df[feature].value_counts(dropna=False).head(top_n).reset_index()
    freq_table.columns = [feature, 'count']
    freq_table['percentage'] = (freq_table['count'] / len(df)) * 100
    return freq_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `ORGANIZATION_TYPE`
    """)
    return


@app.cell
def _(df_parquet):
    category_freq_table(df_parquet, 'ORGANIZATION_TYPE', top_n=15)
    return


@app.cell
def _(df_parquet, px):
    organization_plot_df = category_freq_table(df_parquet, 'ORGANIZATION_TYPE', top_n=100)

    organization_bar_plot = px.bar(
        organization_plot_df,
        x='ORGANIZATION_TYPE',
        y='percentage',
        text='percentage',
        title='Top 15 ORGANIZATION_TYPE Categories',
        labels={'ORGANIZATION_TYPE': 'Organization Type', 'percentage': 'Share (%)'}
    )

    organization_bar_plot.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    organization_bar_plot.update_layout(xaxis_tickangle=-45, yaxis={"categoryorder": "total ascending"})
    organization_bar_plot.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `OCCUPATION_TYPE`
    """)
    return


@app.cell
def _(df_parquet):
    category_freq_table(df_parquet, 'OCCUPATION_TYPE', top_n=15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `NAME_TYPE_SUITE`
    """)
    return


@app.cell
def _(df_parquet):
    category_freq_table(df_parquet, 'NAME_TYPE_SUITE', top_n=15) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `NAME_INCOME_TYPE`
    """)
    return


@app.cell
def _(df_parquet):
    category_freq_table(df_parquet, 'NAME_INCOME_TYPE', top_n=15) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Several categorical features have highly uneven frequency distributions. `ORGANIZATION_TYPE` has a long tail of rare categories, while `NAME_TYPE_SUITE` and `NAME_INCOME_TYPE` are strongly dominated by a few common categories.

    - **Evidence:** `ORGANIZATION_TYPE` contains 58 categories, 41 of which occur in less than 1% of observations; the largest categories are `Business Entity Type 3` (22.1%), `XNA` (18.0%), and `Self-employed` (12.5%). In `NAME_TYPE_SUITE`, `Unaccompanied` accounts for 80.8% of observations. `NAME_INCOME_TYPE` is dominated by `Working` (51.6%), while several categories such as `Businessman`, `Student`, `Unemployed`, and `Maternity leave` contain only a few dozen observations. `OCCUPATION_TYPE` also contains 19 categories and 31.35% missing values, with several occupations represented by less than 1% of the dataset.

    - **Implication:** Rare categories may produce unstable estimates when their relationship with the target is analyzed because of their small sample size. Their predictive value should therefore be interpreted together with category support. Rare categories should not be merged or removed automatically before checking their relationship with the target and the behavior of the chosen model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Anomaly handling
    """)
    return


@app.cell
def _(pd):
    anomaly_table = pd.DataFrame({
        "feature": ["DAYS_EMPLOYED", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_INCOME_TOTAL", "NONLIVINGAREA_AVG", "FLAG_MOBIL", "FLAG_DOCUMENT_12"],
        "issue": ["value of 365243 appears 50k+ times", "one extreme value", "extreme outliers", "zero-centered right skew", "almost zero variance, mean around 1 for binary feature", "zero variance, mean around 0 for binary feature"],
        "interpretation": ["sentinal value for missing data", "isolated suspicious extreme value / possible data error", "extreme outliers may indicate data entry errors or rare cases", "zero-centered right skew may indicate a non-normal distribution", "almost zero variance may indicate a constant feature", "near-zero variance indicates a constant feature"],
        "planned_action": ["represent as not a number and create flag feature", "keep and investigate", "leave", "leave", "keep for now; evaluate predictive value later", "keep for now; evaluate predictive value later"],
        "reason": ["large number may affect model performance and create uncertanty in predictions and feature importance", "insufficient evidence to treat it as erroneous", "extreme numbers are not errors, they represent real-world distribution", "boosting are less sensitive for skewed distributions, values are absolutely normal", "affects performance and might not have predictive value", "affects performance and might not have predictive value"]
    })

    anomaly_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Extreme values in the dataset have different origins: some are sentinel values, some are plausible heavy-tailed observations, and some are sparse/near-constant features.
    - **Evidence:** DAYS_EMPLOYED=365243 occurs 55,374 times and is semantically impossible, whereas AMT_REQ_CREDIT_BUREAU_QRT=261 occurs only once and cannot yet be classified as an error. Income and property-area variables show natural right-skewed distributions.
    - **Implication:** Outliers should be handled feature-by-feature. Automatic IQR removal or dropping rare observations would risk removing valid information.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bivariative analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numeric features against target
    """)
    return


@app.cell
def _(df_parquet, np):
    df_analysis = df_parquet.copy()
    df_analysis['DAYS_EMPLOYED'] = (df_analysis['DAYS_EMPLOYED'].replace(365243, np.nan)).astype('float32')
    return (df_analysis,)


@app.cell
def _(df_analysis, numeric_cols):
    pearson_corr = df_analysis[numeric_cols].corrwith(df_analysis["TARGET"], method='pearson')

    spearman_corr = df_analysis[numeric_cols].corrwith(df_analysis["TARGET"], method='spearman')
    return pearson_corr, spearman_corr


@app.cell
def _(numeric_cols, pd, pearson_corr, spearman_corr):
    target_corr = pd.DataFrame({
        "feature": numeric_cols,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr
    })

    target_corr['abs_pearson_corr'] = target_corr['pearson_corr'].abs()
    target_corr['abs_spearman_corr'] = target_corr['spearman_corr'].abs()

    target_corr.sort_values(by='abs_spearman_corr', ascending=False).head(15)
    return (target_corr,)


@app.cell
def _(target_corr):
    target_corr.sort_values(by='abs_pearson_corr', ascending=False).head(15)
    return


@app.cell
def _(df_analysis, px):
    def plot_dist_target(feature, df=df_analysis, target="TARGET"):
    
        target_dist_fig = px.histogram(
            df,
            x=feature,
            color=target,
            histnorm="probability density",
            barmode="overlay",
            opacity=0.55,
            nbins=60,
            title=f"{feature} Distribution by {target}",
        )

        target_dist_fig.show()

    return (plot_dist_target,)


@app.cell
def _(df_analysis, px):
    def plot_box_target(feature, df=df_analysis, target="TARGET"):
        plot_df = (
            df[[target, feature]]
            .dropna()
            .sample(n=min(20_000, len(df)), random_state=67)
        )

        fig = px.box(
            plot_df,
            x=target,
            y=feature,
            points=False,
            title=f"{feature} Boxplot by {target}",
        )

        fig.show()

    return (plot_box_target,)


@app.function
def agg_against_target(df, feature, target='TARGET'):
    agg_df = df.groupby(target)[feature].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    return agg_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EXT_SOURCE_3
    """)
    return


@app.cell
def _(plot_dist_target):
    plot_dist_target('EXT_SOURCE_3')
    return


@app.cell
def _(plot_box_target):
    plot_box_target('EXT_SOURCE_3')
    return


@app.cell
def _(df_analysis):
    agg_against_target(df_analysis, 'EXT_SOURCE_3')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `DAYS_BIRTH`
    """)
    return


@app.cell
def _(plot_box_target):
    plot_box_target('DAYS_BIRTH')
    return


@app.cell
def _(df_analysis):
    agg_against_target(df_analysis, 'DAYS_BIRTH')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `DAYS_EMPLOYED`
    """)
    return


@app.cell
def _(df_analysis):
    agg_against_target(df_analysis, 'DAYS_EMPLOYED')
    return


@app.cell
def _(plot_box_target):
    plot_box_target('DAYS_EMPLOYED')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `REGION_RATING_CLIENT_W_CITY`
    """)
    return


@app.cell
def _(df_analysis):
    agg_against_target(df_analysis, 'REGION_RATING_CLIENT_W_CITY')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `DAYS_LAST_PHONE_CHANGE`
    """)
    return


@app.cell
def _(df_analysis):
    agg_against_target(df_analysis, 'DAYS_LAST_PHONE_CHANGE')
    return


@app.cell
def _(plot_box_target):
    plot_box_target('DAYS_LAST_PHONE_CHANGE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** `EXT_SOURCE_1/2/3` show the strongest Pearson and Spearman correlations with `TARGET`. Defaulted clients tend to be younger and have shorter employment duration. More recent phone changes are also associated with default. Most individual numeric features have relatively weak marginal correlations with the target.

    - **Evidence:** `EXT_SOURCE_3` has Pearson correlation of about `-0.179`, with median values of approximately `0.546` for `TARGET=0` and `0.379` for `TARGET=1`. `DAYS_BIRTH` medians are `-15,877` vs `-14,282`, `DAYS_EMPLOYED` medians are `-1,691` vs `-1,230`, and `DAYS_LAST_PHONE_CHANGE` medians are `-776` vs `-594`.

    - **Implication:** `EXT_SOURCE_1/2/3` appear to be promising predictors but require an explicit missing-value strategy. Age and employment duration should later be transformed into more interpretable features. Low linear correlation alone is not a reason to remove a feature, since tree-based models can exploit non-linear relationships and feature interactions. These observed patterns are suitable candidates for hypothesis testing in Section 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical features against target
    """)
    return


@app.cell
def _(df_analysis):
    overal_target_rate =df_analysis['TARGET'].mean() * 100
    return (overal_target_rate,)


@app.cell
def _(overal_target_rate, pd):
    def target_by_category(df: pd.DataFrame, feature: str, target: str = "TARGET") -> pd.DataFrame:
        target_by_cat = (
            df.groupby(feature, dropna=False, observed=True)[target]
            .agg(target_rate="mean", count="count")
            .reset_index()
        )

        target_by_cat[feature] = target_by_cat[feature].astype(str).replace({"nan": "Missing", "<NA>": "Missing", "None": "Missing"})
        target_by_cat['rate_diff'] = target_by_cat['target_rate'] - overal_target_rate / 100
        target_by_cat['abs_rate_diff'] = target_by_cat['rate_diff'].abs()
        target_by_cat["target_rate_pct"] = target_by_cat["target_rate"] * 100
        target_by_cat["category_pct"] = target_by_cat["count"] / len(df) * 100
    
        return target_by_cat.sort_values(by="abs_rate_diff", ascending=False)

    return (target_by_category,)


@app.cell
def _(overal_target_rate, px, target_by_category):
    def target_by_category_plot(df, feature, target='TARGET'):
    
        target_by_cat = target_by_category(df, feature, target)
        target_by_cat = target_by_cat.reset_index()
    
        fig = px.bar(
            target_by_cat.sort_values(by='target_rate_pct', ascending=False),
            x='target_rate_pct',
            y=feature,
            orientation='h',
            text='count',
            title=f'Default Rate by {feature}',
            labels={feature: feature, 'target_rate_pct': 'Default Rate (%)'}
        )
    
        fig.update_traces(textposition='outside')
        fig.add_vline(x=overal_target_rate, line_width=2, line_dash='dash', line_color='red', annotation_text=f'Overall Default Rate: {overal_target_rate:.2f}%', annotation_position='top right')
        fig.show()

    return (target_by_category_plot,)


@app.cell
def _():
    cat_features_previous = [
        "ORGANIZATION_TYPE",
        "OCCUPATION_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_TYPE_SUITE",
    ]
    return


@app.cell
def _(df_analysis, target_by_category):
    target_by_category(df_analysis, 'ORGANIZATION_TYPE')
    return


@app.cell
def _(df_analysis, target_by_category_plot):
    target_by_category_plot(df_analysis, 'ORGANIZATION_TYPE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `NAME_INCOME_TYPE`
    """)
    return


@app.cell
def _(df_analysis, target_by_category):
    target_by_category(df_analysis, 'NAME_INCOME_TYPE')
    return


@app.cell
def _(df_analysis, target_by_category_plot):
    target_by_category_plot(df_analysis, 'NAME_INCOME_TYPE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `OCCUPATION_TYPE`
    """)
    return


@app.cell
def _(df_analysis, target_by_category):
    target_by_category(df_analysis, 'OCCUPATION_TYPE')
    return


@app.cell
def _(df_analysis, target_by_category_plot):
    target_by_category_plot(df_analysis, 'OCCUPATION_TYPE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `NAME_TYPE_SUITE`
    """)
    return


@app.cell
def _(df_analysis, target_by_category):
    target_by_category(df_analysis, 'NAME_TYPE_SUITE')
    return


@app.cell
def _(df_analysis, target_by_category_plot):
    target_by_category_plot(df_analysis, 'NAME_TYPE_SUITE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Several categorical features show meaningful differences in default rate in both directions relative to the overall dataset rate of 8.07%. `OCCUPATION_TYPE` and `ORGANIZATION_TYPE` show the clearest separation between sufficiently represented categories, while `NAME_TYPE_SUITE` appears substantially weaker.

    - **Evidence:** For `OCCUPATION_TYPE`, low-skill laborers have a default rate of about 17.15%, drivers about 11.33%, and laborers about 10.58%, while accountants and several highly skilled occupations are substantially below the overall default rate. For `ORGANIZATION_TYPE`, `Transport: type 3` has a default rate of about 15.75% with 1,187 observations, while several other organization types are considerably below the dataset baseline. `NAME_INCOME_TYPE` also shows lower rates for pensioners (~5.39%) and state servants (~5.75%) and a higher rate for working clients (~9.59%). Extremely high rates for very rare categories such as `Maternity leave` and `Unemployed` have very small support and should therefore be interpreted cautiously.

    - **Implication:** Categorical variables may provide predictive signal through both elevated and reduced default risk. Target rate must always be interpreted together with category support because extreme rates in small groups are unstable. These associations should be formally assessed with categorical statistical tests in Section 5 rather than using target rate alone for feature selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation and multicollinearity
    """)
    return


@app.cell
def _(df_analysis):
    corr_cols = (df_analysis.select_dtypes(include='number')
                 .drop(columns=['TARGET', 'SK_ID_CURR'])
                 .columns.tolist())
    return (corr_cols,)


@app.cell
def _(corr_cols, df_analysis):
    corr_matrix = df_analysis[corr_cols].corr(method='pearson')
    return (corr_matrix,)


@app.cell
def _(corr_matrix, np):
    corr_pairs = (
        corr_matrix
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )

    corr_pairs.columns = ["feature_1", "feature_2", "correlation"]

    corr_pairs["abs_correlation"] = corr_pairs["correlation"].abs()

    corr_pairs = corr_pairs.sort_values(
        "abs_correlation",
        ascending=False,
    )

    corr_pairs.head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** The dataset contains groups of extremely highly correlated numeric features.
    - **Evidence:** The top feature pairs have correlations close to 0.99, mostly among related representations such as `*_AVG`, `*_MEDI`, and `*_MODE`.
    - **Implication:** The dataset contains substantial redundant information. This is not necessarily harmful for tree-based boosting, but redundant features may complicate interpretation and increase computational cost. Feature redundancy should therefore be considered later during feature selection rather than handled aggressively during EDA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hypothesis testing
    """)
    return


@app.cell
def _(mannwhitneyu):
    def numeric_stattest(df, feature, target='TARGET'):
        group_0 = df[df[target] == 0][feature].dropna()
        group_1 = df[df[target] == 1][feature].dropna()
    
        stat, p_value = mannwhitneyu(group_0, group_1, alternative='two-sided')
    
        print('Statistic:', stat)
        print('P-Value:', p_value)
        return stat, p_value

    return (numeric_stattest,)


@app.cell
def _(chi2_contingency, pd):
    def categorical_stattest(df, feature, target='TARGET'):
        contingency_table = pd.crosstab(df[feature].astype("string").fillna('MISSING'), df[target])
    
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
        print('Chi-Squared Statistic:', chi2)
        print('P-Value:', p_value)
        print('Degrees of Freedom:', dof)
        print('Expected Frequencies:\n', expected)
        return chi2, p_value, dof, expected

    return (categorical_stattest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypothesis 1 — EXT_SOURCE_3

    **Domain idea:** External source related to default score

    **H0:** Distribution of `EXT_SOURCE_3` is the same for both target classes

    **H1:** Distributions of `EXT_SOURCE_3` differ for TARGET=0 and TARGET=1

    **Variables:** Continous and Binary

    **Planned test:** Mann-Whitney U
    """)
    return


@app.cell
def _(df_analysis, numeric_stattest):
    numeric_stattest(df_analysis, 'EXT_SOURCE_3')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypothesis 2 — DAYS_BIRTH

    **Domain idea:** Age affects default score

    **H0:** Distribution of `DAYS_BIRTH` is the same for both target classes

    **H1:** Distributions of `DAYS_BIRTH` differ for TARGET=0 and TARGET=1

    **Variables:** Numeric and Binary

    **Planned test:** Mann-Whitney U
    """)
    return


@app.cell
def _(df_analysis, numeric_stattest):
    numeric_stattest(df_analysis, 'DAYS_BIRTH')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypothesis 3 — DAYS_EMPLOYED

    **Domain idea:** Experience affects default score

    **H0:** Distribution of `DAYS_EMPLOYED` is the same for both target classes

    **H1:** Distributions of `DAYS_EMPLOYED` differ for TARGET=0 and TARGET=1

    **Variables:** Numeric and Binary

    **Planned test:** Mann-Whitney U
    """)
    return


@app.cell
def _(df_analysis, numeric_stattest):
    numeric_stattest(df_analysis, 'DAYS_EMPLOYED')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypothesis 4 — OCCUPATION_TYPE

    **Domain idea:** Career affects default rate

    **H0:** Target is independent from `OCCUPATION_TYPE`

    **H1:** Target is dependent from `OCCUPATION_TYPE`

    **Variables:** Categorical nominal and Binary

    **Planned test:** Chi-Square
    """)
    return


@app.cell
def _(categorical_stattest, df_analysis):
    categorical_stattest(df_analysis, 'OCCUPATION_TYPE')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypothesis 5 — NAME_TYPE_SUITE

    **Domain idea:** Accompaniment affects default rate

    **H0:** `TARGET` and `NAME_TYPE_SUITE` are independent.

    **H1:** `TARGET` and `NAME_TYPE_SUITE` are associated.

    **Variables:** Categorical nominal and Binary

    **Planned test:** Chi-Square
    """)
    return


@app.cell
def _(categorical_stattest, df_analysis):
    categorical_stattest(df_analysis, 'NAME_TYPE_SUITE')
    return


@app.cell
def _(pd):
    stat_results = pd.DataFrame({
        "feature": ["EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "NAME_TYPE_SUITE"],
        "test_type": ["Mann-Whitney U", "Mann-Whitney U", "Mann-Whitney U", "Chi-Squared", "Chi-Squared"],
        "statistic": [2.958252e+09, 2926354691.5, 2.1000069e+09, 1975.0827518430256, 45.190053728369556],
        "p_value": [0.0, 0.0, 0.0, 0.0, 1.2562030207008644e-07]
    })
    stat_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Effect size
    """)
    return


@app.function
def rank_biserial_effect(df, feature, target, u_stat):
    group_0 = df[df[target] == 0][feature].dropna()
    group_1 = df[df[target] == 1][feature].dropna()

    n0 = len(group_0)
    n1 = len(group_1)

    return 1 - (2 * u_stat) / (n0 * n1)


@app.cell
def _(chi2_contingency, np, pd):
    def cramers_v(df, feature, target):
        contingency_table = pd.crosstab(df[feature].astype("string").fillna('MISSING'), df[target])
        chi2, _, _, _ = chi2_contingency(contingency_table)

        n = contingency_table.to_numpy().sum()
        r, c = contingency_table.shape

        return np.sqrt(
            chi2 / (n * min(r - 1, c - 1))
        )

    return (cramers_v,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `EXT_SOURCE_3`
    """)
    return


@app.cell
def _(df_analysis):
    print(rank_biserial_effect(df_analysis, 'EXT_SOURCE_3', 'TARGET', 2.958252e+09))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `DAYS_BIRTH`
    """)
    return


@app.cell
def _(df_analysis):
    print(rank_biserial_effect(df_analysis, 'DAYS_BIRTH', 'TARGET', 2926354691.5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `DAYS_EMPLOYED`
    """)
    return


@app.cell
def _(df_analysis):
    print(rank_biserial_effect(df_analysis, 'DAYS_EMPLOYED', 'TARGET', 2100006900))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `OCCUPATION_TYPE`
    """)
    return


@app.cell
def _(cramers_v, df_analysis):
    print(cramers_v(df_analysis, 'OCCUPATION_TYPE', 'TARGET'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `NAME_TYPE_SUITE`
    """)
    return


@app.cell
def _(cramers_v, df_analysis):
    print(cramers_v(df_analysis, 'NAME_TYPE_SUITE', 'TARGET'))
    return


@app.cell
def _(pd):
    effect_results = pd.DataFrame({
        "feature": ["EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "NAME_TYPE_SUITE"],
        "test_type": ["Mann-Whitney U", "Mann-Whitney U", "Mann-Whitney U", "Chi-Squared", "Chi-Squared"],
        "effect_size": [-0.3588, 0.1660, 0.1648, 0.0801, 0.0121],
        "p_value": [0.0, 0.0, 0.0, 0.0, 1.2562030207008644e-07],
        "interpretation": ["medium", "small", "small", "small", "negligible"]
    })
    effect_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** All five tested features show statistically significant associations with `TARGET`, but the magnitude of these associations differs substantially. `EXT_SOURCE_3` has the strongest practical effect, while `DAYS_BIRTH` and `DAYS_EMPLOYED` show weaker but still noticeable effects. `OCCUPATION_TYPE` has a small categorical association with default, whereas the effect of `NAME_TYPE_SUITE` is negligible despite its very small p-value.

    - **Evidence:** `EXT_SOURCE_3` has rank-biserial effect size `|r| ≈ 0.359`, which corresponds to a medium effect. `DAYS_BIRTH` and `DAYS_EMPLOYED` have effect sizes of approximately `0.166` and `0.165`, indicating small effects. `OCCUPATION_TYPE` has Cramér's V ≈ `0.080`, also a small effect, while `NAME_TYPE_SUITE` has Cramér's V ≈ `0.012`, which is negligible. All p-values remain below the Bonferroni-adjusted significance threshold.

    - **Implication:** Statistical significance alone is insufficient for feature evaluation, especially with a large dataset. Effect size must be considered together with p-value, sample size, model validation performance, and domain relevance. Features with tiny p-values can still have negligible practical importance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Engineering
    """)
    return


@app.function
def features(df, main_df):
    df = df.copy()

    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25)
    df['EMPLOYED_YEARS'] = -(df['DAYS_EMPLOYED'] / 365.25)
    df['AGE_AT_CURRENT_EMPLOYMENT_START'] = df['AGE_YEARS'] - df['EMPLOYED_YEARS']
    df['DAYS_EMPLOYED_ANOMALY'] = (main_df['DAYS_EMPLOYED'] == 365243).astype('int8')

    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['EMPLOYED_AGE_RATIO'] = (df['EMPLOYED_YEARS'] / df['AGE_YEARS'])

    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCES_COUNT'] = df[ext_cols].notna().sum(axis=1)

    return df


@app.cell
def _(df_analysis, df_parquet):
    df_fe = features(df_analysis, df_parquet)
    return (df_fe,)


@app.cell
def _():
    new_features = [
        'AGE_YEARS', 'EMPLOYED_YEARS', 'AGE_AT_CURRENT_EMPLOYMENT_START', 'DAYS_EMPLOYED_ANOMALY',
        'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'ANNUITY_CREDIT_RATIO',
        'EMPLOYED_AGE_RATIO', 'EXT_SOURCES_MEAN', 'EXT_SOURCES_STD',
        'EXT_SOURCES_MIN', 'EXT_SOURCES_MAX', 'EXT_SOURCES_COUNT'
    ]
    return (new_features,)


@app.cell
def _(df_fe, new_features):
    df_fe[new_features].describe().T
    return


@app.cell
def _(df_fe, new_features):
    df_fe[new_features].isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluation of engineered features
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Numeric
    """)
    return


@app.cell
def _(df_fe, new_features, pd):
    fe_corr = pd.DataFrame({
        "pearson": df_fe[new_features].corrwith(
            df_fe["TARGET"],
            method="pearson",
        ),
        "spearman": df_fe[new_features].corrwith(
            df_fe["TARGET"],
            method="spearman",
        ),
    })

    fe_corr["abs_pearson"] = fe_corr["pearson"].abs()
    fe_corr["abs_spearman"] = fe_corr["spearman"].abs()

    fe_corr.sort_values("abs_spearman", ascending=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Discrete
    """)
    return


@app.cell
def _(df_fe):
    (
        df_fe
        .groupby("EXT_SOURCES_COUNT")["TARGET"]
        .agg(["mean", "count"])
    )
    return


@app.cell
def _(df_fe):
    (
        df_fe
        .groupby("DAYS_EMPLOYED_ANOMALY")["TARGET"]
        .agg(["mean", "count"])
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** Aggregating the three external score features produced a stronger marginal association with default than any individual `EXT_SOURCE` feature. Missingness-related features also contain predictive information.

    - **Evidence:** `EXT_SOURCES_MEAN` has Pearson correlation of approximately `-0.222` with `TARGET`, compared with about `-0.179` for the strongest individual feature, `EXT_SOURCE_3`. Clients with `DAYS_EMPLOYED_ANOMALY=1` have a default rate of approximately `5.40%`, compared with `8.66%` for other clients. Default rate also decreases from about `9.92%` when only one external score is available to `7.30%` when all three are available.

    - **Implication:** Aggregating semantically related features can create additional predictive signal, and structured missingness should be preserved through explicit indicator features when appropriate. Weak marginal correlation of engineered ratio features is not sufficient evidence to remove them; their incremental value should ultimately be evaluated through cross-validation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final decisions before modeling
    """)
    return


@app.cell
def _(df_fe):
    df_model = df_fe.copy()
    y = df_model.pop("TARGET")
    ids = df_model.pop("SK_ID_CURR")
    return df_model, ids, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical missingness
    """)
    return


@app.cell
def _(df_model, pd):
    categorical = df_model.select_dtypes(include=["category", "object", "string"]).columns.tolist()

    for col in categorical:
        if isinstance(df_model[col].dtype, pd.CategoricalDtype) and "__MISSING__" not in df_model[col].cat.categories:
            df_model[col] = df_model[col].cat.add_categories("__MISSING__")

        df_model[col] = df_model[col].fillna("__MISSING__")
    return (categorical,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Housing `AVG/MEDI/MODE`
    """)
    return


@app.cell
def _(df_model):
    suffixes = ["_AVG", "_MEDI", "_MODE"]

    bases = {}

    for col_suffix in df_model.columns:
        for suffix in suffixes:
            if col_suffix.endswith(suffix):
                base = col_suffix.removesuffix(suffix)
                bases.setdefault(base, []).append(col_suffix)
    return (bases,)


@app.cell
def _(bases):
    housing_groups = {
        base: cols
        for base, cols in bases.items()
        if len(cols) == 3
    }
    return (housing_groups,)


@app.cell
def _(df_model, housing_groups, np, pd):
    housing_redundancy = []

    for base_group, cols in housing_groups.items():
        corr = df_model[cols].corr().abs()

        pair_corrs = [
            corr.iloc[0, 1],
            corr.iloc[0, 2],
            corr.iloc[1, 2],
        ]

        housing_redundancy.append({
            "base": base_group,
            "min_pair_corr": min(pair_corrs),
            "mean_pair_corr": np.mean(pair_corrs),
        })

    housing_redundancy = pd.DataFrame(housing_redundancy)
    return (housing_redundancy,)


@app.cell
def _(housing_redundancy):
    housing_redundancy.sort_values(
        "min_pair_corr",
        ascending=False,
    )
    return


@app.cell
def _(df_model, housing_redundancy):
    redundant_bases = housing_redundancy.loc[
        housing_redundancy["min_pair_corr"] >= 0.95,
        "base",
    ]

    housing_drop = []

    for redundant_base in redundant_bases:
        for suf in ["_MEDI", "_MODE"]:
            col_suf = f"{redundant_base}{suf}"

            if col_suf in df_model.columns:
                housing_drop.append(col_suf)
    return (housing_drop,)


@app.cell
def _(df_fe):
    housing_cols = [
        col for col in df_fe.columns
        if any(
            token in col
            for token in [
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
        )
    ]
    return (housing_cols,)


@app.cell
def _(df_fe, df_model, housing_cols):
    df_model["HOUSING_INFO_MISSING_PCT"] = (
        df_fe[housing_cols]
        .isna()
        .mean(axis=1)
    )
    return


@app.cell
def _(df_model, housing_drop):
    df_model.drop(columns=housing_drop, inplace=True)
    return


@app.cell
def _(categorical, df_model, housing_drop, np):
    print("Shape:", df_model.shape)
    print("Missing cells:", df_model.isna().sum().sum())
    print("Categorical columns:", len(categorical))
    print("Housing columns dropped:", len(housing_drop))
    print("Inf values:", np.isinf(
        df_model.select_dtypes(include="number")
    ).sum().sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Findings

    - **Finding:** The housing-related feature block contains substantial redundancy. All 14 `AVG/MEDI/MODE` feature groups show very high within-group correlations, with minimum pairwise correlations ranging from approximately `0.963` to `0.989`.

    - **Evidence:** Based on the `min_pair_corr >= 0.95` criterion, 28 redundant `_MEDI` and `_MODE` columns were removed while the corresponding `_AVG` features were retained. The resulting modeling dataset contains `106` features for `307,511` observations. Approximately `3.61M` missing cells remain, while no infinite values were detected.

    - **Implication:** Highly redundant housing features can be removed without discarding the underlying information represented by each feature family, reducing dimensionality and simplifying interpretation. Remaining missing values are intentionally preserved because several missingness patterns were shown to be structured and potentially informative, and CatBoost can handle numerical missing values natively. No global outlier clipping or imputation is applied before the baseline model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data types and export
    """)
    return


@app.cell
def _(df_model):
    df_final = df_model.copy()
    return (df_final,)


@app.cell
def _(df_final, pd):
    int_cols = df_final.select_dtypes(include=["integer"]).columns
    float_cols = df_final.select_dtypes(include=["float"]).columns

    for numeric_col in int_cols:
        df_final[numeric_col] = pd.to_numeric(
            df_final[numeric_col],
            downcast="integer",
        )

    for numeric_col in float_cols:
        df_final[numeric_col] = pd.to_numeric(
            df_final[numeric_col],
            downcast="float",
        )
    return


@app.cell
def _(df_final):
    df_final.select_dtypes(
        include=["category", "object", "string"]
    ).dtypes
    return


@app.cell
def _(df_final):
    df_final.select_dtypes(
        include=["category", "object", "string"]
    ).isna().sum()
    return


@app.cell
def _(df_final, np):
    numeric_final = df_final.select_dtypes(include="number")

    print(
        "Inf:",
        np.isinf(numeric_final).sum().sum()
    )

    print(
        "Columns:",
        df_final.shape[1]
    )

    df_final.dtypes.value_counts()
    return


@app.cell
def _(df_final, ids, pd, y):
    model_dataset = pd.concat(
        [
            ids.rename("SK_ID_CURR"),
            y.rename("TARGET"),
            df_final,
        ],
        axis=1,
    )

    X_model = model_dataset.drop(
        columns=["SK_ID_CURR", "TARGET"]
    )

    y_model = model_dataset["TARGET"]
    return (model_dataset,)


@app.cell
def _(DATA_DIR, model_dataset):
    FINAL_PATH = DATA_DIR / "application_train_eda_ready.parquet"

    model_dataset.to_parquet(
        FINAL_PATH,
        engine="pyarrow",
        index=False,
    )
    return (FINAL_PATH,)


@app.cell
def _(FINAL_PATH, pd):
    check_df = pd.read_parquet(FINAL_PATH)

    print(check_df.shape)
    check_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA Summary

    ## Dataset and target

    - The dataset contains 307,511 loan applications. `SK_ID_CURR` is unique, no duplicated rows or constant columns were found.
    - The target is strongly imbalanced: 24,825 defaults versus 282,686 non-defaults, corresponding to a default rate of approximately 8.07% and an imbalance ratio of about 11.4:1.
    - A majority-class classifier reaches approximately 91.9% accuracy while having zero recall for defaults, confirming that accuracy alone is unsuitable for model evaluation.

    ## Missing values and data quality

    - Missingness is highly structured rather than uniformly random.
    - Housing-related features contain approximately 60–70% missing values. Renting increases missingness in several housing variables, although housing ownership alone does not explain the pattern.
    - `OWN_CAR_AGE` is missing for virtually 100% of clients without a car and almost never missing for car owners.
    - `OCCUPATION_TYPE` missingness strongly depends on income type: it is almost always missing for pensioners and unemployed clients.
    - Missingness in `EXT_SOURCE_1` also varies substantially across education groups.
    - These patterns are consistent with structural missingness / MAR rather than purely random missingness, so numerical NaNs were not globally imputed before modeling.

    ## Numeric features and anomalies

    - Several numerical features have strongly skewed or heavy-tailed distributions. These values were not automatically clipped because extreme values are not necessarily errors and tree-based boosting is relatively robust to them.
    - `DAYS_EMPLOYED = 365243` occurs in 55,374 observations and is physically impossible as employment duration. It was treated as a sentinel value, replaced with NaN, and preserved through a `DAYS_EMPLOYED_ANOMALY` indicator.
    - `AMT_REQ_CREDIT_BUREAU_QRT = 261` occurs only once and was treated as a suspicious isolated extreme value rather than automatically deleted.
    - Housing `AVG/MEDI/MODE` representations are extremely highly correlated. Across 14 feature families, minimum within-family correlations range from approximately 0.963 to 0.989.

    ## Relationships with default

    - `EXT_SOURCE_1/2/3` show the strongest individual numerical associations with `TARGET`. `EXT_SOURCE_3` has Pearson correlation of approximately -0.179; its median is about 0.546 for non-default clients versus 0.379 for default clients.
    - Defaulted clients tend to be younger: median `DAYS_BIRTH` is approximately -14,282 versus -15,877 for non-default clients.
    - Defaulted clients also tend to have shorter current employment duration: median `DAYS_EMPLOYED` is approximately -1,230 versus -1,691 after removing the sentinel value.
    - A more recent phone change is also associated with default, although its causal interpretation is unclear.
    - Most individual numerical correlations are relatively weak, indicating that predictive performance is likely to come from combining many features, nonlinear relationships, and feature interactions rather than one dominant predictor.

    ## Categorical features

    - `OCCUPATION_TYPE` shows a meaningful risk gradient across sufficiently represented categories. For example, low-skill laborers have a default rate of about 17.2%, while several professional occupations are substantially below the overall 8.07% default rate.
    - `ORGANIZATION_TYPE` also contains categories associated with both higher and lower default risk, although many of its 58 categories have small support.
    - `NAME_INCOME_TYPE` contains useful differences among large groups: working clients have a higher default rate (~9.59%), while pensioners (~5.39%) and state servants (~5.75%) are below the dataset baseline.
    - `NAME_TYPE_SUITE` shows comparatively little separation and appears to have weak practical association with default.
    - Target rates of rare categories were interpreted together with their sample sizes; extreme rates based on only a few observations were not treated as reliable evidence.

    ## Statistical hypothesis testing

    - All five predefined hypotheses were statistically significant even after accounting for multiple comparisons.
    - Effect-size analysis showed that statistical significance does not imply a strong practical relationship.
    - `EXT_SOURCE_3` showed the largest tested effect (`|rank-biserial r| ≈ 0.359`, medium).
    - `DAYS_BIRTH` and `DAYS_EMPLOYED` showed smaller effects (~0.166 and ~0.165).
    - `OCCUPATION_TYPE` showed a small association (`Cramér's V ≈ 0.080`), while `NAME_TYPE_SUITE` was negligible (`V ≈ 0.012`) despite a very small p-value.
    - This demonstrates why p-values should be interpreted together with effect size and sample size.

    ## Feature engineering

    - Aggregating `EXT_SOURCE_1/2/3` produced stronger marginal signal than any individual external score. `EXT_SOURCES_MEAN` reached Pearson correlation of approximately -0.222 compared with about -0.179 for `EXT_SOURCE_3`.
    - The `DAYS_EMPLOYED_ANOMALY` flag is informative: clients with the sentinel value have a default rate of approximately 5.40% versus 8.66% for other clients.
    - The number of available external scores also contains information: default rate decreases from approximately 9.92% with one available source to 7.30% when all three are available.
    - Several ratio features showed weak marginal correlation, but they were retained because weak univariate association does not rule out nonlinear or interaction effects in tree-based models.

    ## Final modeling dataset

    - Highly redundant `_MEDI` and `_MODE` housing representations were removed when all members of a feature family had pairwise correlation of at least 0.95; 28 redundant columns were removed.
    - Numerical missing values were intentionally retained for CatBoost where appropriate.
    - Categorical missing values were converted into an explicit missing category.
    - No global IQR clipping, row deletion, median imputation, scaling, or removal of rare categories was applied.
    - The final modeling dataset contains 307,511 rows and 106 predictor columns, plus `SK_ID_CURR` and `TARGET`, with no infinite values.
    """)
    return


if __name__ == "__main__":
    app.run()
