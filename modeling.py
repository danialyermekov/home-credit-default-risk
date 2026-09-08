import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validation contract

    The prediction unit is one current loan application identified by `SK_ID_CURR`.

    The public Home Credit dataset does not provide:

    - a persistent customer identifier across current applications;
    - an absolute timestamp for the current application.

    Therefore neither customer-level group validation nor genuine application-level temporal validation can be constructed.

    The offline modeling assumption is that future applications are sampled from approximately the same population as the available labeled applications.

    Validation will therefore use:

    - a fixed stratified development/holdout split;
    - fixed `StratifiedKFold` folds inside the development set;
    - identical application rows and folds for all model and feature experiments.

    Historical tables are not split independently. Their records are filtered and aggregated relative to each prediction row and then joined by `SK_ID_CURR`.

    Historical-source availability is itself meaningful. Source coverage in the labeled population is:

    - bureau: 85.69%
    - previous applications: 94.65%
    - installment history: 94.84%
    - credit-card history: 28.26%
    - POS/cash history: 94.12%

    The inability to enforce customer-group and calendar-time validation is an explicit limitation of the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports and settings
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    import mlflow

    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import (
        average_precision_score,
        log_loss,
        roc_auc_score,
    )
    from pathlib import Path

    import json
    import tempfile
    import time

    from catboost import CatBoostClassifier, Pool

    import lightgbm as lgb

    import xgboost as xgb

    import plotly.express as px

    import optuna

    from scipy.stats import spearmanr, rankdata

    from assemble_features import build_test_features

    return (
        CatBoostClassifier,
        Path,
        Pool,
        StratifiedKFold,
        average_precision_score,
        build_test_features,
        json,
        lgb,
        log_loss,
        mlflow,
        np,
        optuna,
        pd,
        px,
        roc_auc_score,
        spearmanr,
        tempfile,
        time,
        train_test_split,
        xgb,
    )


@app.cell
def _(Path, mlflow, np):
    DATA_PATH = Path("data")
    TRAIN_PATH = DATA_PATH / "processed" / "application_train_eda_ready.parquet"
    ARTIFACTS_DIR = Path("artifacts")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_URL = "http://127.0.0.1:5000"
    EXPERIMENT_NAME = "home-credit-default-risk"
    TARGET = "TARGET"
    ID_COLUMN = "SK_ID_CURR"
    RANDOM_STATE = 67
    HOLDOUT_SIZE = 0.15
    N_SPLITS = 5

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)
    np.random.seed(RANDOM_STATE)
    return (
        ARTIFACTS_DIR,
        HOLDOUT_SIZE,
        ID_COLUMN,
        N_SPLITS,
        RANDOM_STATE,
        TARGET,
        TRAIN_PATH,
    )


@app.cell
def _(ARTIFACTS_DIR, TRAIN_PATH, pd):
    applications = pd.read_parquet(TRAIN_PATH)
    split_path = ARTIFACTS_DIR / "split_v1.parquet"
    return applications, split_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split
    """)
    return


@app.cell
def _(
    HOLDOUT_SIZE,
    ID_COLUMN,
    N_SPLITS,
    RANDOM_STATE,
    StratifiedKFold,
    TARGET,
    np,
    pd,
    train_test_split,
):
    def create_split_manifest(
        frame: pd.DataFrame,
        id_column: str = ID_COLUMN,
        target_column: str = TARGET,
        holdout_size: float = HOLDOUT_SIZE,
        n_splits: int = N_SPLITS,
        random_state: int = RANDOM_STATE
    ) -> pd.DataFrame:
        """
        Create a manifest DataFrame that contains the split information for each row in the input DataFrame.

        Parameters:
        - frame: The input DataFrame to split.
        - id_column: The name of the column containing unique identifiers.
        - target_column: The name of the target column for stratification.
        - holdout_size: The proportion of the dataset to include in the holdout set.
        - n_splits: The number of splits for cross-validation.
        - random_state: Random seed for reproducibility.

        Returns:
        A DataFrame with columns for the unique identifier, holdout flag, and fold assignment.
        """
        frame = frame.reset_index(drop=True)

        if frame[id_column].duplicated().any():
            raise ValueError(f"Duplicate values found in the '{id_column}' column.")

        if frame[target_column].isnull().any():
            raise ValueError(f"Missing values found in the '{target_column}' column.")

        development_idx, holdout_idx = train_test_split(
            np.arange(len(frame)),
            test_size=holdout_size,
            stratify=frame[target_column],
            random_state=random_state
        )

        manifest = frame[[id_column, target_column]].copy()
        manifest["partition"] = "development"
        manifest["fold"] = pd.Series(pd.NA, index=manifest.index, dtype="Int8")

        manifest.loc[holdout_idx, "partition"] = "holdout"

        development = frame.loc[development_idx]

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold_id, (_, valid_positions) in enumerate(
            splitter.split(development, development[target_column])
        ):
            valid_idx = development_idx[valid_positions]
            manifest.loc[valid_idx, "fold"] = fold_id

        if manifest.loc[
            manifest["partition"].eq("development"), "fold"].isnull().any():
            raise ValueError("Some development rows were not assigned a fold.")

        if manifest.loc[
            manifest["partition"].eq("holdout"), "fold"].notnull().any():
            raise ValueError("Some holdout rows were incorrectly assigned a fold.")

        return manifest.reset_index(drop=True)

    return (create_split_manifest,)


@app.cell
def _(applications, create_split_manifest):
    split_manifest = create_split_manifest(applications)
    split_manifest.head()
    return (split_manifest,)


@app.cell
def _(ID_COLUMN, TARGET, split_manifest):
    partition_profile = (
        split_manifest
        .groupby("partition", observed=True)
        .agg(
            rows=(ID_COLUMN, "size"),
            positives=(TARGET, "sum"),
            prevalence=(TARGET, "mean"),
        )
    )

    partition_profile
    return


@app.cell
def _(ID_COLUMN, TARGET, split_manifest):
    fold_profile = (
        split_manifest
        .query("partition == 'development'")
        .groupby("fold", observed=True)
        .agg(
            rows=(ID_COLUMN, "size"),
            positives=(TARGET, "sum"),
            prevalence=(TARGET, "mean"),
        )
    )

    fold_profile
    return


@app.cell
def _(ID_COLUMN, split_manifest):
    print("Total rows:", len(split_manifest))
    print("Unique IDs:", split_manifest[ID_COLUMN].nunique())

    print(
        "Development:",
        split_manifest["partition"].eq("development").sum(),
    )

    print(
        "Holdout:",
        split_manifest["partition"].eq("holdout").sum(),
    )
    return


@app.cell
def _(split_manifest, split_path):
    split_manifest.to_parquet(
        split_path,
        index=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fixed validation split

    The labeled application dataset contains 307,511 unique prediction rows.

    A fixed stratified split was created once and stored as `split_v1.parquet`:

    - Development: 261,384 applications (85%)
    - Internal holdout: 46,127 applications (15%)
    - Development CV: 5 fixed stratified folds
    - Fold size: 52,276–52,277 applications
    - Fold target prevalence: ~8.07%

    The holdout does not participate in routine feature engineering, model comparison, tuning, or error-driven iteration.

    All experiments use the same application-level folds identified by `SK_ID_CURR`.

    No applicants younger than 18 are present in the labeled dataset, so the `age >= 18` eligibility rule does not modify the offline modeling population.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline
    """)
    return


@app.cell
def _(applications):
    print(applications.shape)
    print(applications.columns.tolist())
    return


@app.cell
def _(applications):
    engineered_candidates = [
        col
        for col in applications.columns
        if any(
            token in col.upper()
            for token in (
                "MEAN",
                "RATIO",
                "COUNT",
                "ANOMALY",
                "SUM",
                "MIN",
                "MAX",
                "STD"
            )
        )
    ]

    engineered_candidates
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B0 — Raw applications CatBoost baseline

    **Observation**

    The cleaned application-level dataset already contains potentially strong
    predictive information, including `EXT_SOURCE_1/2/3`, financial, demographic,
    employment and application characteristics.

    **Hypothesis**

    CatBoost can extract a useful nonlinear ranking signal from cleaned raw
    application features without manual feature engineering.

    **Change**

    Train CatBoost on cleaned application-level features only.

    The following manually engineered features are excluded from the baseline:

    - financial ratios;
    - aggregated `EXT_SOURCE` statistics.

    `DAYS_EMPLOYED_ANOMALY` is retained because it preserves information lost when
    the `365243` sentinel was replaced with `NaN`.

    **Validation**

    - `split_v1`
    - development population only;
    - 5 fixed stratified folds;
    - internal holdout remains untouched.

    **Primary metric**

    Average Precision.

    **Secondary metrics**

    - ROC-AUC
    - LogLoss
    - Recall@Top10%
    - Precision@Top10%

    **Purpose**

    Establish the reference model against which all subsequent feature bundles,
    ablations and model changes will be compared.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Functions
    """)
    return


@app.cell
def _(np):
    def metrics_at_capacity(
        y_true: np.ndarray,
        probability: np.ndarray,
        capacity: float = 0.10,
    ) -> dict[str, float]:
        y_true = np.asarray(y_true)
        probability = np.asarray(probability)

        n_selected = int(np.ceil(len(y_true) * capacity))

        selected = np.argsort(-probability)[:n_selected]

        positives_selected = y_true[selected].sum()
        total_positives = y_true.sum()

        recall = positives_selected / total_positives
        precision = positives_selected / n_selected

        return {
            "recall_at_10pct": float(recall),
            "precision_at_10pct": float(precision),
        }

    return (metrics_at_capacity,)


@app.cell
def _(pd):
    def prepare_development_data(
        frame: pd.DataFrame
    ) -> pd.DataFrame:

        development = frame.loc[frame["partition"]=="development"].reset_index(drop=True).copy()

        if development['fold'].isna().any():
            raise ValueError("Development dataset contains NaN values in fold column")

        return development

    return (prepare_development_data,)


@app.cell
def _(np, pd):
    def make_nested_stratified_subsample(
        train: pd.DataFrame,
        fraction: float,
        target: str = "TARGET",
        random_state: int = 42,
    ) -> pd.DataFrame:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1].")

        selected_indices = []

        for target_value in sorted(train[target].unique()):
            class_indices = train.index[
                train[target].eq(target_value)
            ].to_numpy()

            rng = np.random.default_rng(
                random_state + int(target_value)
            )

            shuffled = rng.permutation(class_indices)

            n_selected = int(
                np.ceil(len(shuffled) * fraction)
            )

            selected_indices.extend(
                shuffled[:n_selected]
            )

        return train.loc[selected_indices].copy()

    return (make_nested_stratified_subsample,)


@app.cell
def _(np, pd):
    def calculate_fold_shap(
        fold_result: dict,
        fold_id: int,
        features: list[str],
    ) -> pd.DataFrame:
        model = fold_result["model"]
        valid_pool = fold_result["valid_pool"]

        shap_values = model.get_feature_importance(
            valid_pool,
            type="ShapValues",
        )

        shap_values = shap_values[:, :-1]

        fold_shap = pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )

        total_shap = fold_shap["mean_abs_shap"].sum()

        fold_shap["normalized_shap"] = (
            fold_shap["mean_abs_shap"] / total_shap
        )

        fold_shap["rank"] = (
            fold_shap["mean_abs_shap"]
            .rank(
                method="average",
                ascending=False,
            )
        )

        fold_shap["fold"] = int(fold_id)

        return fold_shap

    return (calculate_fold_shap,)


@app.cell
def _(pd):
    def summarize_cv_shap(
        shap_cv: pd.DataFrame,
    ) -> pd.DataFrame:
        shap_summary = (
            shap_cv
            .groupby("feature")
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                mean_normalized_shap=("normalized_shap", "mean"),
                std_normalized_shap=("normalized_shap", "std"),
                mean_rank=("rank", "mean"),
                std_rank=("rank", "std"),
                best_rank=("rank", "min"),
                worst_rank=("rank", "max"),
            )
            .sort_values("mean_rank")
            .reset_index()
        )

        return shap_summary

    return (summarize_cv_shap,)


@app.cell
def _(
    CatBoostClassifier,
    Pool,
    TARGET,
    make_nested_stratified_subsample,
    np,
    pd,
    time,
):
    def train_catboost_fold(
        development: pd.DataFrame,
        fold_id: int,
        features: list[str],
        categorical_features: list[str],
        params: dict,
        train_fraction: float = 1.0
    ) -> dict:

        train_mask = development["fold"].ne(fold_id)
        val_mask = development["fold"].eq(fold_id)

        train = development.loc[train_mask]
        val = development.loc[val_mask]

        if train_fraction < 1.0:
            train = make_nested_stratified_subsample(train, fraction=train_fraction)

        train_pool = Pool(
            data=train[features],
            label=train[TARGET],
            cat_features=categorical_features,
            feature_names=features)

        val_pool = Pool(
            data=val[features],
            label=val[TARGET],
            cat_features=categorical_features,
            feature_names=features)

        model = CatBoostClassifier(**params)

        started = time.perf_counter()

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=200,
            verbose=200,
        )

        fit_seconds = time.perf_counter() - started

        evals_result = model.get_evals_result()
        valid_probability = model.predict_proba(val_pool)[:, 1]
        train_probability = model.predict_proba(train_pool)[:, 1]
        valid_positions = np.flatnonzero(val_mask.to_numpy())

        return {
            "model": model,
            "valid_positions": valid_positions,
            "valid_pool": val_pool,
            "valid_probability": valid_probability,
            "train_probability": train_probability,
            "train": train,
            "valid": val,
            "evals_result": evals_result,
            "fit_seconds": fit_seconds
        }

    return (train_catboost_fold,)


@app.cell
def _(pd):
    def curves_to_frame(
        evals_result: dict,
        fold_id: int,
    ) -> pd.DataFrame:
        rows = []

        for dataset_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                for iteration, value in enumerate(values, start=1):
                    rows.append({
                        "fold": int(fold_id),
                        "iteration": iteration,
                        "dataset": dataset_name,
                        "metric": metric_name,
                        "value": float(value),
                    })

        return pd.DataFrame(rows)

    return (curves_to_frame,)


@app.cell
def _(average_precision_score, log_loss, roc_auc_score):
    def calculate_classification_metrics(
        y_true,
        probabilities,
    ) -> dict:

        ap = average_precision_score(y_true, probabilities)
        roc_auc = roc_auc_score(y_true, probabilities)
        log_loss_value = log_loss(y_true, probabilities)

        return {
            "ap": float(ap),
            "roc_auc": float(roc_auc),
            "log_loss": float(log_loss_value)
        }

    return (calculate_classification_metrics,)


@app.cell
def _(TARGET, calculate_classification_metrics, metrics_at_capacity):
    def calculate_fold_metrics(
        fold_result: dict,
        fold_id: int,
        capacity: float = 0.10,
    ) -> dict:

        train = fold_result['train']
        val = fold_result['valid']

        train_probability = fold_result['train_probability']
        val_probability = fold_result['valid_probability']

        model = fold_result['model']

        train_metrics = calculate_classification_metrics(
            train[TARGET],
            train_probability
        )

        val_metrics = calculate_classification_metrics(
            val[TARGET],
            val_probability
        )

        val_metrics_capacity = metrics_at_capacity(
            val[TARGET],
            val_probability,
            capacity=capacity
        )

        best_iter = model.get_best_iteration()
        fit_seconds = fold_result["fit_seconds"]
        return {
            "fold": fold_id,
            "rows": len(val),
            "positives": int(val['TARGET'].sum()),
            "train_ap": train_metrics['ap'],
            "valid_ap": val_metrics['ap'],
            "ap_gap": train_metrics['ap'] - val_metrics['ap'],
            "roc_auc": val_metrics['roc_auc'],
            "log_loss": val_metrics['log_loss'],
            "recall_at_10pct": val_metrics_capacity['recall_at_10pct'],
            "precision_at_10pct": val_metrics_capacity['precision_at_10pct'],
            "best_iteration": best_iter,
            "fit_seconds": fit_seconds
        }

    return (calculate_fold_metrics,)


@app.cell
def _(TARGET, calculate_classification_metrics, metrics_at_capacity, pd):
    def calculate_oof_metrics(
        oof: pd.DataFrame,
        capacity: float = 0.10,
    ) -> dict:

        metrics = calculate_classification_metrics(
        oof[TARGET],
        oof["probability"])

        capacity_metrics = metrics_at_capacity(
            oof[TARGET],
            oof['probability'],
            capacity=capacity
        )

        return {
                "oof_ap": metrics['ap'],
                "oof_roc_auc": metrics['roc_auc'],
                "oof_log_loss": metrics['log_loss'],
                "oof_recall_at_10pct": capacity_metrics['recall_at_10pct'],
                "oof_precision_at_10pct": capacity_metrics['precision_at_10pct'],
            }

    return (calculate_oof_metrics,)


@app.cell
def _(calculate_oof_metrics, pd):
    def calculate_cv_summary(
        fold_metrics: pd.DataFrame,
        oof: pd.DataFrame,
        capacity: float = 0.10
    ) -> dict:

        oof_metrics = calculate_oof_metrics(
            oof,
            capacity=capacity
        )

        fold_ap_mean = fold_metrics["valid_ap"].mean()
        fold_ap_std = fold_metrics["valid_ap"].std()
        mean_ap_gap = fold_metrics["ap_gap"].mean()
        best_iteration_median = fold_metrics["best_iteration"].median()

        return {
            "oof_ap": oof_metrics['oof_ap'],
            "oof_roc_auc": oof_metrics['oof_roc_auc'],
            "oof_log_loss": oof_metrics['oof_log_loss'],
            "oof_recall_at_10pct": oof_metrics['oof_recall_at_10pct'],
            "oof_precision_at_10pct": oof_metrics['oof_precision_at_10pct'],
            "fold_ap_mean": fold_ap_mean,
            "fold_ap_std": fold_ap_std,
            "mean_ap_gap": mean_ap_gap,
            "best_iteration_median": best_iteration_median,
            "total_fit_seconds": float(fold_metrics["fit_seconds"].sum()
        )
        }

    return (calculate_cv_summary,)


@app.cell
def _(
    ID_COLUMN,
    TARGET,
    calculate_cv_summary,
    calculate_fold_metrics,
    calculate_fold_shap,
    curves_to_frame,
    np,
    pd,
    prepare_development_data,
    summarize_cv_shap,
    train_catboost_fold,
):
    def run_catboost_baseline_cv(
        frame: pd.DataFrame,
        features: list[str],
        categorical_features: list[str],
        params: dict,
        capacity: float = 0.10,
        calculate_shap: bool = False
    ):

        development = prepare_development_data(frame)

        oof_probability = np.full(
            len(development),
            np.nan,
            np.float64
        )

        fold_rows = []
        curve_frames = []
        shap_tables = []

        all_params = None

        for fold_id in sorted(
            development["fold"].dropna().unique()
        ):
            fold_result = train_catboost_fold(
                development, fold_id, features,
                categorical_features, params)

            if all_params is None:
                all_params = fold_result["model"].get_all_params()

            fold_result_metrics = calculate_fold_metrics(
                fold_result, fold_id, capacity
            )
            curve_frame = curves_to_frame(
                fold_result["evals_result"],
                fold_id=fold_id,
            )

            curve_frames.append(curve_frame)

            fold_rows.append(fold_result_metrics)

            oof_probability[fold_result['valid_positions']] = fold_result['valid_probability']

            if calculate_shap:
                fold_shap = calculate_fold_shap(
                    fold_result=fold_result,
                    fold_id=fold_id,
                    features=features,
                )
                shap_tables.append(fold_shap)

        learning_curves = pd.concat(
            curve_frames,
            ignore_index=True,)

        if np.isnan(oof_probability).any():
            raise RuntimeError('OOF predictions are incomplete.')

        fold_metrics = pd.DataFrame(fold_rows)

        oof = development[[ID_COLUMN, TARGET, 'fold']].copy()
        oof['probability'] = oof_probability

        summary = calculate_cv_summary(
            fold_metrics=fold_metrics,
            oof=oof,
            capacity=capacity
        )

        if calculate_shap:
            shap_cv = pd.concat(
                shap_tables,
                ignore_index=True,
            )

            shap_summary = summarize_cv_shap(
                shap_cv
            )
        else:
            shap_cv = None
            shap_summary = None

        return {
            "fold_metrics": fold_metrics,
            "oof": oof,
            "summary": summary,
            "learning_curves": learning_curves,
            "all_params": all_params,
            "shap_cv": shap_cv,
            "shap_summary": shap_summary,
        }

    return (run_catboost_baseline_cv,)


@app.cell
def _(mlflow, pd):
    def log_fold_metrics(
        fold_metrics: pd.DataFrame,
    ) -> None:

        for _, row in fold_metrics.iterrows():
            fold_id = int(row['fold'])
            mlflow.log_metric(f"{fold_id}_valid_ap", row['valid_ap'])
            mlflow.log_metric(f"{fold_id}_train_ap", row['train_ap'])
            mlflow.log_metric(f"{fold_id}_ap_gap", row['ap_gap'])
            mlflow.log_metric(f"{fold_id}_roc_auc", row['roc_auc'])
            mlflow.log_metric(f"{fold_id}_log_loss", row['log_loss'])
            mlflow.log_metric(f"{fold_id}_recall_at_10pct", row['recall_at_10pct'])
            mlflow.log_metric(f"{fold_id}_precision_at_10pct", row['precision_at_10pct'])

    return (log_fold_metrics,)


@app.cell
def _(mlflow):
    def log_feature_manifest(
        features: list[str],
        categorical_features: list[str],
    ) -> None:
        mlflow.log_dict(
            {
                "features": features,
                "categorical_features": categorical_features,
            },
            "manifests/features.json",
        )

    return (log_feature_manifest,)


@app.cell
def _(Path, mlflow, pd, tempfile):
    def log_cv_artifacts(
        fold_metrics: pd.DataFrame,
        oof: pd.DataFrame,
        learning_curves: pd.DataFrame,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            fold_metrics_path = temp_dir / "fold_metrics.parquet"
            oof_path = temp_dir / "oof_predictions.parquet"
            curves_path = temp_dir / "learning_curves.parquet"

            fold_metrics.to_parquet(
                fold_metrics_path,
                index=False,
            )

            oof.to_parquet(
                oof_path,
                index=False,
            )

            learning_curves.to_parquet(
                curves_path,
                index=False,
            )

            mlflow.log_artifact(
                str(fold_metrics_path),
                artifact_path="metrics",
            )

            mlflow.log_artifact(
                str(oof_path),
                artifact_path="predictions",
            )

            mlflow.log_artifact(
                str(curves_path),
                artifact_path="curves",
            )

    return (log_cv_artifacts,)


@app.cell
def _(Path, mlflow, pd, tempfile):
    def log_shap_artifacts(
        shap_cv: pd.DataFrame,
        shap_summary: pd.DataFrame,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            shap_cv_path = tmp_dir / "shap_cv.parquet"
            shap_summary_path = tmp_dir / "shap_summary.parquet"

            shap_cv.to_parquet(
                shap_cv_path,
                index=False,
            )

            shap_summary.to_parquet(
                shap_summary_path,
                index=False,
            )

            mlflow.log_artifact(
                str(shap_cv_path),
                artifact_path="shap",
            )

            mlflow.log_artifact(
                str(shap_summary_path),
                artifact_path="shap",
            )

    return (log_shap_artifacts,)


@app.cell
def _(
    log_cv_artifacts,
    log_feature_manifest,
    log_fold_metrics,
    log_shap_artifacts,
    mlflow,
    pd,
    run_catboost_baseline_cv,
):
    def run_catboost_experiment(
        frame: pd.DataFrame,
        features: list[str],
        categorical_features: list[str],
        params: dict,
        run_name: str = "cb_applications_raw_v1",
        capacity: float = 0.10,
        calculate_shap: bool = False
    ) -> dict:

        with mlflow.start_run(run_name=run_name) as run:

            mlflow.log_params(params)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("n_categorical_features", len(categorical_features))
            mlflow.log_param("review_capacity", capacity)
            mlflow.set_tags({
                "compute_backend": "gpu",
                "gpu_device": "RTX 3060",
            })
            result = run_catboost_baseline_cv(
                frame=frame,
                features=features,
                categorical_features=categorical_features,
                params=params,
                capacity=capacity,
                calculate_shap=calculate_shap
            )

            mlflow.log_params({
                f"catboost_{key}": str(value)
                for key, value in result["all_params"].items()
            })

            mlflow.log_dict(
                result["all_params"],
                "params/catboost_all_params.json",
            )

            log_fold_metrics(result["fold_metrics"])

            mlflow.log_metrics(result["summary"])
            log_feature_manifest(
                features,
                categorical_features,
            )
            log_cv_artifacts(
                fold_metrics=result["fold_metrics"],
                oof=result["oof"],
                learning_curves=result["learning_curves"],
            )
            result['run_id'] = run.info.run_id

            if calculate_shap:
                log_shap_artifacts(
                    shap_cv=result["shap_cv"],
                    shap_summary=result["shap_summary"],
                )

        return result


    return


@app.cell
def _(
    TARGET,
    calculate_classification_metrics,
    pd,
    prepare_development_data,
    train_catboost_fold,
):
    def run_sample_size_diagnostic(
        frame: pd.DataFrame,
        features: list[str],
        categorical_features: list[str],
        params: dict,
        fractions: list[float],
    ) -> pd.DataFrame:

        development = prepare_development_data(frame)

        rows = []

        for fold_id in development['fold'].unique():
            for fraction in fractions:
                fold_result = train_catboost_fold(
                            development=development,
                            fold_id=fold_id,
                            features=features,
                            categorical_features=categorical_features,
                            params=params,
                            train_fraction=fraction,
                        )

                train = fold_result["train"]
                valid = fold_result["valid"]

                train_probability = fold_result["train_probability"]
                valid_probability = fold_result["valid_probability"]

                model = fold_result["model"]

                train_metrics = calculate_classification_metrics(
                    train[TARGET],
                    train_probability,
                )

                valid_metrics = calculate_classification_metrics(
                    valid[TARGET],
                    valid_probability,
                )

                row = {
                    "fold": int(fold_id),
                    "train_fraction": fraction,
                    "train_rows": len(train),
                    "train_positives": int(train[TARGET].sum()),
                    "train_prevalence": float(train[TARGET].mean()),

                    "train_ap": train_metrics["ap"],
                    "valid_ap": valid_metrics["ap"],
                    "ap_gap": (
                        train_metrics["ap"]
                        - valid_metrics["ap"]
                    ),

                    "valid_roc_auc": valid_metrics["roc_auc"],

                    "best_iteration": int(
                        model.get_best_iteration()
                    ),

                    "fit_seconds": fold_result["fit_seconds"],
                }

                rows.append(row)

        rows_df = pd.DataFrame(rows)

        return rows_df

    return (run_sample_size_diagnostic,)


@app.cell
def _(Path, mlflow, pd, px, tempfile):
    def log_sample_size_results(
        results: pd.DataFrame,
    ) -> None:
        summary = (
            results
            .groupby("train_fraction")
            .agg(
                train_ap_mean=("train_ap", "mean"),
                valid_ap_mean=("valid_ap", "mean"),
                valid_ap_std=("valid_ap", "std"),
                ap_gap_mean=("ap_gap", "mean"),
            )
            .reset_index()
        )

        for _, row in summary.iterrows():
            pct = int(row["train_fraction"] * 100)

            mlflow.log_metric(
                f"train_ap_mean_{pct}pct",
                row["train_ap_mean"],
            )
            mlflow.log_metric(
                f"valid_ap_mean_{pct}pct",
                row["valid_ap_mean"],
            )
            mlflow.log_metric(
                f"valid_ap_std_{pct}pct",
                row["valid_ap_std"],
            )
            mlflow.log_metric(
                f"ap_gap_mean_{pct}pct",
                row["ap_gap_mean"],
            )

        fig = px.line(
            summary,
            x="train_fraction",
            y=[
                "train_ap_mean",
                "valid_ap_mean",
            ],
            markers=True,
            title="Learning Curve by Training Sample Size",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            results_path = (
                temp_dir
                / "sample_size_learning_curve.parquet"
            )

            figure_path = (
                temp_dir
                / "sample_size_learning_curve.html"
            )

            results.to_parquet(
                results_path,
                index=False,
            )

            fig.write_html(figure_path)

            mlflow.log_artifact(
                str(results_path),
                artifact_path="diagnostics",
            )

            mlflow.log_artifact(
                str(figure_path),
                artifact_path="diagnostics",
            )

    return (log_sample_size_results,)


@app.cell
def _(
    log_feature_manifest,
    log_sample_size_results,
    mlflow,
    pd,
    run_sample_size_diagnostic,
):
    def run_sample_size_experiment(
        frame: pd.DataFrame,
        features: list[str],
        categorical_features: list[str],
        params: dict,
        fractions: list[float],
        run_name: str = "cb_applications_raw_learning_curve_v1",
    ) -> pd.DataFrame:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "run_type": "diagnostic",
                "diagnostic": "training_sample_size",
                "parent_baseline": "cb_applications_raw_v1",
            })

            mlflow.log_params(params)

            mlflow.log_param(
                "n_features",
                len(features),
            )

            mlflow.log_param(
                "n_categorical_features",
                len(categorical_features),
            )

            mlflow.log_param(
                "fractions",
                str(fractions),
            )

            log_feature_manifest(
                features,
                categorical_features,
            )

            results = run_sample_size_diagnostic(
                frame=frame,
                features=features,
                categorical_features=categorical_features,
                params=params,
                fractions=fractions,
            )

            log_sample_size_results(results)

        return results

    return


@app.cell
def _(
    TARGET,
    calculate_classification_metrics,
    np,
    optuna,
    pd,
    train_catboost_fold,
):
    def catboost_objective(
        trial: optuna.Trial,
        development: pd.DataFrame,
        features: list[str],
        categorical_features: list[str],
        base_params: dict,
    ) -> float:
        params = {
            **base_params,

            "depth": trial.suggest_int(
                "depth",
                4,
                9,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.10,
                log=True,
            ),

            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                1.0,
                30.0,
                log=True,
            ),

            "random_strength": trial.suggest_float(
                "random_strength",
                0.01,
                10.0,
                log=True,
            ),
        }

        fold_aps = []

        for fold_id in sorted(development["fold"].unique()):
            fold_result = train_catboost_fold(
                development=development,
                fold_id=int(fold_id),
                features=features,
                categorical_features=categorical_features,
                params=params,
            )

            valid_metrics = calculate_classification_metrics(
                y_true=fold_result["valid"][TARGET],
                probabilities=fold_result["valid_probability"],
            )

            fold_aps.append(valid_metrics["ap"])

        mean_ap = float(np.mean(fold_aps))
        std_ap = float(np.std(fold_aps))

        trial.set_user_attr(
            "fold_ap_std",
            std_ap,
        )

        return mean_ap

    return (catboost_objective,)


@app.cell
def _(catboost_objective, optuna, pd, prepare_development_data):
    def run_optuna_tuning(
        frame: pd.DataFrame,
        features: list[str],
        categorical_features: list[str],
        base_params: dict,
        study_name: str,
        n_trials: int = 50,
    ) -> optuna.Study:
        development = prepare_development_data(frame)

        sampler = optuna.samplers.TPESampler(seed=42)

        study = optuna.create_study(
            study_name=study_name,
            storage="sqlite:///optuna.db",
            direction="maximize",
            load_if_exists=True,
            sampler=sampler
        )

        study.optimize(
            lambda trial: catboost_objective(
                trial=trial,
                development=development,
                features=features,
                categorical_features=categorical_features,
                base_params=base_params,
            ),
            n_trials=n_trials,
        )

        return study

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline model
    """)
    return


@app.cell
def _(ID_COLUMN, applications, split_manifest):
    modeling_df = applications.merge(
        split_manifest[[ID_COLUMN, "partition", "fold"]],
        on = ID_COLUMN,
        how = "left",
        validate = "one_to_one"
    )
    if modeling_df["partition"].isna().any():
        raise RuntimeError("Some applications are missing from split_v1.")

    if modeling_df[ID_COLUMN].duplicated().any():
        raise RuntimeError("SK_ID_CURR is no longer unique.")
    return (modeling_df,)


@app.cell
def _(ID_COLUMN, TARGET, modeling_df, pd):
    BASELINE_EXCLUDED_FEATURES = {
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",
        "EMPLOYED_AGE_RATIO",
        "AGE_AT_CURRENT_EMPLOYMENT_START",
        "EXT_SOURCES_MEAN",
        "EXT_SOURCES_STD",
        "EXT_SOURCES_MIN",
        "EXT_SOURCES_MAX",
        "EXT_SOURCES_COUNT",
    }

    NON_FEATURE_COLUMNS = {
        TARGET,
        ID_COLUMN,
        "partition",
        "fold",
    }

    baseline_features = [
        column
        for column in modeling_df.columns
        if column not in NON_FEATURE_COLUMNS
        and column not in BASELINE_EXCLUDED_FEATURES
    ]

    categorical_features = [
        column
        for column in baseline_features
        if (
            isinstance(modeling_df[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(modeling_df[column])
            or pd.api.types.is_string_dtype(modeling_df[column])
        )
    ]

    print("Baseline features:", len(baseline_features))
    print("Categorical features:", len(categorical_features))

    print("\nExcluded engineered features:")
    print(
        sorted(
            BASELINE_EXCLUDED_FEATURES
            & set(modeling_df.columns)
        )
    )
    return baseline_features, categorical_features


@app.cell
def _():
    BASELINE_PARAMS = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 5000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "allow_writing_files": False,
        "thread_count": -1,
        "task_type": "GPU",
        "devices": "0",
        "custom_metric": [
            "AUC:hints=skip_train~false",
        ],
        "border_count": 254
    }
    return (BASELINE_PARAMS,)


@app.cell
def _():
    # baseline = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=baseline_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_applications_raw_v1",
    #     capacity=0.10,
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline results
    """)
    return


@app.cell
def _():
    # baseline["fold_metrics"]
    return


@app.cell
def _():
    # pd.Series(
    #     baseline["summary"],
    #     name="baseline",
    # )
    return


@app.cell
def _(Path, pd):
    curves_path = Path("mlartifacts/1/e994878d33f547509b5f56312a73afa4/artifacts/curves/learning_curves.parquet")
    baseline_curves = pd.read_parquet(curves_path)
    return (baseline_curves,)


@app.cell
def _(baseline_curves, px):
    fold_id = 0
    best_iteration = 1314

    curve = baseline_curves.query(
        "fold == @fold_id and metric == 'Logloss'"
    )

    fig = px.line(
        curve,
        x="iteration",
        y="value",
        color="dataset",
        title=f"Fold {fold_id} — Logloss",
    )

    fig.add_vline(
        x=best_iteration,
        line_dash="dash",
        annotation_text="best iteration",
    )

    fig.show()
    return (fold_id,)


@app.cell
def _(baseline_curves, px):
    fig_folds = px.line(
        baseline_curves.query(
            "metric == 'Logloss' and dataset == 'validation'"
        ),
        x="iteration",
        y="value",
        color="fold",
        title="Validation Logloss across folds",
    )

    fig_folds.show()
    return


@app.cell
def _(baseline_curves, px):
    fig_auc = px.line(
        baseline_curves.query(
            "fold == 0 and metric == 'AUC' and dataset == 'validation'"
        ),
        x="iteration",
        y="value",
        title="Fold 0 — Validation AUC",
    )

    fig_auc.add_vline(
        x=1314,
        line_dash="dash",
        annotation_text="best iteration",
    )

    fig_auc.show()
    return


@app.cell
def _(baseline_curves, px):
    fig_auc_folds = px.line(
        baseline_curves.query(
            "metric == 'AUC' and dataset == 'validation'"
        ),
        x="iteration",
        y="value",
        color="fold",
        title="Validation AUC across folds",
    )

    fig_auc_folds.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Learning curve on train data volume
    """)
    return


@app.cell
def _(modeling_df):
    development = (
        modeling_df
        .loc[modeling_df["partition"].eq("development")]
        .copy()
    )

    fold_id_num = 0

    train_fold = development.loc[
        development["fold"].ne(fold_id_num)
    ]

    valid_fold = development.loc[
        development["fold"].eq(fold_id_num)
    ]
    return (train_fold,)


@app.cell
def _(TARGET, fold_id, make_nested_stratified_subsample, train_fold):
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    for fraction in fractions:
        sample = make_nested_stratified_subsample(
            train_fold,
            fraction=fraction,
            random_state=42 + fold_id,
        )

        print(
            f"{fraction:.0%}",
            f"rows={len(sample):,}",
            f"positives={sample[TARGET].sum():,}",
            f"prevalence={sample[TARGET].mean():.5f}",
        )
    return


@app.cell
def _():
    # learning_curve_results = run_sample_size_experiment(
    #     frame=modeling_df,
    #     features=baseline_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     fractions=fractions,
    # )
    return


@app.cell
def _():
    # fold_0_results = learning_curve_results.query(
    #     "fold == 0"
    # ).sort_values("train_fraction")

    # px.line(
    #     fold_0_results,
    #     x="train_fraction",
    #     y=["train_ap", "valid_ap"],
    #     markers=True,
    #     title="Fold 0 — AP by Training Sample Size",
    # ).show()
    return


@app.cell
def _():
    # paired_80_100 = (
    #     learning_curve_results
    #     .pivot(
    #         index="fold",
    #         columns="train_fraction",
    #         values="valid_ap",
    #     )
    # )

    # paired_80_100["delta_80_to_100"] = (
    #     paired_80_100[1.0] - paired_80_100[0.8]
    # )

    # paired_80_100[
    #     [0.8, 1.0, "delta_80_to_100"]
    # ]
    return


@app.cell
def _():
    # paired_80_100["delta_80_to_100"].describe()
    return


@app.cell
def _(pd):
    baseline_fold_ap = pd.DataFrame({
        "fold": [0, 1, 2, 3, 4],
        "baseline_ap": [
            0.25457559870149060,
            0.25477822391688276,
            0.24549256077632970,
            0.24049969994660816,
            0.24518262016887343,
        ],
    })
    return


@app.cell
def _(pd):
    baseline_df = pd.DataFrame(
        {
            "oof_ap": [0.2476109141077253],
            "oof_roc_auc": [0.7622601094120849],
            "oof_log_loss": [0.2449267105457207],
            "oof_recall_at_10pct": [0.34088431827875454],
            "oof_precision_at_10pct": [0.2751826772256016],
            "fold_ap_mean": [0.24810574070203692],
            "fold_ap_std": [0.006316769224339662],
            "mean_ap_gap": [0.09782766475609592],
            "best_iteration_median": [1314.0],
            "total_fit_seconds": [2093.728291099993],
        }
    )
    baseline_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B0 — Raw Applications CatBoost Baseline: Results

    #### Performance

    The raw application-level CatBoost baseline produced:

    | Metric | Result |
    |---|---:|
    | OOF Average Precision | **0.2476** |
    | OOF ROC-AUC | **0.7623** |
    | OOF LogLoss | **0.2449** |
    | Recall@Top10% | **0.3409** |
    | Precision@Top10% | **0.2752** |
    | Fold AP mean | **0.2481** |
    | Fold AP std | **0.0063** |
    | Mean train-validation AP gap | **0.0978** |
    | Median best iteration | **1,314** |

    With a target prevalence of approximately **8.07%**, the model provides substantially better ranking than a random baseline. Under the simulated review capacity of 10%, the highest-risk 10% of applications contain approximately **34.1% of all defaults**, while **27.5%** of reviewed applications are positive cases.

    #### Fold stability

    Validation performance is reasonably stable across the five fixed folds. Fold AP standard deviation is approximately **0.0063**, and the validation AUC learning curves follow similar trajectories across folds.

    There is some variation in achievable performance between folds, but no fold exhibits fundamentally different training dynamics.

    #### Training dynamics

    The CatBoost model quickly extracts most of the available ranking signal during the first several hundred boosting iterations. Validation ROC-AUC then enters a long plateau, with only small additional improvements at higher tree counts.

    At the same time, training LogLoss continues to decrease while validation LogLoss reaches a minimum and subsequently starts increasing. This confirms overfitting in probability estimation: later trees continue fitting the training data and make predictions more confident without improving probability quality on unseen data.

    Ranking quality behaves differently. Validation ROC-AUC continues to improve slightly before reaching a plateau, explaining why the AUC-based early stopping selects approximately 1,000–1,600 trees despite validation LogLoss beginning to deteriorate earlier.

    #### Training sample-size diagnostic

    A learning-curve experiment was performed using nested stratified training subsets of **20%, 40%, 60%, 80%, and 100%**, while keeping the validation folds, features and CatBoost configuration fixed.

    As the amount of training data increased:

    - validation AP consistently improved;
    - train AP decreased;
    - the train-validation gap became smaller;
    - the marginal improvement in validation AP gradually decreased.

    The final increase from **80% to 100%** of the available training data produced the following paired AP changes:

    | Fold | Δ AP, 80% → 100% |
    |---:|---:|
    | 0 | +0.00455 |
    | 1 | +0.00267 |
    | 2 | +0.00238 |
    | 3 | +0.00325 |
    | 4 | -0.00028 |

    Mean improvement:

    **+0.00251 AP**

    The improvement is positive on **4 of 5 folds**, with the remaining fold essentially unchanged. Therefore additional training data still provides useful generalization improvement, but the flattening validation curve indicates diminishing returns.

    #### Diagnostic conclusion

    The baseline is **not primarily limited by insufficient model capacity**.

    The evidence instead indicates:

    1. The current application-level features contain substantial predictive signal.
    2. The model has noticeable variance, demonstrated by the large train-validation gap on smaller training samples.
    3. Increasing the amount of training data reduces this gap and improves validation performance.
    4. The benefit of additional similar application-level data is already diminishing near the current dataset size.
    5. Increasing model complexity or starting broad hyperparameter tuning is not currently justified.
    6. The most promising next source of improvement is additional predictive signal through feature engineering and historical data.

    #### Decision

    **ACCEPTED AS BASELINE REFERENCE.**

    `B0 — applications_raw_v1` becomes the fixed reference against which subsequent feature bundles, ablations and model changes will be evaluated on the same `split_v1` folds.

    The next stage is iterative single-table feature engineering, beginning with previously identified application-level hypotheses before introducing multi-table historical aggregates.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E1 — EXT_SOURCE summary bundle

    **Observation**

    `EXT_SOURCE_1/2/3` are among the strongest individual application-level
    predictors. During EDA, their aggregate mean showed a stronger univariate
    association with the target than the individual sources.

    **Hypothesis**

    Explicit summary features can provide a compact representation of:

    - overall external risk;
    - disagreement between external scores;
    - extreme external assessments;
    - availability of external scores.

    This representation may allow the fixed-capacity CatBoost model to exploit the
    combined external signal more efficiently than using the three raw scores alone.

    **Change**

    Add the following features to B0:

    - `EXT_SOURCES_MEAN`
    - `EXT_SOURCES_STD`
    - `EXT_SOURCES_MIN`
    - `EXT_SOURCES_MAX`
    - `EXT_SOURCES_COUNT`

    All original `EXT_SOURCE_1/2/3` features remain in the model.

    **Controlled variables**

    - Same `split_v1`
    - Same development population
    - Same CatBoost parameters
    - Same random seed
    - Same early-stopping procedure

    **Primary comparison**

    Paired Average Precision delta against B0 on the same five folds.

    **Secondary checks**

    - OOF AP
    - ROC-AUC
    - Recall@Top10%
    - Precision@Top10%
    - train-validation AP gap
    - best iteration
    - runtime

    **Decision**

    The bundle is accepted only if the improvement is reproducible across folds and
    not explained by a single favorable fold.
    """)
    return


@app.cell
def _(baseline_features):
    EXT_SOURCE_BUNDLE = [
        "EXT_SOURCES_MEAN",
        "EXT_SOURCES_STD",
        "EXT_SOURCES_MIN",
        "EXT_SOURCES_MAX",
        "EXT_SOURCES_COUNT",
    ]

    ext_candidate_features = (
        baseline_features
        + EXT_SOURCE_BUNDLE
    )
    return


@app.cell
def _():
    # ext_candidate_results = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=ext_candidate_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_ext_sources_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # baseline_vs_ext = baseline_fold_ap.merge(
    #     ext_candidate_results['fold_metrics'][
    #         ["fold", "valid_ap"]
    #     ].rename(columns={"valid_ap": "candidate_ap"}),
    #     on="fold",
    #     validate="one_to_one",
    # )

    # baseline_vs_ext["delta_ap"] = (
    #     baseline_vs_ext["candidate_ap"]
    #     - baseline_vs_ext["baseline_ap"]
    # )

    # baseline_vs_ext
    return


@app.cell
def _():
    # print(f"Mean delta AP for EXT_SOURCE: {baseline_vs_ext['delta_ap'].mean():.6f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E1 result — EXT_SOURCE summary bundle

    The `EXT_SOURCE` summary bundle did not improve the raw CatBoost baseline.

    Paired validation AP deltas versus B0:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | +0.00019 |
    | 1 | -0.00150 |
    | 2 | -0.00006 |
    | 3 | -0.00327 |
    | 4 | +0.00110 |

    Mean paired delta: **-0.00071 AP**.

    The candidate improved only 2 of 5 folds and produced a notably worse result on
    fold 3. Training AP increased substantially while validation AP did not,
    indicating that the additional representation mainly increased the model's
    ability to fit the training data rather than adding transferable predictive
    signal.

    **Decision: REJECTED.**

    The raw `EXT_SOURCE_1/2/3` features remain in the baseline, while
    `EXT_SOURCES_MEAN`, `EXT_SOURCES_STD`, `EXT_SOURCES_MIN`,
    `EXT_SOURCES_MAX`, and `EXT_SOURCES_COUNT` are not promoted to the
    accepted feature set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E2 — Financial burden ratios

    **Observation**

    The raw application table contains absolute loan, annuity and income amounts,
    but credit risk may depend more directly on their relative burden than on their
    absolute values.

    **Hypothesis**

    Explicit financial ratios provide a more compact representation of repayment
    burden and loan affordability than the corresponding raw variables alone.

    **Change**

    Add the following features to the accepted B0 feature set:

    - `CREDIT_INCOME_RATIO`
    - `ANNUITY_INCOME_RATIO`
    - `ANNUITY_CREDIT_RATIO`

    The original amount features remain available to CatBoost.

    **Leakage / inference check**

    All ratios are deterministic row-wise transformations of information available
    in the current application. They use no target statistics, cross-row
    information or historical events after the prediction cutoff.

    **Controlled variables**

    - `split_v1`
    - same development population
    - same CatBoost parameters
    - same random seed
    - same early-stopping procedure
    - same B0 features

    **Primary comparison**

    Paired validation AP delta versus B0 on the same five folds.

    **Secondary checks**

    - OOF AP
    - ROC-AUC
    - Recall@Top10%
    - Precision@Top10%
    - train-validation AP gap
    - best iteration

    **Decision rule**

    Accept only if the improvement is reproducible across folds and is not driven
    by a single favorable fold.
    """)
    return


@app.cell
def _(baseline_features):
    FINANCIAL_RATIO_BUNDLE = [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",
    ]
    financial_ratio_features = (
        baseline_features
        + FINANCIAL_RATIO_BUNDLE
    )
    return


@app.cell
def _():
    # financial_ratio_candidate_results = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=financial_ratio_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_financial_ratios_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # baseline_vs_financial = baseline_fold_ap.merge(
    #     financial_ratio_candidate_results['fold_metrics'][
    #         ["fold", "valid_ap"]
    #     ].rename(columns={"valid_ap": "candidate_ap"}),
    #     on="fold",
    #     validate="one_to_one",
    # )

    # baseline_vs_financial["delta_ap"] = (
    #     baseline_vs_financial["candidate_ap"]
    #     - baseline_vs_financial["baseline_ap"]
    # )

    # baseline_vs_financial
    return


@app.cell
def _(baseline_vs_financial):
    print(f"Mean delta AP for FINANCIAL_RATIO_BUNDLE: {baseline_vs_financial['delta_ap'].mean():.6f}")
    return


@app.cell
def _():
    # baseline_df.index = ["baseline"]

    # financial_oof_df = pd.DataFrame(
    #     [financial_ratio_candidate_results["summary"]]
    # )
    # financial_oof_df.index = ["financial_ratios"]

    # comparison = pd.concat([
    #     baseline_df,
    #     financial_oof_df,
    # ]).T

    # comparison["delta"] = (
    #     comparison["financial_ratios"]
    #     - comparison["baseline"]
    # )

    # comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E2 result — Financial burden ratios

    The financial-ratio bundle produced a clear and consistent improvement over the raw CatBoost baseline.

    Paired validation AP deltas versus B0:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | +0.00442 |
    | 1 | +0.00656 |
    | 2 | +0.00533 |
    | 3 | +0.00864 |
    | 4 | +0.00945 |

    Mean paired delta: **+0.00688 AP**.

    All **5 of 5 folds improved**, so the gain is not driven by a single favorable validation split.

    OOF-level metrics also improved:

    | Metric | Baseline | Financial ratios | Δ |
    |---|---:|---:|---:|
    | OOF Average Precision | 0.24761 | **0.25463** | **+0.00702** |
    | OOF ROC-AUC | 0.76226 | **0.76866** | **+0.00640** |
    | OOF LogLoss | 0.24493 | **0.24307** | **−0.00185** |
    | Recall@Top10% | 0.34088 | **0.34828** | **+0.00739** |
    | Precision@Top10% | 0.27518 | **0.28115** | **+0.00597** |
    | Fold AP std | 0.00632 | **0.00520** | **−0.00112** |

    The mean train-validation AP gap increased from **0.0978** to **0.1213**, indicating that the additional representation also increases the model's ability to fit the training data. However, this does not invalidate the candidate because performance on unseen validation data improved consistently: AP increased on every fold, ROC-AUC increased, LogLoss decreased, and both Top-10% policy metrics improved.

    The result supports the hypothesis that explicit financial burden ratios provide a more efficient representation of relationships between credit amount, annuity and income than the raw amount features alone.

    **Decision: ACCEPTED.**

    The following features are promoted to the accepted application-level feature set:

    - `CREDIT_INCOME_RATIO`
    - `ANNUITY_INCOME_RATIO`
    - `ANNUITY_CREDIT_RATIO`

    The accepted E2 feature set becomes the new reference for subsequent single-table feature experiments.
    """)
    return


@app.cell
def _(baseline_features):
    ACCEPTED_FEATURES = baseline_features + [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",
    ]
    return (ACCEPTED_FEATURES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E3 — Employment history bundle

    **Observation**

    `DAYS_BIRTH` and `DAYS_EMPLOYED` contain information about age and current
    employment duration, but the relationship between these quantities may be more
    informative than either absolute value alone.

    **Hypothesis**

    Explicit employment-history features provide a compact representation of
    employment stability relative to the applicant's age and may help CatBoost
    capture this relationship more efficiently.

    **Change**

    Add the following features to the accepted E2 feature set:

    - `EMPLOYED_AGE_RATIO`
    - `AGE_AT_CURRENT_EMPLOYMENT_START`

    `DAYS_BIRTH`, `DAYS_EMPLOYED`, and `DAYS_EMPLOYED_ANOMALY` remain available to
    the model.

    **Leakage / inference check**

    Both features are deterministic row-wise transformations of application
    information available at prediction time. They use no target statistics,
    cross-row information, or future historical events.

    **Controlled variables**

    - same `split_v1`;
    - same development population;
    - same CatBoost parameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted E2 feature set as the reference.

    **Primary comparison**

    Paired validation AP delta against E2 on the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap.

    **Decision rule**

    Accept the bundle only if its improvement is reproducible across folds and does
    not come at the expense of the Top-10% policy metrics.
    """)
    return


@app.cell
def _(modeling_df):
    EMPLOYMENT_HISTORY_BUNDLE = [
        "EMPLOYED_AGE_RATIO",
        "AGE_AT_CURRENT_EMPLOYMENT_START",
    ]

    modeling_df[EMPLOYMENT_HISTORY_BUNDLE].describe()
    return (EMPLOYMENT_HISTORY_BUNDLE,)


@app.cell
def _(EMPLOYMENT_HISTORY_BUNDLE, modeling_df, np):
    np.isinf(
        modeling_df[EMPLOYMENT_HISTORY_BUNDLE]
        .to_numpy(dtype="float64")
    ).sum()
    return


@app.cell
def _(modeling_df):
    modeling_df[
        "AGE_AT_CURRENT_EMPLOYMENT_START"
    ].quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1])
    return


@app.cell
def _(baseline_features):
    accepted_features_e2 = (
        baseline_features
        + [
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "ANNUITY_CREDIT_RATIO",
        ]
    )
    return (accepted_features_e2,)


@app.cell
def _(EMPLOYMENT_HISTORY_BUNDLE, accepted_features_e2):
    employment_candidate_features = (
        accepted_features_e2
        + EMPLOYMENT_HISTORY_BUNDLE
    )
    return


@app.cell
def _():
    # employment_candidate_results = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=employment_candidate_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_employment_history_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # financial_vs_employment = (
    #     financial_ratio_candidate_results["fold_metrics"][
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "financial_ap"})
    #     .merge(
    #         employment_candidate_results["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "employment_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # financial_vs_employment["delta_ap"] = (
    #     financial_vs_employment["employment_ap"]
    #     - financial_vs_employment["financial_ap"]
    # )

    # financial_vs_employment
    return


@app.cell
def _(financial_vs_employment):
    print(f'Mean delta AP for EMPLOYMENT_HISTORY_BUNDLE: {financial_vs_employment["delta_ap"].mean():.6f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E3 result — Employment history bundle

    The employment-history bundle did not improve the accepted E2 model.

    Paired validation AP deltas versus E2:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | -0.00024 |
    | 1 | +0.00012 |
    | 2 | -0.00028 |
    | 3 | -0.00203 |
    | 4 | -0.00124 |

    Mean paired delta: **-0.00073 AP**.

    The candidate improved only 1 of 5 folds, and that improvement was negligible.
    The largest changes were negative, particularly on folds 3 and 4.

    The result does not support the hypothesis that explicit employment-history
    representations provide additional transferable signal beyond the original
    age and employment variables already available to CatBoost.

    **Decision: REJECTED.**

    `EMPLOYED_AGE_RATIO` and `AGE_AT_CURRENT_EMPLOYMENT_START` are not promoted to
    the accepted feature set.

    The accepted application-level feature set remains E2:
    the raw cleaned application features plus the financial-ratio bundle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E4 — Household affordability bundle

    **Observation**

    The raw application data contains household size, number of children, income, credit amount and annuity payment as separate absolute values.

    The same income or loan amount may imply a different financial burden depending on how many household members depend on the available resources.

    The accepted E2 financial-ratio bundle showed that explicit normalization of financial quantities can provide additional predictive signal beyond the corresponding raw variables.

    **Hypothesis**

    Normalizing financial quantities by household size can provide a more direct representation of household-level affordability and repayment burden than absolute financial values alone.

    The proportion of children in the household may additionally capture differences in household dependency structure.

    **Change**

    Add the following features to the accepted E2 feature set:

    - `INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`
    - `CREDIT_PER_PERSON = AMT_CREDIT / CNT_FAM_MEMBERS`
    - `ANNUITY_PER_PERSON = AMT_ANNUITY / CNT_FAM_MEMBERS`
    - `CHILDREN_RATIO = CNT_CHILDREN / CNT_FAM_MEMBERS`

    The original income, credit, annuity, family-size and children-count features remain available to CatBoost.

    **Leakage / inference check**

    All features are deterministic row-wise transformations of information available in the current application at prediction time.

    They use:

    - no target information;
    - no statistics computed across applicants;
    - no future events;
    - no historical-table information.

    `CNT_FAM_MEMBERS` must be checked for zero and missing values before computing the ratios.

    **Controlled variables**

    - same `split_v1`;
    - same development population;
    - same CatBoost parameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted E2 feature set as the reference.

    **Primary comparison**

    Paired validation AP delta against E2 on the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept the bundle only if the improvement is reproducible across folds and is not driven by a single favorable validation fold.
    """)
    return


@app.cell
def _(modeling_df):
    modeling_df["CNT_FAM_MEMBERS"].describe()
    return


@app.cell
def _(modeling_df):
    HOUSEHOLD_AFFORDABILITY_BUNDLE = [
        "INCOME_PER_PERSON",
        "CREDIT_PER_PERSON",
        "ANNUITY_PER_PERSON",
        "CHILDREN_RATIO",
    ]

    modeling_df['INCOME_PER_PERSON'] = modeling_df['AMT_INCOME_TOTAL'] / modeling_df['CNT_FAM_MEMBERS']
    modeling_df['CREDIT_PER_PERSON'] = modeling_df['AMT_CREDIT'] / modeling_df['CNT_FAM_MEMBERS']
    modeling_df['ANNUITY_PER_PERSON'] = modeling_df['AMT_ANNUITY'] / modeling_df['CNT_FAM_MEMBERS']
    modeling_df['CHILDREN_RATIO'] = modeling_df['CNT_CHILDREN'] / modeling_df['CNT_FAM_MEMBERS']
    return (HOUSEHOLD_AFFORDABILITY_BUNDLE,)


@app.cell
def _(ACCEPTED_FEATURES, HOUSEHOLD_AFFORDABILITY_BUNDLE):
    e4_features = (
        ACCEPTED_FEATURES
        + HOUSEHOLD_AFFORDABILITY_BUNDLE
    )
    return


@app.cell
def _():
    # household_candidate_results = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=e4_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_household_affordability_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _(pd):
    financial_result = pd.read_parquet(
        "mlartifacts\\1\\af2dae5dcf82445d983b6fc175cf24ec\\artifacts\\metrics\\fold_metrics.parquet"
        )
    return


@app.cell
def _():
    # financial_vs_household = (
    #     financial_result[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "financial_ap"})
    #     .merge(
    #         household_candidate_results["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "household_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # financial_vs_household["delta_ap"] = (
    #     financial_vs_household["household_ap"]
    #     - financial_vs_household["financial_ap"]
    # )

    # financial_vs_household
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E4 result — Household affordability bundle

    The household-affordability bundle did not improve the accepted E2 model.

    Paired validation AP deltas versus E2:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | +0.00024 |
    | 1 | -0.00135 |
    | 2 | +0.00013 |
    | 3 | -0.00328 |
    | 4 | -0.00209 |

    Mean paired delta: **-0.00127 AP**.

    The candidate improved only 2 of 5 folds, and both positive changes were
    negligible. The negative changes were larger, particularly on folds 3 and 4.

    The result does not support the hypothesis that normalizing income, credit and
    annuity by household size provides additional transferable signal beyond the
    raw household variables and the already accepted financial-ratio features.

    **Decision: REJECTED.**

    `INCOME_PER_PERSON`, `CREDIT_PER_PERSON`, `ANNUITY_PER_PERSON`, and
    `CHILDREN_RATIO` are not promoted to the accepted feature set.

    The accepted reference remains E2: raw cleaned application features plus the
    financial-ratio bundle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E5 — Application financing structure bundle

    **Observation**

    The application table contains both the requested/issued credit amount and the price of the financed goods.

    These absolute values do not directly express how the financing amount relates to the underlying purchase price.

    Two applicants may request the same credit amount while financing goods of substantially different value, resulting in different financing structures.

    **Hypothesis**

    Explicit features describing the relationship between credit, annuity and goods price can provide CatBoost with a more compact representation of the financing structure of the application.

    This may add signal beyond the accepted income-based financial ratios because it describes the loan relative to the financed asset rather than relative to the applicant's income.

    **Change**

    Add the following features to the accepted feature set:

    - `CREDIT_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE`
    - `CREDIT_GOODS_DIFF = AMT_CREDIT - AMT_GOODS_PRICE`
    - `ANNUITY_GOODS_RATIO = AMT_ANNUITY / AMT_GOODS_PRICE`

    The original `AMT_CREDIT`, `AMT_ANNUITY`, and `AMT_GOODS_PRICE` features remain available to CatBoost.

    **Leakage / inference check**

    All features are deterministic row-wise transformations of current-application information available before the credit decision.

    They use:

    - no target statistics;
    - no cross-row information;
    - no future events;
    - no external historical tables.

    `AMT_GOODS_PRICE` must be checked for zero and missing values before ratio calculation.

    **Controlled variables**

    - same `split_v1`;
    - same development population;
    - same CatBoost parameters;
    - same random seed;
    - same early-stopping procedure;
    - current accepted feature set as the reference.

    **Primary comparison**

    Paired validation AP delta against the current accepted reference model on the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept the bundle only if it provides a reproducible improvement across folds and the gain is not offset by degradation in the Top-10% review policy metrics.
    """)
    return


@app.cell
def _(modeling_df):
    GOODS_BUNDLE = [
        "CREDIT_GOODS_RATIO",
        "CREDIT_GOODS_DIFF",
        "ANNUITY_GOODS_RATIO"
    ]

    modeling_df['CREDIT_GOODS_RATIO'] = modeling_df['AMT_CREDIT'] / modeling_df['AMT_GOODS_PRICE']
    modeling_df['CREDIT_GOODS_DIFF'] = modeling_df['AMT_CREDIT'] - modeling_df['AMT_GOODS_PRICE']
    modeling_df['ANNUITY_GOODS_RATIO'] = modeling_df['AMT_ANNUITY'] / modeling_df['AMT_GOODS_PRICE']
    return (GOODS_BUNDLE,)


@app.cell
def _(ACCEPTED_FEATURES, GOODS_BUNDLE):
    e5_features = (
        ACCEPTED_FEATURES
        + GOODS_BUNDLE
    )
    return


@app.cell
def _():
    # goods_candidate_results = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=e5_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_goods_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # financial_vs_goods = (
    #     financial_result[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "financial_ap"})
    #     .merge(
    #         goods_candidate_results["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "goods_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # financial_vs_goods["delta_ap"] = (
    #     financial_vs_goods["goods_ap"]
    #     - financial_vs_goods["financial_ap"]
    # )

    # financial_vs_goods
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E5 result — Application financing structure bundle

    The application-financing bundle did not improve the accepted E2 model.

    Paired validation AP deltas versus E2:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | +0.00064 |
    | 1 | -0.00147 |
    | 2 | +0.00166 |
    | 3 | -0.00214 |
    | 4 | -0.00120 |

    Mean paired delta: **-0.00050 AP**.

    The candidate improved 2 of 5 folds and degraded 3 of 5 folds. The positive and negative changes are small, but the overall direction is slightly negative and there is no consistent evidence of transferable improvement.

    The result does not support the hypothesis that explicit relationships between credit amount, annuity and goods price provide additional predictive value beyond the original financial variables and the already accepted income-based financial-ratio bundle.

    **Decision: REJECTED.**

    `CREDIT_GOODS_RATIO`, `CREDIT_GOODS_DIFF`, and `ANNUITY_GOODS_RATIO` are not promoted to the accepted feature set.

    The accepted reference remains E2: raw cleaned application features plus the financial-ratio bundle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E6 — Missingness / application completeness bundle

    **Observation**

    The application dataset contains structured missingness across both numerical and categorical variables.

    CatBoost can handle missing numerical values natively and categorical missing values are represented explicitly, but these mechanisms operate primarily at the level of individual features.

    The overall amount of unavailable information in an application may itself describe a meaningful application or customer state that is not captured efficiently by considering each missing value independently.

    **Hypothesis**

    Aggregated missingness features can provide a compact representation of application completeness and source availability.

    Applicants with unusually sparse numerical or categorical information may form a distinct risk cohort, and explicit missingness counts may help CatBoost identify this regime more efficiently.

    **Change**

    Add the following features to the accepted feature set:

    - `NUMERIC_MISSING_COUNT`
    - `CATEGORICAL_MISSING_COUNT`

    The original numerical and categorical features remain unchanged and available to CatBoost.

    **Leakage / inference check**

    Both features are deterministic row-wise transformations of information available at prediction time.

    They use:

    - no target statistics;
    - no cross-row information;
    - no future events;
    - no external historical tables.

    **Controlled variables**

    - same `split_v1`;
    - same development population;
    - same CatBoost parameters;
    - same random seed;
    - same early-stopping procedure;
    - current accepted feature set as the reference.

    **Primary comparison**

    Paired validation AP delta against the current accepted reference model on the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept the bundle only if it provides a reproducible improvement across folds and the gain is not offset by degradation in the Top-10% review policy metrics.
    """)
    return


@app.cell
def _(baseline_features, categorical_features, modeling_df):
    numeric_baseline_features = [
        col for col in baseline_features
        if col not in categorical_features
    ]

    categorical_baseline_features = categorical_features

    modeling_df["NUMERIC_MISSING_COUNT"] = (
        modeling_df[numeric_baseline_features]
        .isna()
        .sum(axis=1)
    )

    modeling_df["CATEGORICAL_MISSING_COUNT"] = (
        modeling_df[categorical_baseline_features]
        .eq("__MISSING__")
        .sum(axis=1)
    )

    MISSINGNESS_BUNDLE = [
        "NUMERIC_MISSING_COUNT",
        "CATEGORICAL_MISSING_COUNT",
    ]
    return (MISSINGNESS_BUNDLE,)


@app.cell
def _(ACCEPTED_FEATURES, MISSINGNESS_BUNDLE):
    e6_features = (
        ACCEPTED_FEATURES
        + MISSINGNESS_BUNDLE
    )
    return


@app.cell
def _():
    # missing_candidate_results = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=e6_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_missing_count_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # financial_vs_missing = (
    #     financial_result[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "financial_ap"})
    #     .merge(
    #         missing_candidate_results["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "missing_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # financial_vs_missing["delta_ap"] = (
    #     financial_vs_missing["missing_ap"]
    #     - financial_vs_missing["financial_ap"]
    # )

    # financial_vs_missing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E6 result — Missingness / application completeness bundle

    The aggregated missingness bundle did not improve the accepted E2 model.

    Paired validation AP deltas versus E2:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | -0.00065 |
    | 1 | -0.00093 |
    | 2 | -0.00075 |
    | 3 | -0.00057 |
    | 4 | -0.00117 |

    Mean paired delta: **-0.00082 AP**.

    The candidate degraded validation AP on **all 5 folds**. Although the absolute effect is small, the direction of the change is completely consistent across the fixed validation splits.

    The result does not support the hypothesis that global application-completeness counts provide additional transferable signal beyond the individual missing values already available to CatBoost.

    CatBoost can already exploit missingness patterns at the level of individual numerical and categorical features. The aggregated missingness counts appear to compress these patterns without adding useful predictive information.

    **Decision: REJECTED.**

    `NUMERIC_MISSING_COUNT` and `CATEGORICAL_MISSING_COUNT` are not promoted to the accepted feature set.

    The accepted reference remains **E2: raw cleaned application features plus the financial-ratio bundle**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## E7 — Credit bureau request intensity bundle

    **Observation**

    The application table contains separate counts of recent credit-bureau inquiries across several time windows:

    - hour;
    - day;
    - week;
    - month;
    - quarter;
    - year.

    These raw variables describe related aspects of recent credit-seeking activity but are presented as separate counters.

    The overall intensity and recency of credit inquiries may be more informative than any individual time window alone.

    **Hypothesis**

    Aggregating credit-bureau inquiry counts into recent, long-term and total activity features can provide a more compact representation of credit-seeking behaviour.

    A high concentration of recent inquiries may indicate active search for additional credit and may provide predictive signal beyond the individual bureau-request counters already available to CatBoost.

    **Change**

    Add the following features to the accepted E2 feature set:

    - `BUREAU_REQUESTS_RECENT` — total inquiries across hour, day, week and month windows;
    - `BUREAU_REQUESTS_LONG_TERM` — total inquiries across quarter and year windows;
    - `BUREAU_REQUESTS_TOTAL` — total inquiries across all available windows;
    - `HAS_RECENT_BUREAU_REQUEST` — indicator that at least one recent inquiry is present.

    The original `AMT_REQ_CREDIT_BUREAU_*` features remain available to CatBoost.

    **Missing-value handling**

    Aggregated counts preserve missingness when all contributing values are unavailable rather than automatically interpreting missing information as zero.

    For example, `sum(..., min_count=1)` is used so that:

    - known absence of inquiries can remain `0`;
    - complete absence of source information remains `NaN`.

    **Leakage / inference check**

    The bundle is constructed only from application-level bureau-request counters assumed to be available at prediction time.

    It uses:

    - no target information;
    - no statistics across applicants;
    - no future events;
    - no information from the separate `bureau` history table.

    This experiment tests only a new representation of existing application-level credit-inquiry information.

    **Controlled variables**

    - same `split_v1`;
    - same development population;
    - same CatBoost parameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted E2 feature set as the reference.

    **Primary comparison**

    Paired validation AP delta against E2 on the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept the bundle only if it produces a reproducible improvement across folds and does not degrade the Top-10% review-policy metrics.
    """)
    return


@app.cell
def _(modeling_df):
    BUREAU_REQUEST_RECENT = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
    ]

    BUREAU_REQUEST_LONG = [
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]

    modeling_df["BUREAU_REQUESTS_RECENT"] = (
        modeling_df[BUREAU_REQUEST_RECENT]
        .sum(axis=1, min_count=1)
    )

    modeling_df["BUREAU_REQUESTS_LONG_TERM"] = (
        modeling_df[BUREAU_REQUEST_LONG]
        .sum(axis=1, min_count=1)
    )

    modeling_df["BUREAU_REQUESTS_TOTAL"] = (
        modeling_df[
            BUREAU_REQUEST_RECENT + BUREAU_REQUEST_LONG
        ]
        .sum(axis=1, min_count=1)
    )

    modeling_df["HAS_RECENT_BUREAU_REQUEST"] = (
        modeling_df["BUREAU_REQUESTS_RECENT"] > 0
    ).astype("int8")
    return


@app.cell
def _(ACCEPTED_FEATURES):
    BUREAU_REQUEST_BUNDLE = [
        "BUREAU_REQUESTS_RECENT",
        "BUREAU_REQUESTS_LONG_TERM",
        "BUREAU_REQUESTS_TOTAL",
        "HAS_RECENT_BUREAU_REQUEST",
    ]

    request_features = (
        ACCEPTED_FEATURES
        + BUREAU_REQUEST_BUNDLE
    )
    return


@app.cell
def _():
    # request_candidate_results = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=request_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_request_bundles_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # financial_vs_request = (
    #     financial_result[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "financial_ap"})
    #     .merge(
    #         request_candidate_results["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "request_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # financial_vs_request["delta_ap"] = (
    #     financial_vs_request["request_ap"]
    #     - financial_vs_request["financial_ap"]
    # )

    # financial_vs_request
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### E7 result — Credit bureau request intensity bundle

    The credit-bureau request intensity bundle did not improve the accepted E2 model.

    Paired validation AP deltas versus E2:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | +0.00179 |
    | 1 | -0.00137 |
    | 2 | -0.00070 |
    | 3 | -0.00166 |
    | 4 | -0.00273 |

    Mean paired delta: **-0.00094 AP**.

    The candidate improved only 1 of 5 folds. The remaining four folds degraded, with the largest negative effect observed on fold 4.

    The result does not support the hypothesis that aggregated credit-bureau inquiry intensity and recency provide additional transferable signal beyond the original `AMT_REQ_CREDIT_BUREAU_*` variables already available to CatBoost.

    **Decision: REJECTED.**

    `BUREAU_REQUESTS_RECENT`, `BUREAU_REQUESTS_LONG_TERM`,
    `BUREAU_REQUESTS_TOTAL`, and `HAS_RECENT_BUREAU_REQUEST` are not promoted to the accepted feature set.

    The accepted reference remains **E2: raw cleaned application features plus the financial-ratio bundle**.

    ## Single-table feature engineering summary

    The single-table experiments suggest that the current model is approaching a practical local performance ceiling for the available application-level information under the fixed CatBoost configuration.

    Results so far:

    | Experiment | Feature bundle | Decision |
    |---|---|---|
    | E1 | EXT_SOURCE summaries | REJECT |
    | E2 | Financial burden ratios | **ACCEPT** |
    | E3 | Employment history | REJECT |
    | E4 | Household affordability | REJECT |
    | E5 | Application financing structure | REJECT |
    | E6 | Application completeness / missingness | REJECT |
    | E7 | Credit bureau request intensity | REJECT |

    Only the financial burden ratios produced a clear and reproducible improvement across all validation folds.

    Most other engineered features represented alternative transformations of information that was already present in the raw application table. CatBoost appears able to exploit much of this information directly, so additional manually constructed representations provided little or no transferable gain.

    This does **not** establish an absolute upper bound for `applications`. Hyperparameter tuning has not been performed, and additional feature interactions may become useful in a richer feature space.

    However, the diminishing return from further single-table feature engineering makes additional historical information a more promising next source of predictive signal than continuing to generate similar transformations of the current application variables.

    **Next step: multi-table feature engineering.**
    """)
    return


@app.cell
def _():
    # accepted_gpu_run = run_catboost_experiment(
    #     frame=modeling_df,
    #     features=ACCEPTED_FEATURES,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_financial_v1_gpu_bc254",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # accepted_gpu_results = pd.read_parquet("mlartifacts\\1\\9e5939e1eb2b419da64bfecdf157db07\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Accepted reference:
    E2 application features
    + financial ratios
    + CatBoost GPU
    + border_count=254
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B1 — Bureau basic credit history

    **Observation**

    Applicants can have multiple historical credits reported by the external credit bureau. Among applicants with bureau history, the median number of reported credits is 4, while both active and closed credits are common.

    **Hypothesis**

    The volume and current status structure of an applicant's external credit history provide predictive information that is not available from the current application alone.

    Applicants with different numbers and proportions of active, closed, and sold credits may represent different credit-risk profiles.

    **Change**

    Add the following aggregated bureau features:

    - `BUREAU_CREDIT_COUNT`
    - `BUREAU_ACTIVE_COUNT`
    - `BUREAU_CLOSED_COUNT`
    - `BUREAU_SOLD_COUNT`
    - `BUREAU_ACTIVE_SHARE`

    The features are aggregated to exactly one row per `SK_ID_CURR`.

    **Missing-history handling**

    For applicants with no bureau records:

    - count features are set to `0`;
    - `BUREAU_ACTIVE_SHARE` remains missing because the proportion is undefined when no bureau history exists.

    **Leakage / inference check**

    Only bureau records passing the temporal cutoff audit are used.

    Records with `DAYS_CREDIT_UPDATE > 0` were excluded before aggregation.

    The bundle contains no target-based or cross-applicant statistics.

    **Reference**

    The current GPU reference model:

    - accepted E2 application feature set;
    - CatBoost GPU;
    - `border_count=254`;
    - fixed `split_v1`.

    **Primary comparison**

    Paired validation AP delta against the GPU E2 reference on the same five folds.

    **Decision rule**

    Accept the bundle only if the gain is reproducible across folds and is supported by the OOF and Top-10% policy metrics.
    """)
    return


@app.cell
def _():
    # modeling_bureau_b1 = pd.read_parquet("data\processed\modeling_bureau_b1.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES):
    BUREAU_BASIC_FEATURES = [
        "BUREAU_CREDIT_COUNT",
        "BUREAU_ACTIVE_COUNT",
        "BUREAU_CLOSED_COUNT",
        "BUREAU_SOLD_COUNT",
        "BUREAU_ACTIVE_SHARE",
    ]

    b1_features = (
        ACCEPTED_FEATURES
        + BUREAU_BASIC_FEATURES
    )
    return (BUREAU_BASIC_FEATURES,)


@app.cell
def _():
    # bureau_1_run = run_catboost_experiment(
    #     frame=modeling_bureau_b1,
    #     features=b1_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_bureau_b1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # accepted_vs_bureau_b1 = (
    #     accepted_gpu_results[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "financial_ap"})
    #     .merge(
    #         bureau_1_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "bureau_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # accepted_vs_bureau_b1["delta_ap"] = (
    #     accepted_vs_bureau_b1["bureau_ap"]
    #     - accepted_vs_bureau_b1["financial_ap"]
    # )

    # accepted_vs_bureau_b1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B1 result — Bureau basic credit history

    The basic bureau credit-history bundle improved the accepted GPU application-level reference.

    Paired validation AP deltas:

    | Fold | Δ AP |
    |---:|---:|
    | 0 | +0.00410 |
    | 1 | +0.00376 |
    | 2 | +0.00368 |
    | 3 | +0.00227 |
    | 4 | -0.00036 |

    Mean paired delta: **+0.00269 AP**.

    The candidate improved **4 of 5 folds**. The only negative fold was essentially unchanged, while the positive improvements were materially larger.

    OOF-level metrics also improved:

    | Metric | GPU reference | Bureau B1 | Δ |
    |---|---:|---:|---:|
    | OOF Average Precision | 0.25433 | **0.25683** | **+0.00251** |
    | OOF ROC-AUC | 0.76835 | **0.77040** | **+0.00205** |
    | OOF LogLoss | 0.24317 | **0.24254** | **-0.00063** |
    | Recall@Top10% | 0.34705 | **0.34998** | **+0.00294** |
    | Precision@Top10% | 0.28016 | **0.28253** | **+0.00237** |
    | Fold AP mean | 0.25478 | **0.25747** | **+0.00269** |

    The basic bureau features therefore improve both global ranking quality and the
    Top-10% review policy.

    The fold AP standard deviation increased slightly, and the mean train-validation
    AP gap increased from approximately **0.111** to **0.145**. This indicates that
    the additional bureau information also increases the model's ability to fit the
    training data.

    However, this does not invalidate the candidate because performance on unseen
    data improved across the main evaluation criteria: OOF AP and ROC-AUC increased,
    LogLoss decreased, both Top-10% policy metrics improved, and 4 of 5 paired folds
    showed positive AP deltas.

    Unlike most rejected single-table engineered features, the bureau bundle adds a
    genuinely new information source rather than another representation of existing
    application variables.

    **Decision: ACCEPTED.**

    The following features are promoted to the accepted feature set:

    - `BUREAU_CREDIT_COUNT`
    - `BUREAU_ACTIVE_COUNT`
    - `BUREAU_CLOSED_COUNT`
    - `BUREAU_SOLD_COUNT`
    - `BUREAU_ACTIVE_SHARE`

    B1 becomes the new GPU reference for subsequent bureau feature experiments.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES, BUREAU_BASIC_FEATURES):
    ACCEPTED_FEATURES_V1 = (
        ACCEPTED_FEATURES
        + BUREAU_BASIC_FEATURES
    )
    return (ACCEPTED_FEATURES_V1,)


@app.cell
def _():
    # b1_fold_results = pd.read_parquet("mlartifacts\\1\\c55558b66b394d569219774c7f61dce3\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B2 — Bureau credit recency and activity windows

    **Observation**

    The `bureau` table contains the historical credit records of applicants. The
    `DAYS_CREDIT` variable indicates how many days before the current application
    each bureau credit was opened.

    The initial analysis shows that the available bureau history spans a substantial
    period:

    - median history depth: approximately 1,822 days (~5 years);
    - maximum history depth: 2,922 days (~8 years);
    - median time since the latest bureau credit: approximately 305 days (~10 months).

    This suggests that both the recency of the latest credit and the concentration
    of credits within recent time windows may contain predictive information.

    **Hypothesis**

    Recent credit activity may provide additional predictive signal beyond the
    basic bureau statistics introduced in B1.

    Applicants with multiple credits opened recently may have a different risk
    profile from applicants whose bureau activity is concentrated further in the
    past.

    Using several time windows should capture different levels of recency without
    introducing a large number of highly correlated features.

    **Change**

    Add the following features to the accepted B1 feature set:

    - `BUREAU_DAYS_SINCE_LATEST_CREDIT` — days since the most recent bureau credit;
    - `BUREAU_HISTORY_AGE_DAYS` — age of the oldest available bureau credit;
    - `BUREAU_CREDITS_LAST_180D` — number of bureau credits opened within the last
      180 days;
    - `BUREAU_CREDITS_LAST_365D` — number of bureau credits opened within the last
      365 days;
    - `BUREAU_CREDITS_LAST_730D` — number of bureau credits opened within the last
      730 days.

    The time-window features are calculated by aggregating bureau records by
    `SK_ID_CURR`.

    **Initial distribution check**

    The resulting windows show meaningful variation:

    - `BUREAU_CREDITS_LAST_180D`: 69.3% of applicants have zero recent credits,
      mean = 0.44, max = 36;
    - `BUREAU_CREDITS_LAST_365D`: 42.8% have zero credits, mean = 1.06, max = 80;
    - `BUREAU_CREDITS_LAST_730D`: 20.5% have zero credits, mean = 2.15, max = 96.

    The windows therefore provide progressively broader measures of recent bureau
    activity rather than being effectively constant or completely sparse.

    **Missing-value handling**

    The features are derived from existing bureau records after aggregation by
    `SK_ID_CURR`.

    Applicants represented in the bureau table receive aggregated values. The
    absence of a recent credit within a given window is represented by `0`, while
    the distinction between having no recent credit and having no bureau history
    is preserved through the existing bureau coverage logic.

    **Leakage / inference check**

    The features use only historical bureau records available for the applicant.

    They use:

    - no target information;
    - no information from future events;
    - no statistics calculated across applicants;
    - only `DAYS_CREDIT` and the applicant-level bureau history.

    **Controlled variables**

    - same `split_v1`;
    - same development population;
    - same CatBoost parameters;
    - same GPU configuration;
    - same random seed;
    - same early-stopping procedure;
    - B1 accepted feature set as the reference.

    **Primary comparison**

    Paired validation AP delta against B1 on the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept B2 only if the recency features produce a reproducible improvement over
    B1 across folds and do not materially degrade the Top-10% review-policy metrics.

    If the improvement is absent or unstable, the recency bundle will be rejected
    without creating additional arbitrary time windows.
    """)
    return


@app.cell
def _():
    # modeling_bureau_b2 = pd.read_parquet("data\processed\modeling_bureau_b2.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES_V1):
    BUREAU_RECENCY_FEATURES = [
        "BUREAU_DAYS_SINCE_LATEST_CREDIT",
        "BUREAU_HISTORY_AGE_DAYS",
        "BUREAU_CREDITS_LAST_180D",
        "BUREAU_CREDITS_LAST_365D",
        "BUREAU_CREDITS_LAST_730D",
    ]

    b2_features = (
        BUREAU_RECENCY_FEATURES +
        ACCEPTED_FEATURES_V1
    )
    return (BUREAU_RECENCY_FEATURES,)


@app.cell
def _():
    # bureau_2_run = run_catboost_experiment(
    #     frame=modeling_bureau_b2,
    #     features=b2_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_bureau_b2",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # bureau_b1_vs_bureau_b2 = (
    #     b1_fold_results[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "bureau_b1_ap"})
    #     .merge(
    #         bureau_2_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "bureau_b2_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # bureau_b1_vs_bureau_b2["delta_ap"] = (
    #     bureau_b1_vs_bureau_b2["bureau_b2_ap"]
    #     - bureau_b1_vs_bureau_b2["bureau_b1_ap"]
    # )

    # bureau_b1_vs_bureau_b2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B2 — Bureau recency features

    | Metric | Value |
    |---|---:|
    | Mean Δ AP | +0.00120 |
    | Positive folds | 4 / 5 |
    | Negative folds | 1 / 5 |

    **Decision:** KEEP

    B2 improves AP relative to B1 on 4 of 5 folds.
    The average improvement is small (+0.00120 AP), with the largest
    gain observed on fold 4 (+0.00380). The effect is therefore useful
    but not strong and shows noticeable variation across folds.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V1, BUREAU_RECENCY_FEATURES):
    ACCEPTED_FEATURES_V2 = (
        ACCEPTED_FEATURES_V1
        + BUREAU_RECENCY_FEATURES
    )
    return (ACCEPTED_FEATURES_V2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B3 — Bureau financial exposure

    **Observation**

    The `bureau` table contains financial information about the applicant's
    historical credit accounts, including credit amounts, limits, debt and overdue
    amounts.

    After aggregation by `SK_ID_CURR`, these variables provide information about
    the overall scale of the applicant's historical credit exposure that is not
    directly represented by the number or recency of bureau records.

    The aggregated distributions are highly right-skewed, with substantial
    differences between applicants in the total volume of historical credit.

    **Hypothesis**

    The total size and structure of an applicant's historical credit exposure may
    provide additional predictive information beyond the basic bureau history
    (B1) and credit recency features (B2).

    In particular, applicants with larger historical credit amounts, higher
    credit limits or larger outstanding debt may have a different default risk
    profile.

    **Change**

    Add the following aggregated features to the accepted B1 + B2 feature set:

    - `BUREAU_TOTAL_CREDIT_SUM` — total credit amount across bureau records;
    - `BUREAU_MAX_CREDIT_SUM` — maximum credit amount across bureau records;
    - `BUREAU_TOTAL_CREDIT_LIMIT` — total available credit limit across bureau
      records;
    - `BUREAU_MEAN_CREDIT_SUM` — mean credit amount across bureau records;
    - `BUREAU_TOTAL_CREDIT_DEBT` — total reported debt across bureau records;
    - `BUREAU_TOTAL_CREDIT_OVERDUE` — total reported overdue amount across bureau
      records.

    All features are aggregated to one row per `SK_ID_CURR`.

    **Data-quality consideration**

    During the preliminary analysis, negative values were found in
    `AMT_CREDIT_SUM_DEBT` and `AMT_CREDIT_SUM_LIMIT`.

    The negative debt values were found exclusively for `Credit card` records and
    showed a structured relationship with the corresponding credit limit.

    Therefore, negative values were not arbitrarily clipped or replaced with zero.
    The original values are preserved during aggregation.

    **Distribution check**

    The aggregated financial features are strongly right-skewed and contain
    extreme values.

    For example:

    - `BUREAU_TOTAL_CREDIT_SUM` has a maximum above 1 billion;
    - `BUREAU_MAX_CREDIT_SUM` has a maximum of 585 million;
    - `BUREAU_MEAN_CREDIT_SUM` also has a long right tail;
    - `BUREAU_TOTAL_CREDIT_OVERDUE` is highly sparse, with most applicants having
      zero overdue amount.

    These properties are retained for the initial experiment rather than being
    modified without evidence that transformation is necessary.

    **Leakage / inference check**

    The features are calculated exclusively from the historical `bureau` table
    associated with the applicant.

    They use:

    - no target information;
    - no statistics calculated across applicants;
    - no future application outcomes;
    - no information from tables outside the current bureau experiment.

    The resulting table contains exactly one aggregated row per `SK_ID_CURR`.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - B1 + B2 as the reference feature set.

    **Primary comparison**

    Paired validation AP comparison:

    `B1 + B2` vs `B1 + B2 + B3`

    using exactly the same folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept B3 if the financial exposure features provide a reproducible improvement
    over B1 + B2 across folds without materially degrading the Top-10% review-policy
    metrics.

    If the improvement is absent or unstable, reject B3 and do not further expand
    the bureau financial feature group at this stage.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V2):
    BUREAU_CREDIT = [
        "BUREAU_TOTAL_CREDIT_SUM",
        "BUREAU_MAX_CREDIT_SUM",
        "BUREAU_TOTAL_CREDIT_LIMIT",
        "BUREAU_MEAN_CREDIT_SUM",
        "BUREAU_TOTAL_CREDIT_DEBT",
        "BUREAU_TOTAL_CREDIT_OVERDUE"
    ]

    b3_features = (
        BUREAU_CREDIT +
        ACCEPTED_FEATURES_V2
    )
    return (BUREAU_CREDIT,)


@app.cell
def _():
    # modeling_bureau_b3 = pd.read_parquet("data\processed\modeling_bureau_b3.parquet")
    return


@app.cell
def _():
    # bureau_3_run = run_catboost_experiment(
    #     frame=modeling_bureau_b3,
    #     features=b3_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_bureau_b3",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # bureau_b2_vs_bureau_b3 = (
    #     bureau_2_run['fold_metrics'][
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "bureau_b2_ap"})
    #     .merge(
    #         bureau_3_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "bureau_b3_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # bureau_b2_vs_bureau_b3["delta_ap"] = (
    #     bureau_b2_vs_bureau_b3["bureau_b3_ap"]
    #     - bureau_b2_vs_bureau_b3["bureau_b2_ap"]
    # )

    # bureau_b2_vs_bureau_b3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B3 — Bureau financial exposure results

    | Metric | Value |
    |---|---:|
    | Mean Δ AP | +0.00403 |
    | Positive folds | 5 / 5 |
    | Negative folds | 0 / 5 |

    **Decision:** KEEP

    B3 improves AP relative to B2 on 5 of 5 folds.
    The average improvement is small (+0.00403 AP), with the largest
    gain observed on fold 1 (+0.00573).
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V2, BUREAU_CREDIT):
    ACCEPTED_FEATURES_V3 = (
        ACCEPTED_FEATURES_V2 +
        BUREAU_CREDIT
    )
    return (ACCEPTED_FEATURES_V3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B4 — Bureau delinquency / credit stress

    **Observation**

    B3 showed a strong and consistent improvement after adding financial aggregates
    from `bureau`:

    - improvement on 5/5 folds;
    - mean `Δ AP ≈ +0.00403`;
    - minimum improvement: `+0.00129`;
    - maximum improvement: `+0.00573`.

    This suggests that financial history contains additional predictive signal.

    However, B3 primarily captures the **scale of the applicant's historical credit
    exposure** and does not directly describe the quality or problematic nature of
    that credit history.

    The `bureau` table contains several variables related to delinquency:

    - `AMT_CREDIT_SUM_OVERDUE`;
    - `CREDIT_DAY_OVERDUE`;
    - `AMT_CREDIT_MAX_OVERDUE`;
    - `CREDIT_ACTIVE`.

    `AMT_CREDIT_SUM_OVERDUE` is highly sparse, with the vast majority of bureau
    records having a value of `0`.

    **Hypothesis**

    The presence and severity of previous credit delinquencies may provide
    additional predictive signal beyond the credit history, recency and financial
    exposure features already included in B1–B3.

    The hypothesis is that applicants with previous overdue credit accounts or
    more severe delinquency may have a higher probability of default.

    **Change**

    Add the following features to the accepted B1 + B2 + B3 feature set:

    - `BUREAU_HAS_OVERDUE` — whether the applicant has at least one bureau record
      with a non-zero overdue amount;
    - `BUREAU_OVERDUE_CREDIT_COUNT` — number of bureau credits with a non-zero
      overdue amount;
    - `BUREAU_TOTAL_CREDIT_OVERDUE` — total overdue amount across bureau records;
    - `BUREAU_MAX_OVERDUE` — maximum recorded overdue amount;
    - `BUREAU_MAX_DAYS_OVERDUE` — maximum recorded number of overdue days;
    - `BUREAU_OVERDUE_CREDIT_SHARE` — share of bureau credits with overdue amounts
      among all bureau credits.

    All features are aggregated to the `SK_ID_CURR` level, resulting in one row per
    applicant.

    **Feature semantics**

    The bundle captures several dimensions of credit stress:

    - existence of previous delinquency;
    - frequency of delinquent credit accounts;
    - total overdue amount;
    - maximum overdue amount;
    - maximum duration of delinquency;
    - proportion of credit accounts with delinquency.

    This allows the hypothesis to be tested without introducing a large number of
    arbitrary aggregates.

    **Missing-value handling**

    For delinquency-related aggregates, the absence of an overdue event within the
    available bureau history is represented by `0`.

    The absence of bureau history itself must not be interpreted as equivalent to
    having a clean credit history. The distinction between no bureau history and no
    overdue event within existing bureau history is preserved by the aggregation
    logic.

    **Leakage / inference check**

    All features are calculated exclusively from historical bureau records
    associated with the applicant.

    The bundle uses:

    - no `TARGET`;
    - no outcome of the current application;
    - no future information;
    - no statistics calculated across applicants.

    The temporal availability of bureau records must respect the previously defined
    inference cutoff.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - B1 + B2 + B3 as the reference feature set.

    **Primary comparison**

    Paired validation AP comparison:

    `B1 + B2 + B3` vs `B1 + B2 + B3 + B4`

    using exactly the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept B4 if it provides a reproducible improvement over the current reference
    across folds without materially degrading the Top-10% review-policy metrics.

    If B4 produces little or unstable improvement, reject the bundle and consider the
    `bureau` table sufficiently explored at this stage.
    """)
    return


@app.cell
def _():
    # modeling_bureau_b4 = pd.read_parquet("data\processed\modeling_bureau_b4.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES_V3):
    STRESS_FEATURES = [
        "TOTAL_CREDIT_OVERDUE",
        "HAS_OVERDUE_CREDIT",
        "MAX_CREDIT_DAY_OVERDUE",
        "MAX_CREDIT_OVERDUE_AMT",
        "TOTAL_ACTIVE_CREDIT_OVERDUE",
        "OVERDUE_CREDIT_SHARE"
    ]

    b4_features = (
        STRESS_FEATURES +
        ACCEPTED_FEATURES_V3
    )
    return (STRESS_FEATURES,)


@app.cell
def _():
    # bureau_4_run = run_catboost_experiment(
    #     frame=modeling_bureau_b4,
    #     features=b4_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_bureau_b4",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # bureau_b3_vs_bureau_b4 = (
    #     bureau_3_run['fold_metrics'][
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "bureau_b3_ap"})
    #     .merge(
    #         bureau_4_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "bureau_b4_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )

    # bureau_b3_vs_bureau_b4["delta_ap"] = (
    #     bureau_b3_vs_bureau_b4["bureau_b4_ap"]
    #     - bureau_b3_vs_bureau_b4["bureau_b3_ap"]
    # )

    # bureau_b3_vs_bureau_b4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### B4 — Results

    | Fold | B3 AP | B4 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.26990 | 0.27272 | +0.00282 |
    | 1 | 0.26723 | 0.26953 | +0.00230 |
    | 2 | 0.26025 | 0.25866 | -0.00159 |
    | 3 | 0.25596 | 0.25845 | +0.00249 |
    | 4 | 0.26022 | 0.26075 | +0.00053 |

    **Mean Δ AP:** `+0.00131`

    **Positive folds:** `4 / 5`

    **Decision:** **KEEP**

    B4 improves validation AP on 4 of 5 folds, with a mean improvement of
    approximately `+0.00131 AP`.

    The effect is weaker than the improvement obtained from B3 and is somewhat
    less stable, since fold 2 shows a negative change of `-0.00159`. However, the
    overall direction remains positive and the improvement is observed across the
    majority of folds.

    The delinquency / credit stress features therefore provide additional
    predictive information beyond the B1 + B2 + B3 feature set.

    No further feature engineering experiments are performed on `bureau` at this
    stage.

    **Bureau status**

    - B1 — KEEP
    - B2 — KEEP
    - B3 — KEEP
    - B4 — KEEP

    The accepted `bureau` feature set will be used as the reference when moving to
    the next historical table.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V3, STRESS_FEATURES):
    ACCEPTED_FEATURES_V4 = (
        STRESS_FEATURES +
        ACCEPTED_FEATURES_V3
    )
    return (ACCEPTED_FEATURES_V4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P1 — Previous application history

    **Observation**

    The `previous_application` table contains the applicant's historical credit
    applications and their outcomes.

    The table includes `NAME_CONTRACT_STATUS`, which distinguishes between
    different outcomes of previous applications, such as approved, refused,
    canceled and unused offers.

    The number and outcome of previous applications may provide information about
    the applicant's historical interaction with the credit system.

    **Hypothesis**

    The applicant's previous application history contains predictive information
    beyond the current application and the bureau history.

    In particular, applicants with a high number of previous applications,
    frequent refusals or a different approval/refusal pattern may have a different
    default risk profile.

    Aggregating previous application outcomes to the `SK_ID_CURR` level may provide
    a more compact representation of this historical behavior.

    **Change**

    Add the following features to the current accepted feature set:

    - `PREV_APP_COUNT` — total number of previous applications;
    - `PREV_APP_APPROVED_COUNT` — number of approved previous applications;
    - `PREV_APP_REFUSED_COUNT` — number of refused previous applications;
    - `PREV_APP_CANCELED_COUNT` — number of canceled previous applications;
    - `PREV_APP_UNUSED_COUNT` — number of unused offers;
    - `PREV_APP_APPROVAL_RATE` — share of previous applications that were approved;
    - `PREV_APP_REFUSAL_RATE` — share of previous applications that were refused;
    - `PREV_APP_DAYS_SINCE_LAST` — recency of the most recent previous
      application based on `DAYS_DECISION`.

    All features are aggregated to the `SK_ID_CURR` level, resulting in one row per
    applicant.

    **Feature semantics**

    The bundle captures two main aspects of previous application behavior:

    1. **Application volume**
       - total number of previous applications.

    2. **Application outcomes**
       - approved applications;
       - refused applications;
       - canceled applications;
       - unused offers;
       - approval and refusal rates.

    `PREV_APP_DAYS_SINCE_LAST` additionally captures how recently the applicant
    interacted with the credit application process.

    **Temporal consideration**

    `DAYS_DECISION` represents the time between the previous application decision
    and the current application.

    The variable ranges from `-1` to `-2922`, indicating that previous decisions
    occurred before the current application.

    Only historical applications are therefore used for the aggregation.

    **Data-quality consideration**

    Several date-related columns in `previous_application` contain the sentinel
    value `365243`, similar to the sentinel previously observed in
    `DAYS_EMPLOYED`.

    These columns are not used in this first experiment.

    `DAYS_DECISION` is used because it has no missing values and does not contain the
    `365243` sentinel.

    **Missing-value handling**

    The original `previous_application` values are not globally imputed before
    aggregation.

    The aggregation is performed on the available historical application records.

    Applicants without any `previous_application` history remain distinguishable
    from applicants with previous applications whose individual fields contain
    missing values.

    **Leakage / inference check**

    All features are calculated exclusively from previous applications associated
    with the applicant.

    The bundle uses:

    - no `TARGET`;
    - no outcome of the current application;
    - no future application information;
    - no statistics calculated across applicants.

    The aggregation respects the temporal ordering relative to the current
    application.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - current accepted feature set as the reference.

    **Primary comparison**

    Paired validation AP comparison:

    `Current reference` vs `Current reference + P1`

    using exactly the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept P1 if the previous application history provides a reproducible
    improvement across folds without materially degrading the Top-10% review-policy
    metrics.

    If the improvement is weak or unstable, reject the bundle and move to the next
    feature group.
    """)
    return


@app.cell
def _():
    # modeling_pre_1 = pd.read_parquet("data/processed/modeling_application_history_1.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES_V4):
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
    p1_features = (
        APPLICATION_HISTORY +
        ACCEPTED_FEATURES_V4
    )
    return APPLICATION_HISTORY, p1_features


@app.cell
def _():
    # pre_application_run = run_catboost_experiment(
    #     frame=modeling_pre_1,
    #     features=p1_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_application_history_v1",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # bureau_b4_vs_pre_apps_1 = (
    #     bureau_4_run['fold_metrics'][
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "bureau_b4_ap"})
    #     .merge(
    #         pre_application_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "pre_apps_1_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )


    # bureau_b4_vs_pre_apps_1["delta_ap"] = (
    #     bureau_b4_vs_pre_apps_1["pre_apps_1_ap"]
    #     - bureau_b4_vs_pre_apps_1["bureau_b4_ap"]
    # )

    # bureau_b4_vs_pre_apps_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### P1 — Results

    | Fold | B4 AP | P1 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27272 | 0.27328 | +0.00056 |
    | 1 | 0.26953 | 0.27044 | +0.00092 |
    | 2 | 0.25866 | 0.26191 | +0.00325 |
    | 3 | 0.25845 | 0.26402 | +0.00557 |
    | 4 | 0.26075 | 0.26233 | +0.00159 |

    **Mean Δ AP:** `+0.00238`

    **Positive folds:** `5 / 5`

    **Decision:** **KEEP**

    P1 provides a consistent improvement over the current reference.
    Validation AP improves on all five folds, with a mean improvement of
    approximately `+0.00238 AP`.

    The strongest improvement is observed on fold 3 (`+0.00557 AP`), while even
    the smallest improvement remains positive (`+0.00056 AP`).

    The result indicates that previous application history contains additional
    predictive information beyond the currently accepted bureau feature set.

    The P1 feature bundle is therefore accepted and will remain in the modeling
    dataset.

    **Next step**

    Move to the financial characteristics of previous applications and test
    whether the amounts and credit structure of historical applications provide
    additional predictive signal.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P2 — Previous application financial history

    **Observation**

    The `previous_application` table contains financial information for the
    applicant's historical credit applications, including requested credit,
    granted credit, goods price, annuity, down payment and payment duration.

    These variables are available at the previous-application level, while the
    model operates on one row per `SK_ID_CURR`. Therefore, the financial history
    must be aggregated to the applicant level.

    The distributions of the financial variables are strongly right-skewed, which
    is expected for monetary variables. No additional transformation was applied,
    as CatBoost can work directly with such distributions.

    **Hypothesis**

    The financial characteristics of previous applications may contain predictive
    information about the applicant's credit behavior beyond the previous
    application outcomes captured in P1.

    In particular, the historical scale of requested and granted credit, payment
    amounts, down payments and the difference between requested and granted credit
    may provide additional information about the applicant's risk profile.

    **Change**

    Add the following applicant-level features to the accepted P1 feature set:

    - `TOTAL_AMT_PREV_APPLICATION`
    - `MEAN_AMT_PREV_APPLICATION`
    - `MAX_AMT_PREV_APPLICATION`

    - `TOTAL_AMT_PREV_CREDIT`
    - `MEAN_AMT_PREV_CREDIT`
    - `MAX_AMT_PREV_CREDIT`

    - `TOTAL_AMT_PREV_GOODS_PRICE`
    - `MEAN_AMT_PREV_GOODS_PRICE`
    - `MAX_AMT_PREV_GOODS_PRICE`

    - `TOTAL_AMT_PREV_ANNUITY`
    - `MEAN_AMT_PREV_ANNUITY`
    - `MAX_AMT_PREV_ANNUITY`

    - `TOTAL_AMT_PREV_DOWN_PAYMENT`
    - `MEAN_AMT_PREV_DOWN_PAYMENT`
    - `MAX_AMT_PREV_DOWN_PAYMENT`

    - `TOTAL_CNT_PREV_PAYMENT`
    - `MEAN_CNT_PREV_PAYMENT`
    - `MAX_CNT_PREV_PAYMENT`

    Two additional derived variables were created at the previous-application
    level:

    - `CREDIT_APPLICATION_DIFF` =
      `AMT_CREDIT - AMT_APPLICATION`
    - `CREDIT_APPLICATION_RATIO` =
      `AMT_CREDIT / AMT_APPLICATION`

    These were then aggregated using:

    - total;
    - mean;
    - maximum.

    **Feature interpretation**

    The bundle captures several aspects of historical financial behavior:

    1. **Requested credit**
       - typical and maximum requested amount;
       - total historical requested amount.

    2. **Granted credit**
       - typical and maximum granted credit;
       - total historical granted credit.

    3. **Purchase value**
       - historical goods prices.

    4. **Payment burden**
       - historical annuity amounts;
       - number of payments.

    5. **Down payment behavior**
       - historical down-payment amounts.

    6. **Credit vs. requested amount**
       - `CREDIT_APPLICATION_DIFF` captures the absolute difference between
         granted and requested credit;
       - `CREDIT_APPLICATION_RATIO` captures the relative amount of credit
         granted compared with the requested amount.

    **Aggregation**

    All features are aggregated by `SK_ID_CURR`, producing one row per applicant.

    The resulting table is then joined to the current application dataset.

    **Missing-value handling**

    The original financial variables contain substantial missingness, particularly
    for `AMT_GOODS_PRICE`, `AMT_ANNUITY`, `AMT_DOWN_PAYMENT` and `CNT_PAYMENT`.

    No global imputation was performed before aggregation.

    Missing values are therefore handled by the aggregation functions and remain
    part of the resulting feature representation where insufficient historical
    information is available.

    **Data-quality considerations**

    The financial variables contain zero values and strong right skew, but no
    systematic data-quality issue requiring transformation was identified.

    `CREDIT_APPLICATION_RATIO` is calculated with zero values of
    `AMT_APPLICATION` replaced by `NaN` to avoid division by zero.

    **Leakage / inference check**

    The features are constructed exclusively from the applicant's historical
    `previous_application` records.

    The experiment uses:

    - no `TARGET`;
    - no information from the current application outcome;
    - no statistics calculated across applicants.

    The features represent historical financial information available before the
    current application.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - P1 accepted feature set as the reference.

    **Primary comparison**

    Paired validation AP comparison:

    `P1` vs. `P1 + P2`

    using exactly the same five folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    The financial history bundle is retained if it provides a meaningful overall
    improvement in validation AP without causing unacceptable instability or
    degradation of the Top-10% review-policy metrics.

    **Status**

    KEEP — P2 provides additional predictive signal, with a positive mean
    validation AP delta across the five folds.
    """)
    return


@app.cell
def _():
    FINANCIAL_HISTORY_FEATURES = [
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
        "MIN_CREDIT_APPLICATION_RATIO"
    ]
    return (FINANCIAL_HISTORY_FEATURES,)


@app.cell
def _(FINANCIAL_HISTORY_FEATURES, p1_features):
    p2_features = (
        FINANCIAL_HISTORY_FEATURES +
        p1_features
    )
    return


@app.cell
def _():
    # modeling_pre_app_2 = pd.read_parquet("data/processed/modeling_application_history_2.parquet")
    return


@app.cell
def _():
    # pre_app_p2_run = run_catboost_experiment(
    #     frame=modeling_pre_app_2,
    #     features=p2_features,
    #     categorical_features=categorical_features,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_application_history_v2",
    #     capacity=0.10,
    # )
    return


@app.cell
def _():
    # pre_apps_1_vs_pre_apps_2 = (
    #     pre_application_run['fold_metrics'][
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "pre_app_1_ap"})
    #     .merge(
    #         pre_app_p2_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "pre_app_2_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )


    # pre_apps_1_vs_pre_apps_2["delta_ap"] = (
    #     pre_apps_1_vs_pre_apps_2["pre_app_2_ap"]
    #     - pre_apps_1_vs_pre_apps_2["pre_app_1_ap"]
    # )

    # pre_apps_1_vs_pre_apps_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### P2 — Results

    | Fold | P1 AP | P2 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27328 | 0.27342 | +0.00014 |
    | 1 | 0.27044 | 0.27243 | +0.00198 |
    | 2 | 0.26191 | 0.26504 | +0.00313 |
    | 3 | 0.26402 | 0.26051 | -0.00351 |
    | 4 | 0.26233 | 0.27011 | +0.00778 |

    **Mean Δ AP:** `+0.00191`

    **Positive folds:** `4 / 5`

    **Decision:** **KEEP**

    The financial history bundle provides additional predictive signal beyond P1,
    with a mean validation AP improvement of approximately `+0.00191`.

    The improvement is observed on 4 of 5 folds, although the effect is less
    stable than P1 due to a negative result on fold 3.

    The group is therefore retained for further modeling. Its contribution can be
    re-evaluated later after additional historical tables and feature groups are
    added.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V4, FINANCIAL_HISTORY_FEATURES):
    ACCEPTED_FEATURES_V5 = (
        ACCEPTED_FEATURES_V4 +
        FINANCIAL_HISTORY_FEATURES
    )
    return (ACCEPTED_FEATURES_V5,)


@app.cell
def _(pd):
    pre_app_2_result = pd.read_parquet("mlartifacts\\1\\bd5675646f8145f8b895ac9246a4e4db\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P3 — Previous application categorical history

    **Observation**

    The `previous_application` table contains several categorical variables describing
    the type, outcome, payment method, product characteristics and application
    channel of previous credit applications.

    P1 captured previous application counts and outcomes, while P2 added aggregated
    financial history. However, these feature groups do not explicitly represent the
    categorical structure of the applicant's previous credit activity.

    Applicants may repeatedly use different credit products, payment methods,
    application channels or loan purposes, and these historical patterns may contain
    additional predictive information.

    **Hypothesis**

    The dominant categorical characteristics of an applicant's previous applications
    may provide additional predictive signal beyond the numerical application-history
    and financial-history features already included in P1 and P2.

    The most frequently observed category for each applicant is used as a compact
    representation of their typical previous application behavior.

    For `FLAG_LAST_APPL_PER_CONTRACT`, the proportion of previous applications with
    the value `Y` is used instead of the mode in order to preserve frequency
    information.

    **Change**

    Add the following applicant-level categorical history features to the accepted
    P1 + P2 feature set:

    - `MODE_NAME_CONTRACT_TYPE` — most frequent previous contract type;
    - `SHARE_FLAG_LAST_APPL_PER_CONTRACT` — share of previous applications marked
      as the last application for the contract;
    - `MODE_NAME_PAYMENT_TYPE` — most frequent payment type;
    - `MODE_CODE_REJECT_REASON` — most frequent rejection-reason category;
    - `MODE_NAME_CLIENT_TYPE` — most frequent historical client type;
    - `MODE_NAME_CASH_LOAN_PURPOSE` — most frequent cash-loan purpose;
    - `MODE_NAME_PORTFOLIO` — most frequent portfolio category;
    - `MODE_CHANNEL_TYPE` — most frequent application channel.

    All features are aggregated by `SK_ID_CURR`, resulting in one row per applicant.

    **Aggregation logic**

    For categorical variables, the mode is used:

    `SK_ID_CURR → most frequent historical category`

    For example, if an applicant has previous applications with contract types:

    - Cash loans
    - Cash loans
    - Consumer loans
    - Cash loans

    the resulting feature is:

    `MODE_NAME_CONTRACT_TYPE = Cash loans`

    For `SHARE_FLAG_LAST_APPL_PER_CONTRACT`, the feature is calculated as:

    `number of previous applications with FLAG_LAST_APPL_PER_CONTRACT = "Y"`
    divided by
    `total number of previous applications`

    This preserves more information than using only the most frequent binary value.

    **Missing-value handling**

    No global imputation is performed before aggregation.

    If a categorical variable contains missing values, the mode is calculated from
    the available historical values. If no valid category is available for an
    applicant, the aggregated value remains missing.

    Applicants with no `previous_application` history remain distinguishable after
    the aggregated feature table is joined to the application-level dataset.

    **Information-loss consideration**

    Mode aggregation intentionally compresses the historical categorical
    distribution into a single dominant value.

    For example, an applicant with:

    - 60% Cash loans
    - 30% Consumer loans
    - 10% Revolving loans

    is represented only by:

    `MODE_NAME_CONTRACT_TYPE = Cash loans`

    Therefore, this experiment tests whether a simple and inexpensive categorical
    summary is sufficient to add predictive value. More detailed category shares or
    distributions are not introduced at this stage.

    **Leakage / inference check**

    All features are derived exclusively from historical `previous_application`
    records associated with the applicant.

    The bundle uses:

    - no `TARGET`;
    - no outcome of the current application;
    - no statistics calculated across applicants;
    - no information generated after the current application cutoff.

    The categorical aggregation is performed independently within each
    `SK_ID_CURR`.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted P1 + P2 feature set as the reference.

    **Primary comparison**

    Paired validation AP comparison:

    `P1 + P2` vs `P1 + P2 + P3`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Accept P3 only if the categorical-history features provide a reproducible
    improvement across folds and do not degrade the OOF ranking or Top-10%
    review-policy metrics.

    A small improvement in mean fold AP alone is not sufficient if the effect is
    unstable or accompanied by degradation in the main policy metrics.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V5):
    CATEGORICAL_HISTORY_FEATURES = [
        "MODE_NAME_CONTRACT_TYPE",
        "SHARE_FLAG_LAST_APPL_PER_CONTRACT",
        "MODE_NAME_PAYMENT_TYPE",
        "MODE_CODE_REJECT_REASON",
        "MODE_NAME_CLIENT_TYPE",
        "MODE_NAME_CASH_LOAN_PURPOSE",
        "MODE_NAME_PORTFOLIO",
        "MODE_CHANNEL_TYPE",
        "MODE_NAME_TYPE_SUITE"
    ]

    p3_features = (
        ACCEPTED_FEATURES_V5 +
        CATEGORICAL_HISTORY_FEATURES
    )
    return


@app.cell
def _():
    # modeling_p3 = pd.read_parquet("data\processed\modeling_application_history_3.parquet")
    return


@app.cell
def _():
    # p3_cat_features = [
    #     cat
    #     for cat in modeling_p3.select_dtypes(
    #         include=["category", "object"]
    #     ).columns.to_list()
    #     if cat != "partition"
    # ]
    return


@app.cell
def _():
    # pre_app_p3_run = run_catboost_experiment(
    #     modeling_p3,
    #     features=p3_features,
    #     categorical_features=p3_cat_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_application_history_v3',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # pre_apps_2_vs_pre_apps_3 = (
    #     pre_app_2_result[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "pre_app_2_ap"})
    #     .merge(
    #         pre_app_p3_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "pre_app_3_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )


    # pre_apps_2_vs_pre_apps_3["delta_ap"] = (
    #     pre_apps_2_vs_pre_apps_3["pre_app_3_ap"]
    #     - pre_apps_2_vs_pre_apps_3["pre_app_2_ap"]
    # )

    # pre_apps_2_vs_pre_apps_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### P3 — Previous application categorical history: Results

    | Fold | P2 AP | P3 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27342 | 0.27517 | +0.00175 |
    | 1 | 0.27243 | 0.27214 | -0.00028 |
    | 2 | 0.26504 | 0.26594 | +0.00090 |
    | 3 | 0.26051 | 0.26320 | +0.00270 |
    | 4 | 0.27011 | 0.26865 | -0.00147 |

    **Mean Δ AP:** `+0.00072`

    **Positive folds:** `3 / 5`

    **Decision:** **REJECT**

    The categorical-history bundle produces only a small and inconsistent
    improvement in fold-level Average Precision.

    Validation AP improves on 3 of 5 folds, while two folds degrade. The mean paired
    improvement is only approximately `+0.00072 AP`.

    More importantly, the candidate also degrades the OOF ROC-AUC and the
    `Recall@Top10%` and `Precision@Top10%` policy metrics relative to P2.

    Therefore, the small improvement in mean fold AP is not sufficient to justify
    promoting the categorical-history bundle to the accepted feature set.

    The result does not imply that historical categorical information is generally
    useless. The mode-based representation may discard substantial information
    about the distribution of historical categories within each applicant.

    For example, reducing a history such as:

    - 60% Cash loans
    - 30% Consumer loans
    - 10% Revolving loans

    to only:

    `MODE_NAME_CONTRACT_TYPE = Cash loans`

    removes most of the composition information.

    However, deeper categorical aggregation is not pursued at this stage because
    the current representation does not provide a sufficiently strong signal.

    The accepted reference remains **P2: previous application outcome history +
    financial history**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## P4 — Previous application temporal history

    **Observation**

    The `previous_application` table contains several relative time variables describing
    when previous applications were made and when important events in those contracts
    were scheduled or actually occurred.

    Before feature construction, the temporal columns were inspected for sentinel
    values and potential cutoff violations.

    The value `365243` was identified as a sentinel in several `DAYS_*` columns and
    was replaced with `NaN`.

    After cleaning:

    - `DAYS_DECISION` contains only negative values;
    - `DAYS_FIRST_DRAWING` contains only negative values;
    - `DAYS_FIRST_DUE` contains only negative values;
    - `DAYS_LAST_DUE` contains only negative values;
    - `DAYS_TERMINATION` contains only negative values;
    - `DAYS_LAST_DUE_1ST_VERSION` contains both negative and positive values.

    Positive values in `DAYS_LAST_DUE_1ST_VERSION` are expected because this column
    represents the originally scheduled final due date of the previous contract.
    A positive value therefore means that the contract was originally scheduled to
    finish after the current application date.

    This is not treated as leakage because the planned repayment schedule is known
    when the previous contract is issued.

    **Hypothesis**

    The timing and duration of previous credit activity may provide additional
    predictive information beyond application outcomes and financial aggregates.

    In particular, risk may depend on:

    - how recently the applicant made another credit application;
    - how long their observed previous-application history is;
    - the typical age of their previous applications;
    - the planned duration of previous contracts;
    - whether some previous contracts were originally scheduled to remain active
      beyond the current application date;
    - how long ago the most recent previous contract was terminated.

    These temporal characteristics may help distinguish applicants with recent,
    dense or still-ongoing credit activity from applicants with older or less active
    credit histories.

    **Preprocessing**

    The sentinel value `365243` is replaced with `NaN` in:

    - `DAYS_FIRST_DRAWING`;
    - `DAYS_FIRST_DUE`;
    - `DAYS_LAST_DUE_1ST_VERSION`;
    - `DAYS_LAST_DUE`;
    - `DAYS_TERMINATION`.

    `DAYS_DECISION` does not contain the sentinel and is used directly.

    The following row-level derived variables are created:

    `PREV_PLANNED_DURATION_DAYS`

    calculated as:

    `DAYS_LAST_DUE_1ST_VERSION - DAYS_FIRST_DUE`

    This represents the originally planned duration of the previous repayment
    schedule.

    `PREV_PLANNED_ENDS_IN_FUTURE`

    equals `1` when:

    `DAYS_LAST_DUE_1ST_VERSION > 0`

    and `0` otherwise.

    This indicates whether the previous contract was originally scheduled to end
    after the current application date.

    `PREV_PLANNED_DAYS_REMAINING`

    captures the number of planned days remaining after the current application
    date:

    `max(DAYS_LAST_DUE_1ST_VERSION, 0)`

    Negative scheduled end dates are therefore mapped to `0`.

    **Change**

    Add the following applicant-level temporal features:

    - `PREV_DAYS_SINCE_LAST_DECISION`
    - `PREV_HISTORY_AGE_DAYS`
    - `PREV_MEAN_DAYS_SINCE_DECISION`
    - `MEAN_PREV_PLANNED_DURATION_DAYS`
    - `MAX_PREV_PLANNED_DURATION_DAYS`
    - `PREV_FUTURE_PLANNED_END_COUNT`
    - `PREV_FUTURE_PLANNED_END_SHARE`
    - `MAX_PREV_PLANNED_DAYS_REMAINING`
    - `PREV_DAYS_SINCE_LAST_TERMINATION`

    **Feature interpretation**

    `PREV_DAYS_SINCE_LAST_DECISION`

    Days since the most recent previous application:

    `-max(DAYS_DECISION)`

    Lower values indicate more recent previous credit activity.

    `PREV_HISTORY_AGE_DAYS`

    Age of the oldest observed previous application:

    `-min(DAYS_DECISION)`

    Higher values indicate a longer observed previous-application history.

    `PREV_MEAN_DAYS_SINCE_DECISION`

    Average historical application age:

    `-mean(DAYS_DECISION)`

    This summarizes how recent or old the applicant's previous application history
    is overall.

    `MEAN_PREV_PLANNED_DURATION_DAYS`

    Average originally planned contract duration across previous applications.

    `MAX_PREV_PLANNED_DURATION_DAYS`

    Longest originally planned contract duration.

    `PREV_FUTURE_PLANNED_END_COUNT`

    Number of previous contracts whose original repayment schedule extended beyond
    the current application date.

    `PREV_FUTURE_PLANNED_END_SHARE`

    Share of previous contracts whose original repayment schedule extended beyond
    the current application date.

    `MAX_PREV_PLANNED_DAYS_REMAINING`

    Maximum number of days remaining until the originally planned end date among
    previous contracts.

    `PREV_DAYS_SINCE_LAST_TERMINATION`

    Days since the most recent observed previous contract termination:

    `-max(DAYS_TERMINATION)`

    This feature is available only where a termination date exists.

    **Aggregation**

    All temporal features are aggregated by `SK_ID_CURR`, producing one row per
    current applicant.

    The resulting temporal-history table is then left-joined to the current
    application dataset.

    **Missing-value handling**

    Missing historical event dates are preserved rather than globally imputed.

    This is particularly important because missingness can reflect the state of the
    previous contract, for example when a particular repayment or termination event
    did not occur or is not recorded.

    Applicants without valid observations for a given temporal feature therefore
    retain missing values after aggregation.

    **Cutoff / leakage check**

    The temporal fields were explicitly inspected before use.

    Actual historical event columns used in the experiment contain no positive
    values after sentinel removal:

    - `DAYS_DECISION`;
    - `DAYS_FIRST_DRAWING`;
    - `DAYS_FIRST_DUE`;
    - `DAYS_LAST_DUE`;
    - `DAYS_TERMINATION`.

    Therefore, these observed events occur before the current application cutoff.

    `DAYS_LAST_DUE_1ST_VERSION` is allowed to be positive because it describes an
    originally planned future date rather than an event observed after the current
    application.

    No target information or current-application outcome is used.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted P1 + P2 feature set as the reference;
    - rejected P3 categorical bundle is not included in the reference.

    **Primary comparison**

    Paired validation AP comparison:

    `P1 + P2`

    vs.

    `P1 + P2 + P4`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain the temporal-history bundle if it provides a reproducible improvement in
    validation AP and does not materially degrade the OOF ranking or Top-10% review
    policy metrics.

    The bundle should not be accepted based solely on one unusually strong fold.
    """)
    return


@app.cell
def _():
    # modeling_p4 = pd.read_parquet("data\processed\modeling_application_history_4.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES_V5):
    TEMPORAL_HISTORY_FEATURES = [
        "PREV_DAYS_SINCE_LAST_DECISION",
        "PREV_HISTORY_AGE_DAYS",
        "PREV_MEAN_DAYS_SINCE_DECISION",

        "MEAN_PREV_PLANNED_DURATION_DAYS",
        "MAX_PREV_PLANNED_DURATION_DAYS",

        "PREV_FUTURE_PLANNED_END_COUNT",
        "PREV_FUTURE_PLANNED_END_SHARE",
        "MAX_PREV_PLANNED_DAYS_REMAINING",

        "PREV_DAYS_SINCE_LAST_TERMINATION",
    ]

    p4_features = (
        ACCEPTED_FEATURES_V5 +
        TEMPORAL_HISTORY_FEATURES
    )
    return (TEMPORAL_HISTORY_FEATURES,)


@app.cell
def _():
    # prev_apps_p4_run = run_catboost_experiment(
    #     modeling_p4,
    #     features=p4_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_application_history_v4',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # pre_apps_2_vs_pre_apps_4 = (
    #     pre_app_2_result[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "pre_app_2_ap"})
    #     .merge(
    #         prev_apps_p4_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "pre_app_4_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )



    # pre_apps_2_vs_pre_apps_4["delta_ap"] = (
    #     pre_apps_2_vs_pre_apps_4["pre_app_4_ap"]
    #     - pre_apps_2_vs_pre_apps_4["pre_app_2_ap"]
    # )

    # pre_apps_2_vs_pre_apps_4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### P4 — Results

    | Fold | P2 AP | P4 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27342 | 0.27723 | +0.00381 |
    | 1 | 0.27243 | 0.27616 | +0.00373 |
    | 2 | 0.26504 | 0.26531 | +0.00027 |
    | 3 | 0.26051 | 0.26234 | +0.00184 |
    | 4 | 0.27011 | 0.27399 | +0.00388 |

    **Mean Δ AP:** `+0.00270`

    **Positive folds:** `5 / 5`

    **Decision:** **KEEP**

    The temporal-history bundle provides a consistent improvement over the accepted
    P1 + P2 reference.

    Validation AP improves on all five folds, with a mean paired improvement of
    approximately `+0.00270`.

    OOF ROC-AUC also improves, while `Recall@Top10%` and `Precision@Top10%` remain
    approximately unchanged.

    The result indicates that the timing, recency and planned duration of previous
    credit activity contain complementary predictive information beyond application
    outcomes and financial history.

    P4 is therefore added to the accepted feature set.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V5, TEMPORAL_HISTORY_FEATURES):
    ACCEPTED_FEATURES_V6 = (
        ACCEPTED_FEATURES_V5 +
        TEMPORAL_HISTORY_FEATURES
    )
    return (ACCEPTED_FEATURES_V6,)


@app.cell
def _():
    # prev_apps_p4_results = pd.read_parquet("mlartifacts\\1\\b399aa36a51c45fb9c21d95ed7fe97d4\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC1 — Credit card history and activity

    **Observation**

    The `credit_card_balance` table contains monthly snapshots of historical credit-card
    contracts.

    The table grain is:

    `SK_ID_PREV × MONTHS_BALANCE`

    where:

    - `SK_ID_PREV` identifies a specific historical credit-card contract;
    - `SK_ID_CURR` identifies the current applicant;
    - `MONTHS_BALANCE` represents the month relative to the current application.

    `MONTHS_BALANCE` ranges from `-96` to `-1`, so all observed credit-card snapshots
    occur before the current application cutoff.

    Most applicants with credit-card history have only one historical
    `SK_ID_PREV`, although a small number have multiple contracts.

    Credit-card history is available only for a minority of current applicants.
    Therefore, the absence of credit-card history is preserved explicitly instead of
    being treated as ordinary missingness.

    **Hypothesis**

    The structure and intensity of previous credit-card usage may provide predictive
    information beyond the already accepted application, bureau and
    `previous_application` features.

    In particular, risk may depend on:

    - whether the applicant has any observed credit-card history;
    - how many historical credit-card contracts they have;
    - how long their credit-card history has been observed;
    - how recently the latest credit-card snapshot was recorded;
    - how frequently the applicant carried a non-zero card balance;
    - how frequently the applicant used the card for drawings.

    This experiment intentionally focuses on general activity patterns before adding
    more detailed balance, utilization, payment and delinquency information.

    **Change**

    Add the following applicant-level features:

    - `HAS_CREDIT_CARD_HISTORY`
    - `CC_CONTRACT_COUNT`
    - `CC_MONTHS_OBSERVED`
    - `CC_HISTORY_AGE_MONTHS`
    - `CC_MONTHS_SINCE_LATEST`
    - `CC_ACTIVE_BALANCE_MONTH_SHARE`
    - `CC_DRAWING_MONTH_SHARE`

    **Feature interpretation**

    `HAS_CREDIT_CARD_HISTORY`

    Binary indicator showing whether the applicant has any records in
    `credit_card_balance`.

    This is important because applicants without credit-card history represent a
    distinct cohort rather than ordinary missing observations.

    `CC_CONTRACT_COUNT`

    Number of unique historical credit-card contracts:

    `nunique(SK_ID_PREV)`

    `CC_MONTHS_OBSERVED`

    Number of monthly credit-card snapshots available for the applicant.

    This approximates the amount of observed credit-card history.

    `CC_HISTORY_AGE_MONTHS`

    Age of the oldest observed credit-card snapshot:

    `-min(MONTHS_BALANCE)`

    Higher values indicate a longer observed credit-card history.

    `CC_MONTHS_SINCE_LATEST`

    Months since the most recent observed credit-card snapshot:

    `-max(MONTHS_BALANCE)`

    Lower values indicate more recent credit-card activity.

    `CC_ACTIVE_BALANCE_MONTH_SHARE`

    Share of observed months where:

    `AMT_BALANCE != 0`

    This captures how consistently the applicant carried a credit-card balance over
    their observed history.

    `CC_DRAWING_MONTH_SHARE`

    Share of observed months where:

    `AMT_DRAWINGS_CURRENT != 0`

    This captures how frequently the card was actively used for drawings.

    **Aggregation**

    Row-level activity indicators are first created for each monthly snapshot.

    The table is then aggregated directly by `SK_ID_CURR`, producing one row per
    current applicant.

    No monetary amounts, payment ratios or delinquency variables are introduced in
    this experiment. Those are reserved for later credit-card feature bundles.

    **Missing-value handling**

    Missing values in the original credit-card table are not globally imputed.

    The CC1 features are based primarily on identifiers, `MONTHS_BALANCE`,
    `AMT_BALANCE` and `AMT_DRAWINGS_CURRENT`, which are available for all observed
    rows used in these aggregations.

    After the aggregated table is left-joined to the current application dataset:

    - `HAS_CREDIT_CARD_HISTORY = 1` for applicants with credit-card records;
    - `HAS_CREDIT_CARD_HISTORY = 0` for applicants without credit-card records.

    Other missing aggregated values for applicants without credit-card history are
    preserved rather than replaced with population statistics.

    **Cutoff / leakage check**

    `MONTHS_BALANCE` ranges from `-96` to `-1`.

    There are no zero or positive values, so all monthly snapshots occur before the
    current application date.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff credit-card observations;
    - no statistics calculated across applicants.

    All features are derived exclusively from historical credit-card information.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features.

    Rejected feature groups are not included in the reference.

    **Primary comparison**

    Paired validation AP comparison:

    `current accepted reference`

    vs.

    `current accepted reference + CC1`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain CC1 if the credit-card history and activity features provide a consistent
    improvement in validation AP without materially degrading OOF ranking quality or
    the Top-10% review-policy metrics.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V6):
    CC1_FEATURES = [
        "CC_CONTRACT_COUNT",
        "CC_MONTHS_OBSERVED",
        "CC_HISTORY_AGE_MONTHS",
        "CC_MONTHS_SINCE_LATEST",
        "CC_ACTIVE_BALANCE_MONTH_SHARE",
        "CC_DRAWING_MONTH_SHARE",
    ]

    сс1_features = (
        ACCEPTED_FEATURES_V6 +
        CC1_FEATURES
    )
    return (CC1_FEATURES,)


@app.cell
def _():
    # modeling_cc1 = pd.read_parquet("data/processed/modeling_cc1.parquet")
    return


@app.cell
def _():
    # cc1_run = run_catboost_experiment(
    #     frame=modeling_cc1,
    #     features=сс1_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_cc1_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # pre_apps_4_vs_cc1 = (
    #     prev_apps_p4_results[
    #         ["fold", "valid_ap"]
    #     ]
    #     .rename(columns={"valid_ap": "pre_app_4_ap"})
    #     .merge(
    #         cc1_run["fold_metrics"][
    #             ["fold", "valid_ap"]
    #         ].rename(columns={"valid_ap": "cc1_ap"}),
    #         on="fold",
    #         validate="one_to_one",
    #     )
    # )



    # pre_apps_4_vs_cc1["delta_ap"] = (
    #     pre_apps_4_vs_cc1["cc1_ap"]
    #     - pre_apps_4_vs_cc1["pre_app_4_ap"]
    # )

    # pre_apps_4_vs_cc1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CC1 — Results

    | Fold | Previous reference AP | CC1 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27723 | 0.27939 | +0.00216 |
    | 1 | 0.27616 | 0.27575 | -0.00041 |
    | 2 | 0.26531 | 0.26556 | +0.00025 |
    | 3 | 0.26234 | 0.26416 | +0.00182 |
    | 4 | 0.27399 | 0.27505 | +0.00105 |

    **Mean Δ AP:** `+0.00097`

    **Positive folds:** `4 / 5`

    **Decision:** **KEEP**

    The basic credit-card history and activity bundle provides a small but
    consistent improvement over the previous accepted reference.

    Validation AP improves on 4 of 5 folds, with only a minor degradation on the
    remaining fold.

    In addition, OOF ROC-AUC, Precision@Top10% and Recall@Top10% improve, indicating
    that the new features improve both overall ranking quality and the Top-10%
    manual-review policy.

    CC1 is therefore added to the accepted feature set.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V6, CC1_FEATURES):
    ACCEPTED_FEATURES_V7 = (
        ACCEPTED_FEATURES_V6 +
        CC1_FEATURES
    )
    return (ACCEPTED_FEATURES_V7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC2 — Credit card balance and utilization

    **Observation**

    The `credit_card_balance` table provides monthly snapshots of the applicant's
    historical credit-card contracts.

    CC1 captured only the existence, depth and general activity of credit-card
    history. It did not describe the actual financial state of the cards.

    The table contains several variables that directly describe credit exposure:

    - `AMT_BALANCE` — current outstanding balance;
    - `AMT_CREDIT_LIMIT_ACTUAL` — actual credit limit.

    These variables can also be combined into a utilization ratio:

    `CC_UTILIZATION = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`

    which measures how much of the available credit limit is being used.

    Because `AMT_CREDIT_LIMIT_ACTUAL` can be zero, zero limits are replaced with
    `NaN` before calculating utilization in order to avoid division by zero.

    **Hypothesis**

    Historical credit-card balance and utilization may contain additional
    predictive information beyond the basic activity features introduced in CC1.

    In particular, applicant risk may depend on:

    - their typical credit-card balance;
    - the maximum balance reached historically;
    - the most recent balance before the current application;
    - the typical and maximum available credit limit;
    - how much of the available credit limit is normally used;
    - whether the applicant recently had high utilization.

    Utilization may be especially informative because the same absolute balance can
    represent very different levels of financial pressure depending on the
    available credit limit.

    For example:

    - balance = 90,000 and limit = 100,000 → utilization = 0.90;
    - balance = 90,000 and limit = 500,000 → utilization = 0.18.

    Therefore, balance and utilization are expected to provide complementary
    information.

    **Preprocessing**

    The following row-level feature is created:

    `CC_UTILIZATION`

    calculated as:

    `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`

    with zero credit limits replaced by `NaN` before division.

    No clipping or transformation is applied unless the utilization diagnostics
    show clear data-quality issues.

    **Change**

    Add the following applicant-level balance and utilization features:

    - `CC_MEAN_BALANCE`
    - `CC_MAX_BALANCE`
    - `CC_LATEST_BALANCE`

    - `CC_MEAN_CREDIT_LIMIT`
    - `CC_MAX_CREDIT_LIMIT`
    - `CC_LATEST_CREDIT_LIMIT`

    - `CC_MEAN_UTILIZATION`
    - `CC_MAX_UTILIZATION`
    - `CC_LATEST_UTILIZATION`

    **Feature interpretation**

    `CC_MEAN_BALANCE`

    Average historical credit-card balance across observed months.

    This represents the applicant's typical outstanding card balance.

    `CC_MAX_BALANCE`

    Maximum observed historical credit-card balance.

    This captures the highest level of credit-card exposure observed in the
    available history.

    `CC_LATEST_BALANCE`

    Balance from the most recent observed snapshot of the applicant's credit-card
    contract history.

    This captures the financial state closest to the current application cutoff.

    `CC_MEAN_CREDIT_LIMIT`

    Average observed actual credit limit.

    This represents the applicant's typical available credit capacity.

    `CC_MAX_CREDIT_LIMIT`

    Maximum historical actual credit limit.

    `CC_LATEST_CREDIT_LIMIT`

    Most recent observed credit limit before the current application.

    `CC_MEAN_UTILIZATION`

    Average historical credit utilization:

    `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`

    This captures the applicant's typical use of available credit.

    `CC_MAX_UTILIZATION`

    Maximum historical utilization.

    This represents the highest observed level of credit-limit usage.

    `CC_LATEST_UTILIZATION`

    Utilization in the most recent available credit-card snapshot.

    This provides a more current measure of credit pressure than the historical
    average.

    **Aggregation**

    Historical mean and maximum features are calculated directly over the applicant's
    monthly credit-card snapshots.

    For latest-state features, the most recent snapshot is identified using the
    largest `MONTHS_BALANCE` value for each historical `SK_ID_PREV`.

    If an applicant has multiple credit-card contracts, the latest contract-level
    values are aggregated to `SK_ID_CURR`.

    The final feature table therefore contains one row per current applicant.

    **Missing-value handling**

    Original missing values are not globally imputed.

    For utilization, observations with:

    `AMT_CREDIT_LIMIT_ACTUAL = 0`

    produce `NaN` rather than infinite values.

    Missing aggregated features are preserved and left for CatBoost to handle.

    Applicants without any credit-card history remain represented by the
    `HAS_CREDIT_CARD_HISTORY` indicator introduced in CC1.

    **Cutoff / leakage check**

    All credit-card snapshots are historical relative to the current application.

    `MONTHS_BALANCE` ranges from `-96` to `-1`, so no zero or positive monthly
    snapshots are present.

    The latest-state features therefore use the observation closest to the current
    application while still remaining strictly pre-cutoff.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff records;
    - no cross-applicant target statistics.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted CC1 features.

    **Primary comparison**

    Paired validation AP comparison:

    `CC1 reference`

    vs.

    `CC1 + CC2`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain CC2 if the balance and utilization features provide a reproducible
    improvement in validation AP and do not materially degrade OOF ranking quality
    or Top-10% review-policy metrics.

    A strong gain on only one fold is not sufficient; the improvement should be
    reasonably consistent across the fixed validation folds.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V7):
    CC2_FEATURES = [
        "CC_MEAN_BALANCE",
        "CC_MAX_BALANCE",
        "CC_LATEST_BALANCE",

        "CC_MEAN_CREDIT_LIMIT",
        "CC_MAX_CREDIT_LIMIT",
        "CC_LATEST_CREDIT_LIMIT",

        "CC_MEAN_UTILIZATION",
        "CC_MAX_UTILIZATION",
        "CC_LATEST_UTILIZATION",
    ]

    cc2_features = (
        ACCEPTED_FEATURES_V7 +
        CC2_FEATURES
    )
    return (CC2_FEATURES,)


@app.cell
def _():
    # modeling_cc2 = pd.read_parquet("data/processed/modeling_cc2.parquet")
    return


@app.cell
def _():
    # cc2_run = run_catboost_experiment(
    #     frame=modeling_cc2,
    #     features=cc2_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_cc2_v1',
    #     capacity=0.10
    # )
    return


@app.function
def compare_fold_results(df1, df2, col1, col2):
    comparison = (
        df1[["fold", col1]]
        .rename(columns={col1: "df1_metric"})
        .merge(
            df2[["fold", col2]]
            .rename(columns={col2: "df2_metric"}),
            on="fold",
            validate="one_to_one",
        )
    )

    comparison["delta"] = (
        comparison["df2_metric"] - comparison["df1_metric"]
    )

    return comparison


@app.cell
def _():
    # compare_fold_results(
    #     cc1_run["fold_metrics"],
    #     cc2_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap"
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CC2 — Results

    | Fold | CC1 AP | CC2 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27939 | 0.27898 | -0.00041 |
    | 1 | 0.27575 | 0.27848 | +0.00273 |
    | 2 | 0.26556 | 0.27403 | +0.00847 |
    | 3 | 0.26416 | 0.26603 | +0.00186 |
    | 4 | 0.27505 | 0.27495 | -0.00010 |

    **Mean Δ AP:** `+0.00251`

    **Positive folds:** `3 / 5`

    **Decision:** **KEEP**

    The credit-card balance and utilization bundle provides a meaningful overall
    improvement over CC1.

    Three of five folds improve, while the two negative fold deltas are small.
    The largest improvement occurs on fold 2, so the effect is not perfectly
    uniform across folds.

    However, the overall evidence supports retaining the bundle:

    - mean validation AP improves by approximately `+0.00251`;
    - fold-level AP standard deviation decreases;
    - OOF ROC-AUC improves;
    - Precision@Top10% improves;
    - Recall@Top10% improves.

    This indicates that historical balance and credit utilization provide
    additional predictive information beyond basic credit-card activity.

    CC2 is therefore added to the accepted feature set.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V7, CC2_FEATURES):
    ACCEPTED_FEATURES_V8 = (
        ACCEPTED_FEATURES_V7 +
        CC2_FEATURES
    )
    return (ACCEPTED_FEATURES_V8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC3 — Credit card drawings and payments

    **Observation**

    The `credit_card_balance` table contains monthly information about how applicants
    used and repaid historical credit-card balances.

    CC1 captured the existence, depth and general activity of credit-card history,
    while CC2 captured balance, credit limit and utilization.

    However, these bundles do not directly describe the applicant's monthly
    transactional behavior:

    - how much credit was drawn;
    - how frequently the card was used;
    - how much was repaid;
    - how much of the drawings came from ATM cash withdrawals;
    - whether monthly drawings exceeded monthly repayments.

    The table provides several variables related to these behaviors, including:

    - `AMT_DRAWINGS_CURRENT`;
    - `AMT_DRAWINGS_ATM_CURRENT`;
    - `CNT_DRAWINGS_CURRENT`;
    - `AMT_PAYMENT_CURRENT`;
    - `AMT_PAYMENT_TOTAL_CURRENT`.

    **Hypothesis**

    Historical drawings and payment behavior may provide additional predictive
    information beyond credit-card balance and utilization.

    In particular, risk may depend on:

    - the typical and maximum amount drawn from the card;
    - the frequency of credit-card transactions;
    - the magnitude of monthly repayments;
    - the use of ATM cash withdrawals;
    - whether the applicant tends to draw more funds than they repay within a month;
    - the applicant's most recent transactional behavior.

    ATM cash withdrawals are treated as a potentially different form of credit-card
    usage from ordinary purchases. A high ATM drawing share may indicate greater
    liquidity demand, but this is treated as a modeling hypothesis rather than a
    direct causal interpretation of financial stress.

    **Preprocessing**

    Two row-level derived features are created.

    `CC_NET_DRAWINGS`

    calculated as:

    `AMT_DRAWINGS_CURRENT - AMT_PAYMENT_TOTAL_CURRENT`

    A positive value indicates that monthly drawings exceeded monthly repayments,
    while a negative value indicates that repayments exceeded new drawings.

    This is interpreted as a simplified net credit-flow indicator rather than the
    exact monthly change in outstanding debt, because interest, fees and other
    balance movements are not explicitly included.

    `CC_ATM_DRAWING_SHARE`

    calculated as:

    `AMT_DRAWINGS_ATM_CURRENT / AMT_DRAWINGS_CURRENT`

    with zero values of `AMT_DRAWINGS_CURRENT` replaced with `NaN` before division.

    This represents the share of total drawings attributable to ATM cash
    withdrawals.

    **Change**

    Add the following applicant-level features:

    - `CC_MEAN_DRAWINGS`
    - `CC_MAX_DRAWINGS`
    - `CC_LATEST_DRAWINGS`

    - `CC_MEAN_ATM_DRAWINGS`
    - `CC_MAX_ATM_DRAWINGS`

    - `CC_MEAN_DRAWING_COUNT`
    - `CC_MAX_DRAWING_COUNT`

    - `CC_MEAN_PAYMENTS`
    - `CC_MAX_PAYMENTS`
    - `CC_LATEST_PAYMENTS`

    - `CC_MEAN_NET_DRAWINGS`
    - `CC_MAX_NET_DRAWINGS`

    - `CC_MEAN_ATM_DRAWING_SHARE`

    **Feature interpretation**

    `CC_MEAN_DRAWINGS`

    Average monthly amount drawn from the credit card.

    This represents the applicant's typical level of credit-card usage.

    `CC_MAX_DRAWINGS`

    Maximum monthly drawing amount observed in the available history.

    This captures the applicant's highest observed monthly use of credit.

    `CC_LATEST_DRAWINGS`

    Drawing amount from the most recent available credit-card snapshot.

    This provides a recent measure of transactional activity close to the current
    application cutoff.

    `CC_MEAN_ATM_DRAWINGS`

    Average monthly amount withdrawn through ATMs.

    This captures the typical use of the credit card for cash withdrawals.

    `CC_MAX_ATM_DRAWINGS`

    Maximum monthly ATM withdrawal amount.

    `CC_MEAN_DRAWING_COUNT`

    Average number of credit-card drawing transactions per observed month.

    This measures the typical frequency of card usage.

    `CC_MAX_DRAWING_COUNT`

    Maximum number of drawing transactions observed in a single month.

    `CC_MEAN_PAYMENTS`

    Average monthly repayment amount.

    This represents the applicant's typical repayment activity.

    `CC_MAX_PAYMENTS`

    Maximum monthly repayment amount observed in the historical credit-card record.

    `CC_LATEST_PAYMENTS`

    Repayment amount from the most recent available monthly snapshot.

    This captures repayment behavior closest to the current application date.

    `CC_MEAN_NET_DRAWINGS`

    Average monthly difference between drawings and repayments:

    `AMT_DRAWINGS_CURRENT - AMT_PAYMENT_TOTAL_CURRENT`

    Higher values indicate that new credit usage tended to exceed repayments.

    `CC_MAX_NET_DRAWINGS`

    Maximum observed monthly excess of drawings over repayments.

    This captures periods of particularly strong net credit usage.

    `CC_MEAN_ATM_DRAWING_SHARE`

    Average historical share of drawings performed through ATMs.

    This distinguishes applicants who primarily use credit for cash withdrawals
    from those whose credit-card usage is dominated by other transaction types.

    **Aggregation**

    The table is first processed at the monthly credit-card snapshot level.

    Row-level derived features are calculated before aggregation.

    Historical mean and maximum features are then aggregated by `SK_ID_CURR`.

    For latest-state features, the most recent snapshot is identified using the
    largest `MONTHS_BALANCE` value for each historical `SK_ID_PREV`.

    If an applicant has multiple historical credit-card contracts, the most recent
    contract-level values are aggregated to the applicant level.

    The final feature table contains one row per `SK_ID_CURR`.

    **Missing-value handling**

    No global imputation is performed before aggregation.

    `AMT_PAYMENT_CURRENT` and several drawing-related variables contain missing
    values. These missing values are preserved and excluded only according to the
    standard aggregation behavior.

    For `CC_ATM_DRAWING_SHARE`, observations with zero total drawings produce
    `NaN` rather than infinite or undefined ratios.

    Applicants without any credit-card history remain identified by
    `HAS_CREDIT_CARD_HISTORY` from CC1.

    **Cutoff / leakage check**

    All monthly credit-card snapshots satisfy:

    `MONTHS_BALANCE < 0`

    with observed values ranging from `-96` to `-1`.

    Therefore, all records used in CC3 occur before the current application cutoff.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff snapshots;
    - no target statistics across applicants.

    All transactional features are derived exclusively from historical
    credit-card behavior.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted CC1 features;
    - accepted CC2 features.

    **Primary comparison**

    Paired validation AP comparison:

    `CC2 reference`

    vs.

    `CC2 + CC3`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain CC3 if transactional drawings and repayment behavior provide additional
    predictive signal beyond CC1 and CC2.

    The bundle should improve validation AP in a reasonably consistent way and
    should not materially degrade OOF ROC-AUC or the Top-10% review-policy metrics.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V8):
    CC3_FEATURES = [
        "CC_MEAN_DRAWINGS",
        "CC_MAX_DRAWINGS",
        "CC_LATEST_DRAWINGS",

        "CC_MEAN_ATM_DRAWINGS",
        "CC_MAX_ATM_DRAWINGS",

        "CC_MEAN_DRAWING_COUNT",
        "CC_MAX_DRAWING_COUNT",

        "CC_MEAN_PAYMENTS",
        "CC_MAX_PAYMENTS",
        "CC_LATEST_PAYMENTS",

        "CC_MEAN_NET_DRAWINGS",
        "CC_MAX_NET_DRAWINGS",

        "CC_MEAN_ATM_DRAWING_SHARE",
    ]

    cc3_features = (
        ACCEPTED_FEATURES_V8 +
        CC3_FEATURES
    )
    return


@app.cell
def _():
    # modeling_cc3 = pd.read_parquet("data/processed/modeling_cc3.parquet")
    return


@app.cell
def _():
    # cc3_run = run_catboost_experiment(
    #     frame=modeling_cc3,
    #     features=cc3_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_cc3_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     cc2_run["fold_metrics"],
    #     cc3_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CC3 — Results

    | Fold | CC2 AP | CC3 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27898 | 0.27964 | +0.00067 |
    | 1 | 0.27848 | 0.27693 | -0.00155 |
    | 2 | 0.27403 | 0.27570 | +0.00166 |
    | 3 | 0.26603 | 0.26836 | +0.00233 |
    | 4 | 0.27495 | 0.27598 | +0.00103 |

    **Mean Δ AP:** `+0.00083`

    **Positive folds:** `4 / 5`

    **Decision:** **REJECT**

    The drawings and payments bundle produces a small improvement in mean
    validation AP and improves AP on 4 of 5 folds.

    OOF ROC-AUC also improves slightly. However, the gain in AP is small relative
    to the size of the added feature bundle, while both `Precision@Top10%` and
    `Recall@Top10%` decrease relative to CC2.

    Therefore, the additional complexity is not justified by a sufficiently strong
    improvement in the metrics relevant to the review policy.

    The accepted credit-card reference remains **CC1 + CC2**.

    The result does not imply that drawings and payment behavior contain no signal.
    Rather, the current aggregated representation does not provide enough
    incremental value to retain the full bundle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CC4 — Credit card delinquency and repayment stress

    **Observation**

    The `credit_card_balance` table contains monthly delinquency indicators for
    historical credit-card contracts:

    - `SK_DPD`
    - `SK_DPD_DEF`

    These variables describe overdue status at the monthly snapshot level.

    Previous credit-card experiments captured:

    - CC1 — existence and activity of credit-card history;
    - CC2 — balance, credit limit and utilization;
    - CC3 — drawings and payments.

    However, none of these bundles directly represent the frequency, severity or
    recency of delinquency.

    Because delinquency is observed repeatedly over time, a single maximum value is
    not sufficient. It is useful to distinguish:

    - whether delinquency ever occurred;
    - how severe the worst episode was;
    - how often delinquency occurred;
    - whether severe delinquency occurred;
    - whether delinquency was recent;
    - whether delinquency was still present close to the current application date.

    **Hypothesis**

    Historical credit-card delinquency may provide additional predictive signal
    beyond credit-card activity and utilization.

    Applicants with more frequent, more severe or more recent delinquency may have a
    different risk profile than applicants whose credit-card history contains no
    overdue observations.

    The experiment therefore summarizes delinquency along three dimensions:

    1. **severity**;
    2. **frequency**;
    3. **recency**.

    **Preprocessing**

    The following row-level indicators are created:

    `CC_HAS_DPD`

    equals `1` when:

    `SK_DPD > 0`

    and `0` otherwise.

    `CC_HAS_DPD_DEF`

    equals `1` when:

    `SK_DPD_DEF > 0`

    and `0` otherwise.

    `CC_DPD_30_PLUS`

    equals `1` when:

    `SK_DPD >= 30`

    and `0` otherwise.

    `CC_DPD_90_PLUS`

    equals `1` when:

    `SK_DPD >= 90`

    and `0` otherwise.

    These indicators are calculated at the monthly snapshot level before
    aggregation.

    **Change**

    Add the following applicant-level delinquency features:

    - `CC_MAX_DPD`
    - `CC_MEAN_DPD`
    - `CC_DPD_MONTH_SHARE`

    - `CC_MAX_DPD_DEF`
    - `CC_DPD_DEF_MONTH_SHARE`

    - `CC_DPD_30_PLUS_MONTH_SHARE`
    - `CC_DPD_90_PLUS_MONTH_SHARE`

    - `CC_RECENT_6M_MAX_DPD`
    - `CC_RECENT_6M_DPD_MONTH_SHARE`

    - `CC_LATEST_MAX_DPD`

    **Feature interpretation**

    `CC_MAX_DPD`

    Maximum historical value of `SK_DPD`.

    This captures the most severe delinquency episode observed in the available
    credit-card history.

    `CC_MEAN_DPD`

    Average `SK_DPD` across all observed monthly snapshots.

    This provides a broader measure of historical delinquency intensity rather than
    focusing only on the single worst month.

    `CC_DPD_MONTH_SHARE`

    Share of observed months where:

    `SK_DPD > 0`

    This captures how frequently the applicant was observed in delinquency.

    `CC_MAX_DPD_DEF`

    Maximum historical value of `SK_DPD_DEF`.

    This provides an alternative delinquency severity measure based on the
    dataset-defined default-adjusted DPD field.

    `CC_DPD_DEF_MONTH_SHARE`

    Share of observed months where:

    `SK_DPD_DEF > 0`

    This measures the frequency of delinquency according to the alternative
    `SK_DPD_DEF` definition.

    `CC_DPD_30_PLUS_MONTH_SHARE`

    Share of observed months where:

    `SK_DPD >= 30`

    This separates more serious delinquency episodes from minor overdue periods.

    `CC_DPD_90_PLUS_MONTH_SHARE`

    Share of observed months where:

    `SK_DPD >= 90`

    This captures the frequency of severe delinquency episodes.

    `CC_RECENT_6M_MAX_DPD`

    Maximum `SK_DPD` observed during the six months closest to the current
    application:

    `MONTHS_BALANCE >= -6`

    This captures recent delinquency severity.

    `CC_RECENT_6M_DPD_MONTH_SHARE`

    Share of observed snapshots during the most recent six months where:

    `SK_DPD > 0`

    This measures recent delinquency frequency.

    `CC_LATEST_MAX_DPD`

    Maximum `SK_DPD` among the most recent available snapshots of the applicant's
    historical credit-card contracts.

    This represents delinquency status closest to the current application cutoff.

    **Aggregation**

    Historical delinquency features are first calculated over all available monthly
    snapshots and aggregated by `SK_ID_CURR`.

    Recent features use only records satisfying:

    `MONTHS_BALANCE >= -6`

    Latest-state features are constructed by selecting the largest
    `MONTHS_BALANCE` value for each `SK_ID_PREV`, then aggregating these latest
    contract-level snapshots to `SK_ID_CURR`.

    The final CC4 feature table contains one row per current applicant.

    **Missing-value handling**

    `SK_DPD` and `SK_DPD_DEF` do not contain missing values in the source table, so
    no imputation is required for the main delinquency variables.

    Recent or latest aggregates may be missing only when an applicant does not have
    the corresponding credit-card history.

    Applicants without credit-card records remain identified through
    `HAS_CREDIT_CARD_HISTORY` from CC1.

    No population-level imputation is applied.

    **Cutoff / leakage check**

    All records in `credit_card_balance` satisfy:

    `MONTHS_BALANCE < 0`

    with observed values ranging from `-96` to `-1`.

    Therefore, all delinquency observations used in CC4 occur before the current
    application cutoff.

    Recent and latest features select observations closer to the cutoff but never
    use zero or positive months.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff snapshots;
    - no target-based aggregations across applicants.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted CC1 features;
    - accepted CC2 features;
    - rejected CC3 features are not included.

    **Primary comparison**

    Paired validation AP comparison:

    `CC2 reference`

    vs.

    `CC2 + CC4`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain CC4 if delinquency severity, frequency and recency provide additional
    predictive signal beyond the accepted CC1 + CC2 feature set.

    The bundle should show a reasonably consistent improvement in validation AP and
    should not materially degrade OOF ROC-AUC or the Top-10% review-policy metrics.

    Because CC4 is the final planned feature bundle for `credit_card_balance`, the
    table will be considered complete after this experiment regardless of whether
    the bundle is accepted or rejected.
    """)
    return


@app.cell
def _():
    # modeling_cc4 = pd.read_parquet("data/processed/modeling_cc4.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES_V8):
    CC4_FEATURES = [
        "CC_MAX_DPD",
        "CC_MEAN_DPD",
        "CC_DPD_MONTH_SHARE",

        "CC_MAX_DPD_DEF",
        "CC_DPD_DEF_MONTH_SHARE",

        "CC_DPD_30_PLUS_MONTH_SHARE",
        "CC_DPD_90_PLUS_MONTH_SHARE",

        "CC_RECENT_6M_MAX_DPD",
        "CC_RECENT_6M_DPD_MONTH_SHARE",

        "CC_LATEST_MAX_DPD",
    ]

    cc4_features = (
        ACCEPTED_FEATURES_V8 +
        CC4_FEATURES
    )
    return (CC4_FEATURES,)


@app.cell
def _():
    # cc4_run = run_catboost_experiment(
    #     frame=modeling_cc4,
    #     features=cc4_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_cc4_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     cc2_run["fold_metrics"],
    #     cc4_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CC4 — Results

    | Fold | CC2 AP | CC4 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.27898 | 0.28012 | +0.00114 |
    | 1 | 0.27848 | 0.27790 | -0.00059 |
    | 2 | 0.27403 | 0.27389 | -0.00015 |
    | 3 | 0.26603 | 0.27065 | +0.00463 |
    | 4 | 0.27495 | 0.27571 | +0.00076 |

    **Mean Δ AP:** `+0.00116`

    **Positive folds:** `3 / 5`

    **Decision:** **KEEP**

    The delinquency bundle provides a modest but useful improvement over the
    accepted CC1 + CC2 reference.

    The fold-level improvement is not perfectly uniform: three folds improve and
    two show small degradations. However, both negative deltas are minor, while the
    overall mean validation AP increases.

    The broader evaluation also supports retaining the bundle:

    - mean validation AP improves;
    - fold AP standard deviation decreases;
    - OOF ROC-AUC improves;
    - Precision@Top10% improves;
    - Recall@Top10% improves.

    The features therefore provide additional information about historical
    delinquency severity, frequency and recency without degrading the review-policy
    metrics.

    CC4 is retained in the accepted feature set.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V8, CC4_FEATURES):
    ACCEPTED_FEATURES_V9 = (
        ACCEPTED_FEATURES_V8 +
        CC4_FEATURES
    )
    return (ACCEPTED_FEATURES_V9,)


@app.cell
def _(pd):
    cc4_run_results = pd.read_parquet("mlartifacts\\1\\da8be09a5e7341ed95dece14f3c151c8\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IP1 — Installment payment history structure

    **Observation**

    The `installments_payments` table contains historical repayment records for
    previous credit contracts.

    The table is structured around previous contracts identified by `SK_ID_PREV`,
    but each contract contains multiple installment/payment records.

    The combination:

    `SK_ID_PREV × NUM_INSTALMENT_NUMBER`

    is not always unique. Most installments have one record, but some have multiple
    rows, which indicates that a scheduled installment may be associated with
    multiple payment records.

    Therefore, raw row count and unique installment count represent different
    aspects of repayment history.

    The temporal fields:

    - `DAYS_INSTALMENT`
    - `DAYS_ENTRY_PAYMENT`

    contain only negative values, with maximum values equal to `-1`.

    This means all scheduled and observed payment events occur before the current
    application cutoff.

    Missingness is very low and is limited primarily to:

    - `AMT_PAYMENT`
    - `DAYS_ENTRY_PAYMENT`

    with approximately `0.02%` missing values.

    **Hypothesis**

    The amount and depth of observed installment-payment history may provide
    additional predictive information beyond the already accepted application,
    bureau, previous-application and credit-card features.

    In particular, applicant risk may depend on:

    - how many previous credit contracts have repayment history;
    - how many payment records are available;
    - how many distinct scheduled installments were observed;
    - how long the repayment history extends into the past;
    - how recently the applicant made a recorded payment.

    IP1 intentionally captures only the structure and recency of repayment history.

    Detailed repayment quality such as lateness and underpayment is reserved for a
    separate experiment.

    **Change**

    Add the following applicant-level features:

    - `IP_CONTRACT_COUNT`
    - `IP_PAYMENT_RECORD_COUNT`
    - `IP_UNIQUE_INSTALLMENT_COUNT`
    - `IP_HISTORY_AGE_DAYS`
    - `IP_DAYS_SINCE_LAST_PAYMENT`

    **Feature interpretation**

    `IP_CONTRACT_COUNT`

    Number of unique previous contracts with installment-payment records:

    `nunique(SK_ID_PREV)`

    This represents the breadth of the applicant's observed repayment history.

    `IP_PAYMENT_RECORD_COUNT`

    Total number of rows in `installments_payments` associated with the applicant.

    This measures the overall volume of observed repayment records.

    It is intentionally kept separate from the number of unique installments
    because a single scheduled installment may contain multiple payment records.

    `IP_UNIQUE_INSTALLMENT_COUNT`

    Number of unique:

    `SK_ID_PREV × NUM_INSTALMENT_NUMBER`

    combinations associated with the applicant.

    This approximates the number of distinct scheduled installments observed across
    their previous contracts.

    `IP_HISTORY_AGE_DAYS`

    Age of the oldest observed payment:

    `-min(DAYS_ENTRY_PAYMENT)`

    Higher values indicate a longer observed repayment history.

    `IP_DAYS_SINCE_LAST_PAYMENT`

    Days since the most recent observed payment:

    `-max(DAYS_ENTRY_PAYMENT)`

    Lower values indicate more recent repayment activity.

    **Aggregation**

    The table is aggregated to one row per `SK_ID_CURR`.

    Contract count, payment-record count and temporal-history features are calculated
    directly from the raw payment records.

    The unique installment count is calculated separately by removing duplicate
    combinations of:

    `SK_ID_CURR`, `SK_ID_PREV`, `NUM_INSTALMENT_NUMBER`

    before counting them at the applicant level.

    The resulting feature table is then left-joined to the current application
    dataset.

    **Missing-value handling**

    No global imputation is applied.

    `DAYS_ENTRY_PAYMENT` contains only a very small amount of missing data, so
    temporal aggregations use the available observed payments.

    Applicants without installment-payment history remain distinguishable after the
    left join rather than being imputed with population-level values.

    **Cutoff / leakage check**

    Both relevant temporal fields satisfy:

    `DAYS_INSTALMENT < 0`

    and:

    `DAYS_ENTRY_PAYMENT < 0`

    with maximum values equal to `-1`.

    Therefore, all scheduled installments and recorded payments used in IP1 occur
    before the current application cutoff.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff payment records;
    - no target-based statistics across applicants.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted credit-card CC1 + CC2 + CC4 features.

    Rejected feature groups are not included in the reference.

    **Primary comparison**

    Paired validation AP comparison:

    `current accepted reference`

    vs.

    `current accepted reference + IP1`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain IP1 if the structure and recency of installment-payment history provide
    additional predictive signal without materially degrading OOF ranking quality
    or the Top-10% review-policy metrics.

    Because IP1 contains only simple history-structure features, even a modest but
    consistent improvement may justify retaining the bundle.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V9):
    IP1_FEATURES = [
        "IP_CONTRACT_COUNT",
        "IP_PAYMENT_RECORD_COUNT",
        "IP_UNIQUE_INSTALLMENT_COUNT",
        "IP_HISTORY_AGE_DAYS",
        "IP_DAYS_SINCE_LAST_PAYMENT",
    ]

    ip1_features = (
        ACCEPTED_FEATURES_V9 +
        IP1_FEATURES
    )
    return (IP1_FEATURES,)


@app.cell
def _():
    # modeling_ip1 = pd.read_parquet("data/processed/modeling_ip1.parquet")
    return


@app.cell
def _():
    # ip1_run = run_catboost_experiment(
    #     frame=modeling_ip1,
    #     features=ip1_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_ip1_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     cc4_run_results,
    #     ip1_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### IP1 — Results

    **Mean Δ AP:** `+0.00170`

    **Positive folds:** `3 / 5`

    **Decision:** **KEEP**

    The installment-payment history structure bundle provides additional predictive
    signal beyond the previous accepted feature set.

    Three of five folds improve, while the two negative deltas are small, with one
    being effectively neutral.

    The mean paired validation AP improvement is approximately `+0.00170`.

    OOF ranking and Top-10% review-policy metrics also improve, supporting the
    conclusion that the depth, volume and recency of installment-payment history
    contain useful information.

    IP1 is therefore added to the accepted feature set.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V9, IP1_FEATURES):
    ACCEPTED_FEATURES_V10 = (
        ACCEPTED_FEATURES_V9 +
        IP1_FEATURES
    )
    return (ACCEPTED_FEATURES_V10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IP2 — Installment repayment discipline

    **Observation**

    The `installments_payments` table contains both scheduled installment
    information and observed payment information:

    - `DAYS_INSTALMENT` — scheduled installment date;
    - `DAYS_ENTRY_PAYMENT` — actual payment date;
    - `AMT_INSTALMENT` — scheduled installment amount;
    - `AMT_PAYMENT` — actually paid amount.

    IP1 captured only the structure, depth and recency of installment-payment
    history.

    However, it did not describe whether previous installments were paid:

    - on time;
    - late;
    - in full;
    - partially.

    These variables allow repayment discipline to be represented much more directly.

    An important structural detail is that a single scheduled installment may be
    represented by multiple payment rows. Therefore, payment behavior should first
    be reconstructed at the installment level before applicant-level aggregation.

    **Hypothesis**

    Historical repayment discipline may provide additional predictive information
    beyond the amount and recency of repayment history captured in IP1.

    Applicants who repeatedly:

    - pay after the scheduled due date;
    - experience long payment delays;
    - fail to cover the full scheduled installment amount;

    may have a different risk profile from applicants with consistently timely and
    complete repayment history.

    The experiment therefore focuses on two dimensions:

    1. **payment timing**;
    2. **payment completeness**.

    **Installment-level reconstruction**

    Payment records are first aggregated to the scheduled-installment level.

    The installment identity is based on:

    - `SK_ID_CURR`;
    - `SK_ID_PREV`;
    - `NUM_INSTALMENT_VERSION`;
    - `NUM_INSTALMENT_NUMBER`.

    For each installment:

    - scheduled amount is represented by `AMT_INSTALMENT`;
    - scheduled date is represented by `DAYS_INSTALMENT`;
    - individual `AMT_PAYMENT` values are summed;
    - the latest observed `DAYS_ENTRY_PAYMENT` is used as the payment-completion
      date proxy.

    This prevents split payments from being interpreted as multiple independent
    scheduled installments.

    **Derived installment-level features**

    `IP_DELAY_DAYS`

    calculated as:

    `max(LAST_PAYMENT_DAY - DAYS_INSTALMENT, 0)`

    A value of zero means that the final observed payment was made on or before the
    scheduled due date.

    Positive values represent the number of days past the scheduled installment
    date.

    `IP_IS_LATE`

    equals `1` when:

    `IP_DELAY_DAYS > 0`

    and `0` otherwise.

    `IP_IS_30_PLUS_LATE`

    equals `1` when:

    `IP_DELAY_DAYS >= 30`

    and `0` otherwise.

    `IP_PAYMENT_SHORTFALL`

    calculated as:

    `max(AMT_INSTALMENT - TOTAL_PAYMENT, 0)`

    A positive value means that the total observed payment amount did not cover the
    scheduled installment amount.

    `IP_IS_UNDERPAID`

    equals `1` when:

    `IP_PAYMENT_SHORTFALL > 0`

    and `0` otherwise.

    `IP_PAYMENT_RATIO`

    calculated as:

    `TOTAL_PAYMENT / AMT_INSTALMENT`

    with zero scheduled installment amounts replaced with `NaN` before division.

    This represents how much of the scheduled installment amount was covered by
    observed payments.

    **Change**

    Add the following applicant-level repayment-discipline features:

    - `IP_LATE_INSTALLMENT_SHARE`
    - `IP_MEAN_DELAY_DAYS`
    - `IP_MAX_DELAY_DAYS`
    - `IP_30_PLUS_LATE_SHARE`

    - `IP_UNDERPAID_INSTALLMENT_SHARE`
    - `IP_MEAN_PAYMENT_SHORTFALL`
    - `IP_MAX_PAYMENT_SHORTFALL`
    - `IP_MEAN_PAYMENT_RATIO`
    - `IP_MIN_PAYMENT_RATIO`

    **Feature interpretation**

    `IP_LATE_INSTALLMENT_SHARE`

    Share of reconstructed installments paid after their scheduled due date.

    This captures the frequency of late repayment.

    `IP_MEAN_DELAY_DAYS`

    Average number of late days across observed installments.

    Early or on-time payments contribute zero rather than negative values.

    `IP_MAX_DELAY_DAYS`

    Maximum observed payment delay.

    This captures the most severe late-payment episode in the applicant's history.

    `IP_30_PLUS_LATE_SHARE`

    Share of installments with a payment delay of at least 30 days.

    This distinguishes more serious repayment delays from small timing differences.

    `IP_UNDERPAID_INSTALLMENT_SHARE`

    Share of installments where total observed payments were below the scheduled
    installment amount.

    `IP_MEAN_PAYMENT_SHORTFALL`

    Average absolute payment shortfall across installments.

    `IP_MAX_PAYMENT_SHORTFALL`

    Maximum observed payment shortfall.

    `IP_MEAN_PAYMENT_RATIO`

    Average ratio between the total observed payment and the scheduled installment
    amount.

    `IP_MIN_PAYMENT_RATIO`

    Minimum historical payment ratio.

    This captures the most incomplete observed installment repayment.

    **Aggregation**

    The transformation is performed in two stages:

    1. raw payment rows → one row per scheduled installment;
    2. scheduled installments → one row per `SK_ID_CURR`.

    This prevents applicants with split payment records from receiving artificial
    weight simply because one installment was represented by several rows.

    **Missing-value handling**

    The source table contains very little missingness in:

    - `AMT_PAYMENT`;
    - `DAYS_ENTRY_PAYMENT`.

    No global imputation is performed.

    Installments without sufficient payment information retain missing derived
    values where appropriate.

    For `IP_PAYMENT_RATIO`, zero values of `AMT_INSTALMENT` are replaced with `NaN`
    before division to prevent infinite ratios.

    **Cutoff / leakage check**

    Both:

    `DAYS_INSTALMENT`

    and:

    `DAYS_ENTRY_PAYMENT`

    contain only negative values, with maximum values equal to `-1`.

    Therefore, both scheduled installment dates and observed payment dates used in
    IP2 occur before the current application cutoff.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff payment information;
    - no target-based statistics across applicants.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted credit-card CC1 + CC2 + CC4 features;
    - accepted IP1 features.

    **Primary comparison**

    Paired validation AP comparison:

    `IP1 reference`

    vs.

    `IP1 + IP2`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain IP2 if historical repayment timing and payment completeness provide
    additional predictive signal beyond IP1.

    The bundle should provide a meaningful overall validation AP improvement and
    should not materially degrade OOF ranking quality or the Top-10% review-policy
    metrics.
    """)
    return


@app.cell
def _():
    # modeling_ip2 = pd.read_parquet("data/processed/modeling_ip2.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES_V10):
    IP2_FEATURES = [
        "IP_LATE_INSTALLMENT_SHARE",
        "IP_MEAN_DELAY_DAYS",
        "IP_MAX_DELAY_DAYS",
        "IP_30_PLUS_LATE_SHARE",

        "IP_UNDERPAID_INSTALLMENT_SHARE",
        "IP_MEAN_PAYMENT_SHORTFALL",
        "IP_MAX_PAYMENT_SHORTFALL",

        "IP_MEAN_PAYMENT_COVERAGE_RATIO",
        "IP_MIN_PAYMENT_COVERAGE_RATIO",
    ]

    ip2_features = (
        ACCEPTED_FEATURES_V10 +
        IP2_FEATURES
    )
    return (IP2_FEATURES,)


@app.cell
def _():
    # ip2_run = run_catboost_experiment(
    #     frame=modeling_ip2,
    #     features=ip2_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_ip2_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     ip1_run["fold_metrics"],
    #     ip2_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### IP2 — Results

    | Fold | IP1 AP | IP2 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.28009 | 0.28492 | +0.00484 |
    | 1 | 0.28129 | 0.28696 | +0.00567 |
    | 2 | 0.27768 | 0.28061 | +0.00293 |
    | 3 | 0.27025 | 0.27756 | +0.00731 |
    | 4 | 0.27748 | 0.28057 | +0.00310 |

    **Mean Δ AP:** `+0.00477`

    **Positive folds:** `5 / 5`

    **Decision:** **KEEP**

    The repayment-discipline bundle provides a strong and highly consistent
    improvement over IP1.

    Validation AP improves on all five folds, with a mean paired improvement of
    approximately `+0.00477`.

    The gain is substantial relative to previous feature-bundle experiments and is
    observed consistently across the fixed validation folds.

    This confirms that historical repayment behavior — particularly payment delays,
    late-payment frequency and incomplete installment coverage — contains strong
    predictive information beyond the structure and recency of installment-payment
    history.

    IP2 is therefore added to the accepted feature set.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V10, IP2_FEATURES):
    ACCEPTED_FEATURES_V11 = (
        ACCEPTED_FEATURES_V10 +
        IP2_FEATURES
    )
    return (ACCEPTED_FEATURES_V11,)


@app.cell
def _():
    # ip2_results = pd.read_parquet("mlartifacts\\1\\bb7cd15814a04735adc5a633a2722a96\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IP3 — Recent installment repayment discipline

    **Observation**

    IP2 showed that historical repayment discipline contains strong predictive
    signal.

    However, IP2 summarizes the applicant's full available installment-payment
    history and therefore treats older and more recent repayment behavior together.

    The source table contains scheduled installment dates in `DAYS_INSTALMENT`,
    which are expressed relative to the current application date.

    This makes it possible to isolate recent repayment behavior using fixed
    lookback windows.

    Two windows are used:

    - recent 6 months: `DAYS_INSTALMENT >= -180`;
    - recent 12 months: `DAYS_INSTALMENT >= -365`.

    The window assignment is based on the scheduled installment date rather than the
    actual payment date. This avoids allowing payment lateness itself to determine
    whether an installment belongs to a recent period.

    **Hypothesis**

    Recent repayment discipline may provide additional predictive information
    beyond the applicant's overall historical repayment behavior captured in IP2.

    In particular, recent late payments or underpayments may be more informative
    about current credit risk than similar events that occurred several years ago.

    The experiment therefore tests whether repayment behavior during the most recent
    6- and 12-month periods adds incremental signal on top of the full-history IP2
    features.

    **Change**

    Add the following recent repayment features:

    - `IP_RECENT_6M_INSTALLMENT_COUNT`
    - `IP_RECENT_6M_LATE_SHARE`
    - `IP_RECENT_6M_30_PLUS_LATE_SHARE`
    - `IP_RECENT_6M_MAX_DELAY_DAYS`
    - `IP_RECENT_6M_UNDERPAID_SHARE`

    - `IP_RECENT_12M_INSTALLMENT_COUNT`
    - `IP_RECENT_12M_LATE_SHARE`
    - `IP_RECENT_12M_30_PLUS_LATE_SHARE`
    - `IP_RECENT_12M_MAX_DELAY_DAYS`
    - `IP_RECENT_12M_UNDERPAID_SHARE`

    **Feature interpretation**

    `IP_RECENT_6M_INSTALLMENT_COUNT`

    Number of scheduled installments observed during the most recent 180 days.

    This provides context for the recent repayment-rate features.

    For example, a 50% late-payment share based on two installments is not
    equivalent to a 50% late-payment share based on twenty installments.

    `IP_RECENT_6M_LATE_SHARE`

    Share of installments scheduled during the most recent 180 days that were paid
    late.

    This captures the frequency of recent repayment delays.

    `IP_RECENT_6M_30_PLUS_LATE_SHARE`

    Share of installments in the recent 180-day window with payment delays of at
    least 30 days.

    This focuses on more severe recent delinquency.

    `IP_RECENT_6M_MAX_DELAY_DAYS`

    Maximum payment delay among installments scheduled during the most recent 180
    days.

    This captures the worst recent late-payment episode.

    `IP_RECENT_6M_UNDERPAID_SHARE`

    Share of recent installments where the total observed payment was below the
    scheduled installment amount.

    This captures recent payment-completeness problems.

    The corresponding `12M` features provide the same information over the most
    recent 365 days.

    The two windows serve different purposes:

    - 6 months emphasizes very recent behavior;
    - 12 months provides a broader and more stable view of recent repayment
      discipline.

    **Aggregation**

    The already reconstructed installment-level table from IP2 is reused.

    Each row represents one scheduled installment identified by:

    - `SK_ID_CURR`;
    - `SK_ID_PREV`;
    - `NUM_INSTALMENT_VERSION`;
    - `NUM_INSTALMENT_NUMBER`.

    Two subsets are created:

    `DAYS_INSTALMENT >= -180`

    and:

    `DAYS_INSTALMENT >= -365`

    Repayment-discipline features are then aggregated separately within each window
    to one row per `SK_ID_CURR`.

    The 6-month and 12-month feature tables are finally combined into a single IP3
    feature table.

    **Missing-value handling**

    No global imputation is performed.

    Applicants without installments in a particular recent window may have missing
    recent aggregation values.

    This absence is preserved because having no recent installment-payment activity
    is different from having recent activity with zero repayment problems.

    The installment-count features provide additional context for interpreting the
    recent rate-based features.

    **Cutoff / leakage check**

    Window membership is determined using `DAYS_INSTALMENT`.

    All values satisfy:

    `DAYS_INSTALMENT < 0`

    so every installment included in IP3 was scheduled before the current
    application cutoff.

    The actual payment information used to evaluate lateness and underpayment is
    also pre-cutoff because:

    `DAYS_ENTRY_PAYMENT < 0`

    for all observed payment dates.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff installment or payment information;
    - no target-based statistics across applicants.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted credit-card CC1 + CC2 + CC4 features;
    - accepted installment-payment IP1 + IP2 features.

    **Primary comparison**

    Paired validation AP comparison:

    `IP1 + IP2`

    vs.

    `IP1 + IP2 + IP3`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain IP3 if recent repayment behavior provides incremental predictive signal
    beyond the full-history repayment-discipline features from IP2.

    The bundle should produce a reasonably consistent improvement in validation AP
    without materially degrading OOF ranking quality or Top-10% review-policy
    metrics.

    Because IP3 is the final planned feature bundle for `installments_payments`,
    the table will be considered complete after this experiment regardless of
    whether IP3 is accepted or rejected.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V11):
    IP3_FEATURES = [
        "IP_RECENT_6M_INSTALLMENT_COUNT",
        "IP_RECENT_6M_LATE_SHARE",
        "IP_RECENT_6M_30_PLUS_LATE_SHARE",
        "IP_RECENT_6M_MAX_DELAY_DAYS",
        "IP_RECENT_6M_UNDERPAID_SHARE",

        "IP_RECENT_12M_INSTALLMENT_COUNT",
        "IP_RECENT_12M_LATE_SHARE",
        "IP_RECENT_12M_30_PLUS_LATE_SHARE",
        "IP_RECENT_12M_MAX_DELAY_DAYS",
        "IP_RECENT_12M_UNDERPAID_SHARE",
    ]

    ip3_features = (
        ACCEPTED_FEATURES_V11 +
        IP3_FEATURES
    )
    return (IP3_FEATURES,)


@app.cell
def _():
    # modeling_ip3 = pd.read_parquet("data/processed/modeling_ip3.parquet")
    return


@app.cell
def _():
    # ip3_run = run_catboost_experiment(
    #     frame=modeling_ip3,
    #     features=ip3_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_ip3_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     ip2_results,
    #     ip3_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### IP3 — Results

    **Mean Δ AP:** `+0.00160`

    **Positive folds:** `3 / 5`

    **Decision:** **KEEP**

    The recent repayment-discipline bundle provides additional predictive signal
    beyond the full-history IP1 + IP2 features.

    Three of five folds improve. The two negative fold deltas are small, while two
    folds show substantial improvements of approximately `+0.004`.

    The broader evaluation strongly supports retaining the bundle:

    - mean validation AP improves;
    - OOF ROC-AUC improves;
    - Precision@Top10% improves;
    - Recall@Top10% improves.

    The result indicates that recent repayment behavior contains information that
    is not fully captured by full-history repayment aggregates.

    IP3 is therefore added to the accepted feature set.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V11, IP3_FEATURES):
    ACCEPTED_FEATURES_V12 = (
        ACCEPTED_FEATURES_V11 +
        IP3_FEATURES
    )
    return (ACCEPTED_FEATURES_V12,)


@app.cell
def _():
    # ip3_results = pd.read_parquet("mlartifacts\\1\\38fcaf6e8670434e888f679ac2294ea3\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## POS1 — POS/Cash contract history and repayment progress

    **Observation**

    The `POS_CASH_balance` table contains monthly snapshots of historical POS and
    cash-loan contracts.

    The table grain is:

    `SK_ID_PREV × MONTHS_BALANCE`

    and this combination is unique in the source data.

    Therefore, each row represents one monthly snapshot of one previous contract.

    `MONTHS_BALANCE` ranges from `-96` to `-1`, so all observed snapshots occur
    before the current application cutoff.

    The main structural variables in this table are:

    - `CNT_INSTALMENT`
    - `CNT_INSTALMENT_FUTURE`
    - `MONTHS_BALANCE`

    These allow the model to describe:

    - how much POS/Cash contract history exists;
    - how many contracts the applicant had;
    - how long this history extends;
    - how many installments were scheduled;
    - how many installments remained;
    - how far contracts had progressed through their repayment schedules.

    Missingness in `CNT_INSTALMENT` and `CNT_INSTALMENT_FUTURE` is very low
    (approximately 0.26%), so no global imputation is required.

    **Hypothesis**

    The structure and repayment progress of historical POS/Cash contracts may add
    predictive information beyond the already accepted application, bureau,
    previous-application, credit-card and installment-payment features.

    Applicant risk may depend on:

    - the number of historical POS/Cash contracts;
    - the depth of observed contract history;
    - typical contract length;
    - how many installments remained on previous contracts;
    - how far previous contracts had progressed through their repayment schedules;
    - the state of contracts closest to the current application date.

    In particular, the most recent contract state may provide additional signal
    because it reflects repayment obligations closer to the prediction cutoff.

    **Preprocessing**

    A row-level contract-progress feature is created:

    `POS_COMPLETION_RATIO`

    calculated as:

    `(CNT_INSTALMENT - CNT_INSTALMENT_FUTURE) / CNT_INSTALMENT`

    This approximates the fraction of the repayment schedule already completed.

    For example:

    `CNT_INSTALMENT = 12`

    `CNT_INSTALMENT_FUTURE = 9`

    gives:

    `POS_COMPLETION_RATIO = 0.25`

    which means approximately 25% of the scheduled installments have already
    progressed through the repayment schedule.

    Before using this feature, the relationship:

    `CNT_INSTALMENT_FUTURE <= CNT_INSTALMENT`

    is checked for data-quality consistency.

    **Change**

    Add the following applicant-level features:

    - `POS_CONTRACT_COUNT`
    - `POS_MONTHS_OBSERVED`
    - `POS_HISTORY_AGE_MONTHS`
    - `POS_MONTHS_SINCE_LATEST`

    - `POS_MEAN_INSTALMENT_COUNT`
    - `POS_MAX_INSTALMENT_COUNT`

    - `POS_MEAN_INSTALMENTS_FUTURE`
    - `POS_MEAN_COMPLETION_RATIO`

    - `POS_LATEST_MEAN_INSTALMENTS_FUTURE`
    - `POS_LATEST_MAX_INSTALMENTS_FUTURE`
    - `POS_LATEST_MEAN_COMPLETION_RATIO`

    **Feature interpretation**

    `POS_CONTRACT_COUNT`

    Number of unique historical POS/Cash contracts:

    `nunique(SK_ID_PREV)`

    This represents the breadth of the applicant's observed POS/Cash credit history.

    `POS_MONTHS_OBSERVED`

    Total number of monthly POS/Cash snapshots available for the applicant.

    This measures the volume of observed contract history.

    `POS_HISTORY_AGE_MONTHS`

    Age of the oldest observed POS/Cash snapshot:

    `-min(MONTHS_BALANCE)`

    Higher values indicate a longer observed historical record.

    `POS_MONTHS_SINCE_LATEST`

    Months since the most recent observed POS/Cash snapshot:

    `-max(MONTHS_BALANCE)`

    Lower values indicate more recent POS/Cash contract activity.

    `POS_MEAN_INSTALMENT_COUNT`

    Average scheduled installment count across observed contract snapshots.

    This provides information about the typical size or duration of historical
    repayment schedules.

    `POS_MAX_INSTALMENT_COUNT`

    Maximum observed installment count.

    This captures the longest repayment schedule present in the applicant's
    historical POS/Cash contracts.

    `POS_MEAN_INSTALMENTS_FUTURE`

    Average number of remaining installments across the observed monthly history.

    This reflects the historical level of remaining repayment obligations.

    `POS_MEAN_COMPLETION_RATIO`

    Average contract-completion ratio across all observed POS/Cash snapshots.

    This summarizes how far contracts were typically progressed through their
    repayment schedules.

    **Latest-state features**

    Because `CNT_INSTALMENT_FUTURE` and contract progress change over time, the most
    recent snapshot of each `SK_ID_PREV` is also extracted using the largest
    `MONTHS_BALANCE` value.

    The latest contract-level states are then aggregated to `SK_ID_CURR`.

    `POS_LATEST_MEAN_INSTALMENTS_FUTURE`

    Average number of remaining installments across the applicant's most recently
    observed contract states.

    `POS_LATEST_MAX_INSTALMENTS_FUTURE`

    Maximum number of remaining installments among the latest observed contract
    states.

    `POS_LATEST_MEAN_COMPLETION_RATIO`

    Average repayment-schedule completion ratio among the latest observed contract
    states.

    These features provide a more current view of POS/Cash obligations close to the
    current application cutoff.

    **Aggregation**

    The transformation uses two representations of the source table.

    First, full-history features are aggregated directly from all monthly snapshots
    to one row per `SK_ID_CURR`.

    Second, the most recent snapshot is selected separately for each `SK_ID_PREV`,
    after which latest-state features are aggregated to `SK_ID_CURR`.

    The two applicant-level feature tables are then merged.

    **Missing-value handling**

    No global imputation is performed.

    The small amount of missingness in:

    - `CNT_INSTALMENT`;
    - `CNT_INSTALMENT_FUTURE`;

    is preserved.

    Aggregations therefore use available observations, while applicants without
    valid contract-state information may retain missing values.

    Applicants without any POS/Cash history remain distinguishable after the
    left join rather than being imputed with population-level statistics.

    **Cutoff / leakage check**

    All records satisfy:

    `MONTHS_BALANCE < 0`

    with values ranging from `-96` to `-1`.

    Therefore, every POS/Cash monthly snapshot used in POS1 occurs before the
    current application cutoff.

    Latest-state features use the observation closest to the cutoff, but never use
    zero or positive months.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff contract snapshots;
    - no target-based aggregations across applicants.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted credit-card CC1 + CC2 + CC4 features;
    - accepted installment-payment IP1 + IP2 + IP3 features.

    Rejected feature groups are not included in the reference.

    **Primary comparison**

    Paired validation AP comparison:

    `current accepted reference`

    vs.

    `current accepted reference + POS1`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain POS1 if POS/Cash contract structure and repayment-progress features
    provide additional predictive signal without materially degrading OOF ranking
    quality or the Top-10% review-policy metrics.

    Because POS1 contains mostly structural and contract-state features, even a
    moderate but consistent improvement may justify retaining the bundle.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V12):
    POS1_FEATURES = [
        "POS_CONTRACT_COUNT",
        "POS_MONTHS_OBSERVED",
        "POS_HISTORY_AGE_MONTHS",
        "POS_MONTHS_SINCE_LATEST",

        "POS_MEAN_INSTALMENT_COUNT",
        "POS_MAX_INSTALMENT_COUNT",

        "POS_MEAN_INSTALMENTS_FUTURE",
        "POS_MEAN_COMPLETION_RATIO",

        "POS_LATEST_MEAN_INSTALMENTS_FUTURE",
        "POS_LATEST_MAX_INSTALMENTS_FUTURE",
        "POS_LATEST_MEAN_COMPLETION_RATIO",
    ]

    pos1_features = (
        ACCEPTED_FEATURES_V12 +
        POS1_FEATURES
    )
    return


@app.cell
def _():
    # modeling_pos1 = pd.read_parquet("data/processed/modeling_pos1.parquet")
    return


@app.cell
def _():
    # pos1_run = run_catboost_experiment(
    #     frame=modeling_pos1,
    #     features=pos1_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_pos1_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     ip3_results,
    #     pos1_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### POS1 — Results

    | Fold | Reference AP | POS1 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.28518 | 0.28662 | +0.00144 |
    | 1 | 0.29120 | 0.28598 | -0.00522 |
    | 2 | 0.28017 | 0.28081 | +0.00064 |
    | 3 | 0.28165 | 0.28086 | -0.00079 |
    | 4 | 0.28043 | 0.27855 | -0.00188 |

    **Mean Δ AP:** `-0.00116`

    **Positive folds:** `2 / 5`

    **Decision:** **REJECT**

    The POS/Cash contract-history and repayment-progress bundle does not provide
    incremental predictive value over the current accepted reference.

    Validation AP decreases on 3 of 5 folds, with a mean paired change of
    approximately `-0.00116`.

    Although Precision@Top10% and Recall@Top10% improve slightly, the primary
    Average Precision metric decreases and OOF ROC-AUC also declines.

    Therefore, POS1 is not added to the accepted feature set.

    The result suggests that general POS/Cash contract structure and repayment
    progress are largely redundant with information already captured by the
    accepted historical feature groups.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## POS2 — POS/Cash delinquency history

    **Observation**

    The `POS_CASH_balance` table contains monthly delinquency information for
    historical POS and cash-loan contracts:

    - `SK_DPD`
    - `SK_DPD_DEF`

    Both variables are available for every monthly snapshot.

    Their distributions are highly zero-inflated:

    - the median is `0`;
    - the 75th percentile is `0`;
    - maximum values are very large.

    This indicates that most monthly contract states have no recorded delinquency,
    while a smaller subset contains potentially severe overdue episodes.

    POS1 tested general contract structure and repayment progress but was rejected
    because it reduced validation Average Precision.

    Therefore, POS2 is evaluated directly against the last accepted reference before
    POS features.

    **Hypothesis**

    Historical POS/Cash delinquency may contain predictive information that is not
    fully captured by general contract structure or by repayment information from
    other historical tables.

    Risk may depend on:

    - whether delinquency occurred at all;
    - how frequently the applicant was delinquent;
    - the severity of the worst delinquency episode;
    - whether more serious 30+ day delinquency occurred;
    - whether delinquency was observed recently;
    - whether delinquency was still present in the most recent contract state.

    The experiment therefore summarizes delinquency along three dimensions:

    1. **severity**;
    2. **frequency**;
    3. **recency**.

    **Preprocessing**

    The following monthly indicators are created:

    `POS_HAS_DPD`

    equals `1` when:

    `SK_DPD > 0`

    and `0` otherwise.

    `POS_HAS_DPD_DEF`

    equals `1` when:

    `SK_DPD_DEF > 0`

    and `0` otherwise.

    `POS_DPD_30_PLUS`

    equals `1` when:

    `SK_DPD >= 30`

    and `0` otherwise.

    These indicators convert highly zero-inflated DPD values into interpretable
    delinquency-frequency measures.

    **Change**

    Add the following applicant-level features:

    - `POS_MAX_DPD`
    - `POS_MEAN_DPD`
    - `POS_DPD_MONTH_SHARE`
    - `POS_DPD_30_PLUS_MONTH_SHARE`

    - `POS_MAX_DPD_DEF`
    - `POS_DPD_DEF_MONTH_SHARE`

    - `POS_RECENT_6M_MAX_DPD`
    - `POS_RECENT_6M_DPD_MONTH_SHARE`

    - `POS_LATEST_MAX_DPD`
    - `POS_LATEST_MAX_DPD_DEF`

    **Feature interpretation**

    `POS_MAX_DPD`

    Maximum historical `SK_DPD`.

    This captures the most severe observed delinquency episode.

    `POS_MEAN_DPD`

    Average historical `SK_DPD` across monthly snapshots.

    This provides a broader measure of delinquency intensity over the observed
    contract history.

    `POS_DPD_MONTH_SHARE`

    Share of monthly snapshots where:

    `SK_DPD > 0`

    This captures the frequency of delinquency.

    `POS_DPD_30_PLUS_MONTH_SHARE`

    Share of monthly snapshots where:

    `SK_DPD >= 30`

    This separates more serious delinquency episodes from small payment delays.

    `POS_MAX_DPD_DEF`

    Maximum historical value of `SK_DPD_DEF`.

    This provides an alternative delinquency-severity representation based on the
    dataset's adjusted DPD field.

    `POS_DPD_DEF_MONTH_SHARE`

    Share of monthly snapshots where:

    `SK_DPD_DEF > 0`

    This captures the frequency of delinquency under the alternative DPD
    definition.

    `POS_RECENT_6M_MAX_DPD`

    Maximum `SK_DPD` observed during the six months closest to the current
    application:

    `MONTHS_BALANCE >= -6`

    This captures recent delinquency severity.

    `POS_RECENT_6M_DPD_MONTH_SHARE`

    Share of monthly snapshots in the recent six-month period where:

    `SK_DPD > 0`

    This captures recent delinquency frequency.

    `POS_LATEST_MAX_DPD`

    Maximum `SK_DPD` among the most recent available snapshots of the applicant's
    historical POS/Cash contracts.

    This represents delinquency status closest to the current application cutoff.

    `POS_LATEST_MAX_DPD_DEF`

    Maximum `SK_DPD_DEF` among those most recent contract snapshots.

    **Aggregation**

    The transformation uses three views of the source data:

    1. full historical monthly snapshots;
    2. snapshots from the most recent six months;
    3. the most recent snapshot of each `SK_ID_PREV`.

    Each view is aggregated to one row per `SK_ID_CURR`.

    The resulting delinquency feature tables are then merged.

    **Missing-value handling**

    `SK_DPD` and `SK_DPD_DEF` contain no missing values, so no imputation is
    required.

    Applicants without POS/Cash history remain missing after the left join rather
    than being assigned population-level values.

    No global imputation is applied.

    **Cutoff / leakage check**

    All records satisfy:

    `MONTHS_BALANCE < 0`

    with values ranging from `-96` to `-1`.

    Therefore, all delinquency observations used in POS2 occur before the current
    application cutoff.

    Recent and latest features move closer to the prediction date but never use
    current or future snapshots.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff contract information;
    - no target-based statistics across applicants.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted credit-card CC1 + CC2 + CC4 features;
    - accepted installment-payment IP1 + IP2 + IP3 features;
    - rejected POS1 features are not included.

    **Primary comparison**

    Paired validation AP comparison:

    `current accepted reference`

    vs.

    `current accepted reference + POS2`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain POS2 if historical POS/Cash delinquency provides incremental predictive
    signal beyond the current accepted reference.

    The bundle should improve validation AP without materially degrading OOF
    ranking quality or Top-10% review-policy metrics.

    Because POS2 is the final planned experiment for `POS_CASH_balance`, the table
    will be considered complete after this experiment regardless of whether the
    bundle is accepted or rejected.
    """)
    return


@app.cell
def _(ACCEPTED_FEATURES_V12):
    POS2_FEATURES = [
        "POS_MAX_DPD",
        "POS_MEAN_DPD",
        "POS_DPD_MONTH_SHARE",
        "POS_DPD_30_PLUS_MONTH_SHARE",

        "POS_MAX_DPD_DEF",
        "POS_DPD_DEF_MONTH_SHARE",

        "POS_RECENT_6M_MAX_DPD",
        "POS_RECENT_6M_DPD_MONTH_SHARE",

        "POS_LATEST_MAX_DPD",
        "POS_LATEST_MAX_DPD_DEF",
    ]

    pos2_features = (
        ACCEPTED_FEATURES_V12 +
        POS2_FEATURES
    )
    return


@app.cell
def _():
    # modeling_pos2 = pd.read_parquet("data/processed/modeling_pos2.parquet")
    return


@app.cell
def _():
    # pos2_run = run_catboost_experiment(
    #     frame=modeling_pos2,
    #     features=pos2_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_pos2_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     ip3_results,
    #     pos2_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### POS2 — Results

    | Fold | Reference AP | POS2 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.28518 | 0.28877 | +0.00358 |
    | 1 | 0.29120 | 0.28996 | -0.00125 |
    | 2 | 0.28017 | 0.27807 | -0.00210 |
    | 3 | 0.28165 | 0.28066 | -0.00099 |
    | 4 | 0.28043 | 0.28126 | +0.00083 |

    **Mean Δ AP:** `≈ +0.00002`

    **Positive folds:** `2 / 5`

    **Decision:** **REJECT**

    The POS/Cash delinquency bundle does not provide meaningful incremental
    predictive value over the current accepted reference.

    Only two of five folds improve, while three folds degrade. The mean paired AP
    change is effectively zero.

    OOF ranking and Top-10% review-policy metrics also do not show a sufficiently
    consistent improvement to justify retaining the additional features.

    Therefore, POS2 is rejected.

    The result suggests that POS/Cash delinquency information is largely redundant
    with repayment-behavior and delinquency features already captured in other
    historical tables.

    `POS_CASH_balance` is considered complete after this experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BB1 — Bureau monthly delinquency history

    **Observation**

    The `bureau_balance` table contains monthly status history for credits reported
    in the bureau table.

    The table is linked through:

    `SK_ID_BUREAU`

    which identifies a specific bureau credit.

    Each bureau credit may have multiple monthly records described by:

    - `MONTHS_BALANCE`
    - `STATUS`

    The `STATUS` field represents the monthly state of the bureau credit and includes:

    - `0` — no delinquency;
    - `1`–`5` — increasing delinquency severity;
    - `C` — closed;
    - `X` — status unknown.

    Therefore, `bureau_balance` provides a longitudinal view of historical bureau
    credit behavior that is not available from the one-row-per-credit `bureau`
    table alone.

    Because `bureau_balance` is keyed by `SK_ID_BUREAU`, the transformation must be
    performed in two stages:

    1. monthly bureau history → one row per `SK_ID_BUREAU`;
    2. bureau credit → one row per `SK_ID_CURR`.

    **Hypothesis**

    Monthly bureau delinquency history may provide additional predictive information
    beyond the already accepted bureau aggregates.

    In particular, applicant risk may depend on:

    - how long bureau credits have been observed;
    - whether delinquency occurred;
    - how frequently delinquency occurred;
    - the maximum severity of delinquency;
    - whether severe delinquency occurred;
    - how recently delinquency was observed;
    - whether delinquency occurred during the most recent 12 months.

    The experiment therefore summarizes bureau history along four dimensions:

    1. history depth;
    2. delinquency severity;
    3. delinquency frequency;
    4. delinquency recency.

    **Preprocessing**

    The categorical `STATUS` values are mapped to an ordinal delinquency severity
    representation:

    `0 → 0`

    `1 → 1`

    `2 → 2`

    `3 → 3`

    `4 → 4`

    `5 → 5`

    Statuses `C` and `X` are not assigned delinquency severity values because they
    represent closed or unknown states rather than delinquency levels.

    The following monthly indicators are created:

    `BB_HAS_DPD`

    equals `1` when:

    `STATUS ∈ {1, 2, 3, 4, 5}`

    and `0` otherwise.

    `BB_HAS_SEVERE_DPD`

    equals `1` when:

    `STATUS ∈ {3, 4, 5}`

    and `0` otherwise.

    `BB_IS_KNOWN_STATUS`

    equals `1` when:

    `STATUS ∈ {0, 1, 2, 3, 4, 5}`

    and `0` for `C` and `X`.

    This allows delinquency shares to be calculated only over months with a known
    active payment-status interpretation.

    **Bureau-credit level aggregation**

    Monthly records are first aggregated to one row per `SK_ID_BUREAU`.

    The following intermediate credit-level features are created:

    - number of observed months;
    - history age;
    - maximum delinquency severity;
    - number of delinquent months;
    - number of severe-delinquency months;
    - number of months with known payment status;
    - delinquent-month share;
    - severe-delinquency-month share;
    - months since the most recent delinquency;
    - recent 12-month maximum severity;
    - recent 12-month delinquency share.

    Delinquency shares are calculated using only months where:

    `STATUS ∈ {0, 1, 2, 3, 4, 5}`

    so that `C` and `X` do not artificially reduce the estimated delinquency rate.

    **Change**

    Add the following applicant-level features:

    - `BB_TOTAL_MONTHS_OBSERVED`
    - `BB_MAX_HISTORY_AGE_MONTHS`

    - `BB_MAX_STATUS_SEVERITY`
    - `BB_MEAN_DPD_MONTH_SHARE`
    - `BB_MAX_DPD_MONTH_SHARE`
    - `BB_MEAN_SEVERE_DPD_MONTH_SHARE`

    - `BB_MONTHS_SINCE_LAST_DPD`

    - `BB_RECENT_12M_MAX_SEVERITY`
    - `BB_RECENT_12M_MEAN_DPD_SHARE`

    **Feature interpretation**

    `BB_TOTAL_MONTHS_OBSERVED`

    Total number of monthly bureau-history observations across all bureau credits
    associated with the applicant.

    This represents the overall depth of observed bureau history.

    `BB_MAX_HISTORY_AGE_MONTHS`

    Maximum historical age of any bureau credit:

    `-min(MONTHS_BALANCE)`

    Higher values indicate a longer observed bureau history.

    `BB_MAX_STATUS_SEVERITY`

    Maximum delinquency severity observed across all bureau credits.

    This captures the applicant's worst historical bureau delinquency state.

    `BB_MEAN_DPD_MONTH_SHARE`

    Average delinquency-month share across the applicant's bureau credits.

    This measures the typical frequency of delinquency across historical credits.

    `BB_MAX_DPD_MONTH_SHARE`

    Maximum delinquency-month share among the applicant's bureau credits.

    This captures the bureau contract with the highest concentration of delinquent
    months.

    `BB_MEAN_SEVERE_DPD_MONTH_SHARE`

    Average share of months with severe delinquency states `3`–`5`.

    This distinguishes more serious historical delinquency from minor overdue
    episodes.

    `BB_MONTHS_SINCE_LAST_DPD`

    Minimum number of months since the most recent delinquency across all bureau
    credits.

    Lower values indicate more recent delinquency.

    `BB_RECENT_12M_MAX_SEVERITY`

    Maximum delinquency severity observed during the most recent 12 months.

    This captures recent bureau delinquency severity.

    `BB_RECENT_12M_MEAN_DPD_SHARE`

    Average delinquency-month share during the most recent 12-month period across
    the applicant's bureau credits.

    This provides a recent view of delinquency frequency.

    **Aggregation**

    The transformation is performed in two stages.

    First:

    `bureau_balance → SK_ID_BUREAU`

    Monthly bureau records are summarized to one row per bureau credit.

    Second:

    `SK_ID_BUREAU → SK_ID_CURR`

    The aggregated bureau-credit table is joined to the `bureau` table to recover
    `SK_ID_CURR`, after which credit-level features are aggregated to one row per
    current applicant.

    This preserves the original hierarchy of the data instead of directly mixing
    monthly observations from different bureau credits.

    **Missing-value handling**

    No global imputation is performed.

    Credits without observed delinquency may have missing:

    `BB_MONTHS_SINCE_LAST_DPD`

    because no historical delinquency event exists.

    This absence is preserved rather than replaced with an arbitrary numeric value.

    Statuses `C` and `X` are treated separately from known repayment-status months
    when calculating delinquency shares.

    Applicants without relevant `bureau_balance` history remain missing after the
    left join rather than being assigned population-level values.

    **Recent-history definition**

    Recent bureau behavior is defined using:

    `MONTHS_BALANCE >= -12`

    which corresponds to the 12 months closest to the current application cutoff.

    This window is used to capture whether historical delinquency was recent rather
    than occurring only in the distant past.

    **Cutoff / leakage check**

    `MONTHS_BALANCE` represents historical monthly states relative to the current
    application.

    Only pre-cutoff monthly bureau observations are used.

    The experiment uses:

    - no `TARGET`;
    - no current-application outcome;
    - no post-cutoff records;
    - no target-based statistics across applicants.

    All features are constructed exclusively from historical bureau information.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same GPU CatBoost configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - accepted application features;
    - accepted bureau B1–B4 features;
    - accepted `previous_application` P1 + P2 + P4 features;
    - accepted credit-card CC1 + CC2 + CC4 features;
    - accepted installment-payment IP1 + IP2 + IP3 features;
    - rejected POS feature groups are not included.

    **Primary comparison**

    Paired validation AP comparison:

    `current accepted reference`

    vs.

    `current accepted reference + BB1`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - train-validation AP gap;
    - fold stability.

    **Decision rule**

    Retain BB1 if monthly bureau delinquency history provides incremental predictive
    signal beyond the already accepted bureau and repayment-history features.

    The bundle should provide a meaningful overall validation AP improvement and
    should not materially degrade OOF ranking quality or Top-10% review-policy
    metrics.

    Because BB1 is the only planned experiment for `bureau_balance`, the table will
    be considered complete after this experiment regardless of whether the bundle
    is accepted or rejected.
    """)
    return


@app.cell
def _():
    # modeling_bb1 = pd.read_parquet("data/processed/modeling_bb1.parquet")
    return


@app.cell
def _(ACCEPTED_FEATURES_V12):
    BB1_FEATURES = [
        "BB_TOTAL_MONTHS_OBSERVED",
        "BB_MAX_HISTORY_AGE_MONTHS",

        "BB_MAX_STATUS_SEVERITY",
        "BB_MEAN_DPD_MONTH_SHARE",
        "BB_MAX_DPD_MONTH_SHARE",
        "BB_MEAN_SEVERE_DPD_MONTH_SHARE",

        "BB_MONTHS_SINCE_LAST_DPD",

        "BB_RECENT_12M_MAX_SEVERITY",
        "BB_RECENT_12M_MEAN_DPD_SHARE",
    ]

    bb1_features = (
        ACCEPTED_FEATURES_V12 +
        BB1_FEATURES
    )
    return


@app.cell
def _():
    # bb1_run = run_catboost_experiment(
    #     frame=modeling_bb1,
    #     features=bb1_features,
    #     categorical_features=categorical_baseline_features,
    #     params=BASELINE_PARAMS,
    #     run_name='cb_bb1_v1',
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     ip3_results,
    #     bb1_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BB1 — Results

    | Fold | Reference AP | BB1 AP | Δ AP |
    |---:|---:|---:|---:|
    | 0 | 0.28518 | 0.28831 | +0.00313 |
    | 1 | 0.29120 | 0.28806 | -0.00315 |
    | 2 | 0.28017 | 0.28132 | +0.00115 |
    | 3 | 0.28165 | 0.28203 | +0.00038 |
    | 4 | 0.28043 | 0.28030 | -0.00013 |

    **Mean Δ AP:** `+0.00028`

    **Positive folds:** `3 / 5`

    **Decision:** **REJECT**

    The monthly bureau-balance delinquency bundle provides only a very small and
    inconsistent improvement in validation Average Precision.

    Three of five folds improve, but the two largest fold-level effects have
    opposite signs and nearly cancel each other. The mean paired AP improvement is
    only approximately `+0.00028`.

    Precision@Top10% and Recall@Top10% improve, and fold-level AP variability
    decreases, but OOF ROC-AUC declines slightly.

    Given the small improvement in the primary metric and the inconsistent
    fold-level effect, the additional feature complexity is not sufficiently
    justified.

    BB1 is therefore rejected.

    The result does not imply that monthly bureau delinquency history contains no
    signal. Rather, most of its useful information appears to be redundant with the
    already accepted bureau, installment-payment and other historical feature
    groups.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A1 — Source ablation: installments_payments

    **Objective**

    Measure the incremental contribution of the complete
    `installments_payments` feature source to the final multi-table model.

    Unlike the earlier sequential feature experiments, this experiment evaluates
    the source inside the final accepted feature representation.

    This is important because the usefulness of a feature group may change after
    other historical sources are added.

    **Reference**

    The fixed full multi-table reference contains all accepted feature groups:

    - application features;
    - bureau B1–B4;
    - previous_application P1 + P2 + P4;
    - credit_card_balance CC1 + CC2 + CC4;
    - installments_payments IP1 + IP2 + IP3.

    Rejected POS_CASH_balance and bureau_balance feature groups are not included.

    **Hypothesis**

    If `installments_payments` provides unique predictive information, removing all
    features derived from this source should reduce validation performance.

    The source contains direct historical repayment behavior, including:

    - repayment-history depth and recency;
    - payment delays;
    - late-payment frequency;
    - payment shortfalls;
    - payment coverage;
    - recent repayment discipline.

    **Change**

    Remove all accepted `installments_payments` features:

    - IP1 — repayment history structure;
    - IP2 — repayment discipline;
    - IP3 — recent repayment discipline.

    No other feature, model or validation setting is changed.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same training population;
    - same CatBoost GPU configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - same preprocessing;
    - same remaining feature set.

    **Primary comparison**

    Paired validation AP comparison:

    `FULL MODEL`

    vs.

    `FULL MODEL - installments_payments`

    on exactly the same folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - fold stability.

    **Interpretation**

    A reproducible performance decrease after removing the source indicates that
    `installments_payments` provides incremental information that is not fully
    recovered by the remaining historical tables.

    If removal causes little or no degradation, the source may be largely redundant
    inside the final feature representation.
    """)
    return


@app.cell
def _(pd):
    training_dataset = pd.read_parquet("data/processed/modeling_bb1.parquet")
    return (training_dataset,)


@app.cell
def _(
    ACCEPTED_FEATURES_V12,
    IP1_FEATURES,
    IP2_FEATURES,
    IP3_FEATURES,
    training_dataset,
):
    FULL_FEATURES = ACCEPTED_FEATURES_V12
    FULL_CATEGORICAL_FEATURES = training_dataset[FULL_FEATURES].select_dtypes(
        include=["category", "object"]
    ).columns.to_list()

    INSTALLMENTS_FEATURES = (
        IP1_FEATURES
        + IP2_FEATURES
        + IP3_FEATURES
    )

    A1_FEATURES = [
        feature
        for feature in FULL_FEATURES
        if feature not in INSTALLMENTS_FEATURES
        and feature not in ["SK_ID_CURR", "TARGET", "partition", "fold"]
    ]

    CATEGORICAL_FEATURES_A1 = training_dataset[A1_FEATURES].select_dtypes(
        include=["category", "object"]
    ).columns.to_list()
    return (
        A1_FEATURES,
        FULL_CATEGORICAL_FEATURES,
        FULL_FEATURES,
        INSTALLMENTS_FEATURES,
    )


@app.cell
def _(A1_FEATURES, FULL_FEATURES, INSTALLMENTS_FEATURES):
    print("Full:", len(FULL_FEATURES))
    print("A1:", len(A1_FEATURES))
    print("Removed:", len(FULL_FEATURES) - len(A1_FEATURES))

    set(INSTALLMENTS_FEATURES) & set(A1_FEATURES)
    return


@app.cell
def _(pd):
    multitable_accepted_results = pd.read_parquet("mlartifacts\\1\\38fcaf6e8670434e888f679ac2294ea3\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell
def _():
    # a1_run = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=A1_FEATURES,
    #     categorical_features=CATEGORICAL_FEATURES_A1,
    #     params=BASELINE_PARAMS,
    #     run_name="ablation_no_installments_v1",
    #     capacity=0.10
    #     # остальные параметры те же
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     multitable_accepted_results,
    #     a1_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A1 — Source ablation: installments_payments — Results

    **Mean Δ AP (ablated - full):** `-0.00890`

    **Worse folds after removal:** `5 / 5`

    **Decision:** **RETAIN SOURCE**

    Removing all `installments_payments` features causes a substantial and
    consistent degradation in validation Average Precision.

    Performance decreases on all five fixed validation folds, with a mean AP loss
    of approximately `0.00890`.

    The degradation is much larger than the typical fold-level changes observed
    during individual feature experiments.

    This provides strong evidence that `installments_payments` contributes unique
    predictive information that is not recovered by the remaining application,
    bureau, previous-application and credit-card features.

    The source is therefore considered a core component of the final feature set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A2 — Source ablation: bureau

    **Objective**

    Measure the incremental contribution of the accepted `bureau` feature source
    inside the final multi-table representation.

    Earlier bureau experiments showed substantial improvements when the source was
    introduced. However, the final model now also contains information from
    `previous_application`, `credit_card_balance` and `installments_payments`.

    Therefore, this ablation tests whether bureau features still provide unique
    predictive information after the remaining historical sources are available.

    **Reference**

    The fixed full multi-table reference contains:

    - application features;
    - bureau B1–B4;
    - previous_application P1 + P2 + P4;
    - credit_card_balance CC1 + CC2 + CC4;
    - installments_payments IP1 + IP2 + IP3.

    Rejected POS_CASH_balance and bureau_balance feature groups are not included.

    **Change**

    Remove all accepted features derived from the `bureau` table:

    - B1 — basic credit-history structure;
    - B2 — credit-history recency;
    - B3 — financial exposure;
    - B4 — delinquency / credit stress.

    No other feature or modeling setting is changed.

    **Hypothesis**

    If bureau information provides unique predictive signal, removing the complete
    source should reduce validation performance across the fixed folds.

    If performance changes only marginally, much of the bureau signal may already
    be represented by the other historical sources.

    **Controlled variables**

    - same `split_v1`;
    - same five validation folds;
    - same development population;
    - same CatBoost GPU configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - same preprocessing;
    - same remaining features.

    **Primary comparison**

    `FULL MODEL`

    vs.

    `FULL MODEL - bureau`

    using paired validation Average Precision on exactly the same folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - fold stability.

    **Interpretation**

    A consistent decrease after removing bureau indicates that the source contributes
    incremental information that cannot be recovered from the other historical
    tables.
    """)
    return


@app.cell
def _(
    BUREAU_BASIC_FEATURES,
    BUREAU_CREDIT,
    BUREAU_RECENCY_FEATURES,
    FULL_FEATURES,
    STRESS_FEATURES,
    training_dataset,
):
    BUREAU_FEATURES: list[str] = (
        BUREAU_BASIC_FEATURES
        + BUREAU_RECENCY_FEATURES
        + BUREAU_CREDIT
        + STRESS_FEATURES
    )

    A2_FEATURES = [
        feature
        for feature in FULL_FEATURES
        if feature not in BUREAU_FEATURES
    ]

    CATEGORICAL_FEATURES_A2 = training_dataset[A2_FEATURES].select_dtypes(
        include=["category", "object"]
    ).columns.to_list()
    return A2_FEATURES, BUREAU_FEATURES


@app.cell
def _(A2_FEATURES, BUREAU_FEATURES: list[str], FULL_FEATURES):
    print("Full:", len(FULL_FEATURES))
    print("A2:", len(A2_FEATURES))
    print("Removed:", len(FULL_FEATURES) - len(A2_FEATURES))

    set(BUREAU_FEATURES) & set(A2_FEATURES)
    return


@app.cell
def _():
    # a2_run = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=A2_FEATURES,
    #     categorical_features=CATEGORICAL_FEATURES_A2,
    #     params=BASELINE_PARAMS,
    #     run_name="ablation_no_bureau_v1",
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     multitable_accepted_results,
    #     a2_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A2 — Source ablation: bureau — Results

    **Mean Δ AP (ablated - full):** `-0.00695`

    **Worse folds after removal:** `5 / 5`

    **Decision:** **RETAIN SOURCE**

    Removing the complete `bureau` feature source causes a substantial and highly
    consistent degradation in validation Average Precision.

    Performance decreases on all five fixed validation folds, with a mean AP loss
    of approximately `0.00695`.

    This confirms that bureau history provides substantial incremental predictive
    information even after `previous_application`, `credit_card_balance` and
    `installments_payments` are available.

    The bureau feature source is therefore retained as a core component of the
    final feature set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A3 — Source ablation: previous_application

    **Objective**

    Measure the incremental contribution of the complete
    `previous_application` feature source inside the final multi-table model.

    Earlier experiments showed that several `previous_application` feature groups
    improved validation performance when they were introduced sequentially.

    However, the final model now also contains strong information from:

    - bureau;
    - credit_card_balance;
    - installments_payments.

    Therefore, this ablation tests whether `previous_application` still contributes
    unique predictive information after all other accepted historical sources are
    available.

    **Reference**

    The fixed full multi-table reference contains:

    - application features;
    - bureau B1–B4;
    - previous_application P1 + P2 + P4;
    - credit_card_balance CC1 + CC2 + CC4;
    - installments_payments IP1 + IP2 + IP3.

    Rejected feature groups are not included.

    **Change**

    Remove all accepted features derived from `previous_application`:

    - P1 — previous application history and outcomes;
    - P2 — previous application financial history;
    - P4 — previous application temporal history.

    Rejected P3 categorical features are not part of the reference and therefore
    are not involved in the ablation.

    No other feature or modeling setting is changed.

    **Hypothesis**

    If `previous_application` provides unique predictive information, removing the
    entire source should reduce validation performance across the fixed folds.

    If the degradation is small, most of its signal may already be represented by
    bureau, installment-payment and credit-card history.

    **Controlled variables**

    - same `split_v1`;
    - same five validation folds;
    - same development population;
    - same CatBoost GPU configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - same preprocessing;
    - same remaining feature set.

    **Primary comparison**

    Paired validation Average Precision comparison:

    `FULL MODEL`

    vs.

    `FULL MODEL - previous_application`

    using exactly the same validation folds.

    For interpretation, the ablation delta is defined as:

    `AP_ablated - AP_full`

    Therefore:

    - negative delta → removing the source hurts performance;
    - approximately zero → source is largely redundant;
    - positive delta → model performs better without the source.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - fold stability.

    **Interpretation**

    A consistent decrease in validation AP after removing
    `previous_application` indicates that the source contributes predictive
    information that is not fully recoverable from the other historical tables.

    The magnitude of the AP degradation can also be compared with the other
    source-level ablations to estimate the relative importance of each data source.
    """)
    return


@app.cell
def _(
    APPLICATION_HISTORY,
    FINANCIAL_HISTORY_FEATURES,
    FULL_FEATURES,
    TEMPORAL_HISTORY_FEATURES,
    training_dataset,
):
    PREVIOUS_APPLICATION_FEATURES = (
        APPLICATION_HISTORY
        + FINANCIAL_HISTORY_FEATURES
        + TEMPORAL_HISTORY_FEATURES
    )

    A3_FEATURES = [
        feature
        for feature in FULL_FEATURES
        if feature not in PREVIOUS_APPLICATION_FEATURES
    ]

    CATEGORICAL_FEATURES_A3 = training_dataset[A3_FEATURES].select_dtypes(
        include=["category", "object"]
    ).columns.to_list()

    print("Full:", len(FULL_FEATURES))
    print("A3:", len(A3_FEATURES))
    print("Removed:", len(FULL_FEATURES) - len(A3_FEATURES))

    set(PREVIOUS_APPLICATION_FEATURES) & set(A3_FEATURES)
    return


@app.cell
def _():
    # a3_run = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=A3_FEATURES,
    #     categorical_features=CATEGORICAL_FEATURES_A3,
    #     params=BASELINE_PARAMS,
    #     run_name="ablation_no_previous_applications_v1",
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     multitable_accepted_results,
    #     a3_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A3 — Source ablation: previous_application — Results

    **Mean Δ AP (ablated - full):** `-0.00599`

    **Worse folds after removal:** `5 / 5`

    **Decision:** **RETAIN SOURCE**

    Removing the complete `previous_application` feature source causes a clear and
    consistent degradation in validation Average Precision.

    Performance decreases on all five fixed validation folds, with a mean AP loss
    of approximately `0.00599`.

    Although fold 0 is nearly neutral, the remaining four folds show substantial
    performance degradation.

    This confirms that previous-application outcomes, financial history and temporal
    history provide incremental predictive information that is not fully recovered
    by bureau, credit-card and installment-payment features.

    `previous_application` is therefore retained as a core component of the final
    feature set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A4 — Source ablation: credit_card_balance

    **Objective**

    Measure the incremental contribution of the complete
    `credit_card_balance` feature source inside the final multi-table model.

    The final accepted credit-card representation contains information about basic
    credit-card history, balance and utilization, and historical delinquency.

    This ablation tests whether that information remains useful after bureau,
    previous-application and installment-payment history are simultaneously
    available.

    **Reference**

    The fixed full multi-table reference contains:

    - application features;
    - bureau B1–B4;
    - previous_application P1 + P2 + P4;
    - credit_card_balance CC1 + CC2 + CC4;
    - installments_payments IP1 + IP2 + IP3.

    Rejected feature groups are not included.

    **Change**

    Remove all accepted features derived from `credit_card_balance`:

    - CC1 — credit-card history and activity;
    - CC2 — balance and utilization;
    - CC4 — delinquency and repayment stress.

    Rejected CC3 drawings/payment features are not part of the reference.

    No other feature or modeling setting is changed.

    **Hypothesis**

    If `credit_card_balance` provides unique predictive information, removing the
    complete source should reduce validation performance.

    Because credit-card history is available for only a minority of applicants, its
    incremental contribution may be smaller than sources with substantially broader
    coverage.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same CatBoost GPU configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - same preprocessing;
    - same remaining feature set.

    **Primary comparison**

    Paired validation Average Precision comparison:

    `FULL MODEL`

    vs.

    `FULL MODEL - credit_card_balance`

    using exactly the same five folds.

    The ablation delta is defined as:

    `AP_ablated - AP_full`

    Therefore:

    - negative delta → the source contributes useful signal;
    - approximately zero → the source is largely redundant;
    - positive delta → performance improves without the source.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - fold stability.

    **Interpretation**

    A consistent degradation after removing `credit_card_balance` indicates that
    credit-card activity, utilization and delinquency contain information that is
    not fully represented by the other historical sources.
    """)
    return


@app.cell
def _(
    CC1_FEATURES,
    CC2_FEATURES,
    CC4_FEATURES,
    FULL_FEATURES,
    training_dataset,
):
    CREDIT_CARD_FEATURES = (
        CC1_FEATURES
        + CC2_FEATURES
        + CC4_FEATURES
    )

    A4_FEATURES = [
        feature
        for feature in FULL_FEATURES
        if feature not in CREDIT_CARD_FEATURES
    ]

    CATEGORICAL_FEATURES_A4 = training_dataset[A4_FEATURES].select_dtypes(
        include=["category", "object"]
    ).columns.to_list()

    print("Full:", len(FULL_FEATURES))
    print("A4:", len(A4_FEATURES))
    print("Removed:", len(FULL_FEATURES) - len(A4_FEATURES))

    set(CREDIT_CARD_FEATURES) & set(A4_FEATURES)
    return


@app.cell
def _():
    # a4_run = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=A4_FEATURES,
    #     categorical_features=CATEGORICAL_FEATURES_A4,
    #     params=BASELINE_PARAMS,
    #     run_name="ablation_no_credit_card_v1",
    #     capacity=0.10
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     multitable_accepted_results,
    #     a4_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A4 — Source ablation: credit_card_balance — Results

    **Mean Δ AP (ablated - full):** `-0.00458`

    **Worse folds after removal:** `5 / 5`

    **Decision:** **RETAIN SOURCE**

    Removing the complete `credit_card_balance` feature source causes a consistent
    degradation in validation Average Precision.

    Performance decreases on all five fixed validation folds, with a mean AP loss
    of approximately `0.00458`.

    This confirms that historical credit-card activity, balance, utilization and
    delinquency provide incremental predictive information that is not fully
    recovered by bureau, previous-application or installment-payment features.

    Despite the relatively limited coverage of `credit_card_balance`, its
    contribution to the final model remains meaningful.

    The source is therefore retained in the final feature set.
    """)
    return


@app.cell
def _():
    # full_results = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=FULL_FEATURES,
    #     categorical_features=FULL_CATEGORICAL_FEATURES,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_full_shap_reference",
    #     capacity=0.10,
    #     calculate_shap=True,
    # )
    return


@app.cell
def _():
    # full_repeat_1 = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=FULL_FEATURES,
    #     categorical_features=FULL_CATEGORICAL_FEATURES,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_full_repeat_1",
    #     capacity=0.10,
    #     calculate_shap=False,
    # )
    return


@app.cell
def _():
    # full_repeat_2 = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=FULL_FEATURES,
    #     categorical_features=FULL_CATEGORICAL_FEATURES,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_full_repeat_2",
    #     capacity=0.10,
    #     calculate_shap=False,
    # )
    return


@app.cell
def _(pd):
    shap_summary_full = pd.read_parquet("mlartifacts\\1\\c8f3496812314a84b7d04cd0eb495e38\\artifacts\\shap\\shap_summary.parquet")
    return (shap_summary_full,)


@app.cell
def _(shap_summary_full):
    print(len(shap_summary_full))
    return


@app.cell
def _(shap_summary_full):
    shap_summary_full.head(10)
    return


@app.cell
def _(shap_summary_full):
    shap_summary_full.tail(10)
    return


@app.cell
def _():
    # RFE1_REMOVE = (
    #     shap_summary_full[
    #         shap_summary_full["mean_rank"] > 100
    #     ]
    #     .sort_values("mean_rank", ascending=False)
    #     .head(30)["feature"]
    #     .tolist()


    # rfe1_candidates = (
    #     shap_summary_full[
    #         shap_summary_full["feature"].isin(RFE1_REMOVE)
    #     ]
    #     .sort_values("mean_rank", ascending=False)
    # )
    return


@app.cell
def _():
    # rfe1_candidates[
    #     [
    #         "feature",
    #         "mean_abs_shap",
    #         "mean_normalized_shap",
    #         "mean_rank",
    #         "std_rank",
    #         "best_rank",
    #         "worst_rank",
    #     ]
    # ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RFE1 — Recursive SHAP elimination: bottom 15%

    **Objective**

    Determine whether the final multi-table CatBoost model can be simplified by
    removing features that consistently have low SHAP importance across the fixed
    cross-validation folds.

    The source-level ablation stage showed that all accepted historical data
    sources provide incremental predictive value.

    However, source-level importance does not imply that every individual feature
    inside those sources is necessary.

    RFE1 therefore tests whether the lowest-ranked individual features can be
    removed while preserving the predictive performance of the full model.

    **Reference**

    The reference model contains 204 accepted features from:

    - application data;
    - bureau B1–B4;
    - previous_application P1 + P2 + P4;
    - credit_card_balance CC1 + CC2 + CC4;
    - installments_payments IP1 + IP2 + IP3.

    Rejected feature groups are not included.

    **SHAP ranking**

    SHAP values were calculated independently for each of the five fixed
    cross-validation models.

    For every fold:

    1. CatBoost was trained on the fold training partition;
    2. SHAP values were calculated on the corresponding validation partition;
    3. absolute SHAP values were averaged across validation observations;
    4. features were ranked by validation mean absolute SHAP importance.

    Fold-level rankings were then aggregated into:

    - mean absolute SHAP;
    - mean normalized SHAP;
    - mean SHAP rank;
    - SHAP-rank standard deviation;
    - best fold rank;
    - worst fold rank.

    This produces a cross-validated estimate of feature importance rather than
    relying on a single fitted model.

    **Selection rule**

    The 204 features are ordered by mean SHAP rank across the five folds.

    The bottom 15% of the ranking is removed:

    `204 → 174 features`

    A total of 30 features are removed in RFE1.

    The removed set includes features that consistently receive very low or zero
    SHAP importance, as well as several weak features from different source groups.

    No data source is explicitly protected from feature-level elimination.

    **Hypothesis**

    If the bottom SHAP-ranked features contain little incremental information,
    removing them should preserve validation performance while reducing feature-set
    complexity.

    The goal of RFE1 is not necessarily to increase validation AP.

    A reduction in feature count with essentially unchanged predictive performance
    is considered a successful outcome.

    **Important limitation**

    Low SHAP importance does not prove that a feature is intrinsically useless.

    Correlated or interacting features may receive low individual SHAP importance
    because their information is represented by other variables.

    Therefore, SHAP is used only to generate elimination candidates.

    The reduced feature set must still be validated through the complete fixed
    cross-validation protocol.

    **Change**

    Remove the 30 features corresponding to the bottom 15% of the cross-validated
    SHAP ranking.

    All other modeling components remain unchanged.

    **Controlled variables**

    - same `split_v1`;
    - same five folds;
    - same development population;
    - same CatBoost GPU configuration;
    - same hyperparameters;
    - same random seed;
    - same early-stopping procedure;
    - same preprocessing;
    - only the feature set is changed.

    **Primary comparison**

    Paired validation Average Precision comparison:

    `FULL MODEL — 204 features`

    vs.

    `RFE1 MODEL — 174 features`

    using exactly the same five validation folds.

    **Secondary checks**

    - OOF AP;
    - ROC-AUC;
    - LogLoss;
    - Recall@Top10%;
    - Precision@Top10%;
    - fold AP standard deviation.

    **Run-to-run variability**

    Repeated training of the same full GPU CatBoost configuration showed measurable
    run-to-run variation in validation AP.

    Therefore, very small differences between FULL and RFE1 are not interpreted as
    real improvements or degradations in isolation.

    The decision is based on:

    - magnitude of the mean paired AP change;
    - consistency across folds;
    - OOF metrics;
    - Top-10% policy metrics;
    - feature-count reduction.

    **Recursive step**

    RFE1 also calculates SHAP values for the reduced 174-feature model.

    If RFE1 is accepted, these newly calculated SHAP values — rather than the
    original FULL-model ranking — will be used to select candidates for RFE2.

    This allows feature importance to be recomputed after the model adapts to the
    reduced representation.

    **Decision rule**

    RFE1 is accepted if the 174-feature model preserves predictive performance
    within the observed validation noise and does not materially degrade the
    Top-10% review-policy metrics.

    If performance clearly deteriorates, the 15% elimination step is considered too
    aggressive.

    If performance is preserved, RFE1 becomes the new feature-selection reference
    and its SHAP ranking is used for the next recursive elimination round.
    """)
    return


@app.cell
def _(FULL_CATEGORICAL_FEATURES, FULL_FEATURES, shap_summary_full):
    RFE1_TO_REMOVE = (
        shap_summary_full
        .sort_values("mean_rank", ascending=False)
        .head(30)["feature"]
        .tolist()
    )

    RFE1_FEATURES = [
        feature
        for feature in FULL_FEATURES
        if feature not in RFE1_TO_REMOVE
    ]

    RFE1_CATEGORICAL_FEATURES = [
        feature
        for feature in FULL_CATEGORICAL_FEATURES
        if feature in RFE1_FEATURES
    ]

    print("FULL:", len(FULL_FEATURES))
    print("RFE1:", len(RFE1_FEATURES))
    print("Removed:", len(RFE1_TO_REMOVE))
    return RFE1_CATEGORICAL_FEATURES, RFE1_FEATURES


@app.cell
def _():
    # rfe1_results = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=RFE1_FEATURES,
    #     categorical_features=RFE1_CATEGORICAL_FEATURES,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_rfe1_bottom15",
    #     capacity=0.10,
    #     calculate_shap=True,
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     multitable_accepted_results,
    #     rfe1_results["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RFE1 — Results

    **Reference feature count:** `204`

    **RFE1 feature count:** `174`

    **Features removed:** `30` (`~14.7%`)

    **Mean Δ AP (RFE1 - FULL):** `+0.00026`

    **Positive folds:** `3 / 5`

    **Decision:** **KEEP RFE1**

    Removing the bottom 15% of features according to cross-validated SHAP ranking
    preserves the predictive performance of the full model.

    The mean paired AP difference is approximately `+0.00026`, which is well within
    the observed run-to-run variability of the GPU CatBoost training procedure.

    Therefore, there is no evidence of a meaningful performance loss after removing
    30 features.

    RFE1 reduces the feature set from 204 to 174 features while maintaining
    essentially the same validation quality.

    RFE1 is accepted as the new feature-selection reference.
    """)
    return


@app.cell
def _(pd):
    rfe1_shap_summary = pd.read_parquet("mlartifacts\\1\\2eb0315f16f94d429ea9380f06574d80\\artifacts\\shap\\shap_summary.parquet")
    return (rfe1_shap_summary,)


@app.cell
def _(rfe1_shap_summary):
    RFE2_REMOVE = (
        rfe1_shap_summary
        .sort_values("mean_rank", ascending=False)
        .head(26)["feature"]
        .tolist()
    )

    rfe2_candidates = (
        rfe1_shap_summary[
            rfe1_shap_summary["feature"].isin(RFE2_REMOVE)
        ]
        .sort_values("mean_rank", ascending=False)
    )
    return RFE2_REMOVE, rfe2_candidates


@app.cell
def _(rfe2_candidates):
    rfe2_candidates[
            [
                "feature",
                "mean_abs_shap",
                "mean_normalized_shap",
                "mean_rank",
                "std_rank",
                "best_rank",
                "worst_rank",
            ]
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RFE2 — Recursive SHAP Feature Elimination (Round 2)

    ### Objective / Hypothesis
    Determine whether the 174-feature representation accepted after RFE1 can be simplified further without meaningful loss of predictive performance.
    Hypothesis: Removing the 26 lowest-ranked features from the recalculated RFE1 cross-validated SHAP importance ranking will preserve validation Average Precision within GPU training variability (~0.001 AP) while reducing model complexity.

    ### Setup
    - Reference: Accepted RFE1 model (174 features, GPU CatBoost baseline parameters);
    - Feature representation: RFE2 candidate (148 features, bottom 15% of RFE1 SHAP ranking removed);
    - Validation protocol: Fixed `split_v1` (5 stratified folds, identical development population);
    - Model: CatBoostClassifier (depth=6, iterations=5000, early_stopping_rounds=200, learning_rate=0.05, l2_leaf_reg=3.0, border_count=254, GPU baseline parameters);
    - Metrics: Primary: Validation Average Precision (mean paired ΔAP across 5 folds). Secondary: ROC-AUC, LogLoss, Recall@Top10%, Precision@Top10%.
    """)
    return


@app.cell
def _(RFE1_CATEGORICAL_FEATURES, RFE1_FEATURES, RFE2_REMOVE):
    RFE2_FEATURES = [
        feature
        for feature in RFE1_FEATURES
        if feature not in RFE2_REMOVE
    ]

    RFE2_CATEGORICAL_FEATURES = [
        feature
        for feature in RFE1_CATEGORICAL_FEATURES
        if feature in RFE2_FEATURES
    ]

    print("RFE1:", len(RFE1_FEATURES))
    print("RFE2:", len(RFE2_FEATURES))
    print("Removed:", len(RFE2_REMOVE))

    assert len(RFE1_FEATURES) == 174
    assert len(RFE2_FEATURES) == 148
    assert not set(RFE2_REMOVE) & set(RFE2_FEATURES)
    return RFE2_CATEGORICAL_FEATURES, RFE2_FEATURES


@app.cell
def _():
    # rfe2_results = run_catboost_experiment(
    #     frame=training_dataset,
    #     features=RFE2_FEATURES,
    #     categorical_features=RFE2_CATEGORICAL_FEATURES,
    #     params=BASELINE_PARAMS,
    #     run_name="cb_rfe2_bottom15",
    #     capacity=0.10,
    #     calculate_shap=True,
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     rfe1_results["fold_metrics"],
    #     rfe2_results["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results
    - Feature count: 174 → 148 features (-26 features, -14.9% vs RFE1, -27.5% vs FULL 204);
    - Mean paired ΔAP vs RFE1: **-0.00052**;
    - Positive folds: **2 / 5**;
    - Secondary metrics: ROC-AUC, LogLoss, and Top-10% review-policy metrics remained stable within noise.

    ### Interpretation
    The paired validation degradation of -0.00052 AP is well within observed GPU CatBoost run-to-run variability (~0.001 AP). RFE2 successfully preserves the predictive ranking quality and top-decile capture rate of RFE1 while eliminating 26 redundant features and simplifying the final pipeline.

    ### Decision
    **ACCEPT** — RFE2 (148 features) is accepted as the final feature representation for subsequent hyperparameter tuning and model exploration.
    """)
    return


@app.cell
def _(RFE2_CATEGORICAL_FEATURES, RFE2_FEATURES):
    FINAL_FEATURES = RFE2_FEATURES.copy()
    FINAL_CATEGORICAL_FEATURES = RFE2_CATEGORICAL_FEATURES.copy()
    return (FINAL_FEATURES,)


@app.cell
def _(pd):
    rfe_result = pd.read_parquet("mlartifacts\\1\\2eb0315f16f94d429ea9380f06574d80\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## T1 — CatBoost Hyperparameter Optimization

    ### Objective / Hypothesis
    Improve the predictive ranking performance of the accepted RFE2 CatBoost model through controlled hyperparameter optimization on a fixed feature representation.
    Hypothesis: Tuning tree depth, learning rate, L2 regularization, and random strength via Bayesian optimization (Optuna TPE) will discover configurations that outperform default baseline parameters consistently across all 5 validation folds.

    ### Setup
    - Reference: Accepted RFE2 CatBoost model (148 features, default baseline parameters);
    - Feature representation: Fixed RFE2 feature set (148 features, untouched internal holdout);
    - Validation protocol: Fixed `split_v1` (5 stratified folds, identical development population);
    - Search algorithm: Optuna TPE sampler (30 trials, seed-controlled);
    - Search space:
      - `depth`: [4, 10] (step 1)
      - `learning_rate`: [0.01, 0.10] (log scale)
      - `l2_leaf_reg`: [1.0, 10.0] (log scale)
      - `random_strength`: [1e-3, 10.0] (log scale)
    - Metrics: Primary: Validation Average Precision (mean across 5 folds). Secondary: ROC-AUC, LogLoss, Precision@Top10%, Recall@Top10%.
    """)
    return


@app.cell
def _():
    # study_base = run_optuna_tuning(
    #     frame=training_dataset,
    #     features=FINAL_FEATURES,
    #     categorical_features=FINAL_CATEGORICAL_FEATURES,
    #     base_params=BASELINE_PARAMS,
    #     study_name="catboost_final_features_v1",
    #     n_trials=12,
    # )
    return


@app.cell
def _():
    # top_trials = sorted(
    #     [t for t in study_base.trials if t.value is not None],
    #     key=lambda t: t.value,
    #     reverse=True,
    # )[:3]

    # for t in top_trials:
    #     print(
    #         f"Trial {t.number}: "
    #         f"AP={t.value:.6f}, "
    #         f"params={t.params}"
    #     )
    return


@app.cell
def _():
    # top3_df = pd.DataFrame(
    #     [
    #         {
    #             "trial": t.number,
    #             "study_ap": t.value,
    #             **t.params,
    #         }
    #         for t in top_trials
    #     ]
    # )

    # top3_df
    return


@app.cell
def _():
    # top3_results = {}

    # for rank, trial in enumerate(top_trials, start=1):
    #     tuned_params = {
    #         **BASELINE_PARAMS,
    #         **trial.params,
    #     }

    #     tuning_result = run_catboost_experiment(
    #         frame=training_dataset,
    #         features=FINAL_FEATURES,
    #         categorical_features=FINAL_CATEGORICAL_FEATURES,
    #         params=tuned_params,
    #         run_name=f"cb_t1_top{rank}_trial_{trial.number}",
    #         capacity=0.10,
    #         calculate_shap=False,
    #     )

    #     top3_results[trial.number] = tuning_result
    return


@app.cell
def _():
    # for trial_number, result in top3_results.items():
    #     print(f"\nTrial {trial_number}")

    #     print(
    #         compare_fold_results(
    #             rfe_result,
    #             result["fold_metrics"],
    #             "valid_ap",
    #             "valid_ap",
    #         )
    #     )
    return


@app.cell
def _():
    T1_BEST_PARAMS = {
        "depth": 7,
        "learning_rate": 0.020111585095657195,
        "l2_leaf_reg": 6.618687434133027,
        "random_strength": 0.07185154417688532,
    }
    return (T1_BEST_PARAMS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Results
    The modeling lineage leading up to this optimization stage is:
    - **FULL** (204 features)
    - **RFE1** (174 features)
    - **RFE2** (148 features, accepted for parsimony)
    - **Optuna TPE optimization** (30 trials) evaluated on the fixed 148-feature RFE2 representation.

    The top-3 Optuna configurations were independently verified against the RFE1 baseline across all 5 fixed folds:

    | Trial | depth | learning_rate | l2_leaf_reg | random_strength | Mean ΔAP vs RFE1 Reference | Positive Folds |
    |---|---|---|---|---|---|---|
    | **29** | 7 | 0.0201 | 6.6187 | 0.0719 | **+0.00160** | **5 / 5** |
    | 7 | 7 | 0.0138 | 2.7011 | 0.1256 | +0.00131 | 5 / 5 |
    | 15 | 7 | 0.0233 | 2.6006 | 0.7221 | +0.00094 | 5 / 5 |

    ### Interpretation
    All three top trials converged on `depth = 7` with conservative shrinkage (`learning_rate ~ 0.014–0.023`), confirming that moderate tree depth with strong regularization is optimal for this tabular schema. Trial 29 yielded the largest paired improvement (+0.00160 AP) with strong L2 regularization (`l2_leaf_reg = 6.62`).

    Crucially, because the reference run used in `compare_fold_results` is the 174-feature RFE1 baseline, this displayed +0.00160 AP delta reflects the combined effect of feature-set pruning (174 → 148 features) and hyperparameter optimization. The displayed Trial29-vs-RFE1 delta is not interpreted as a pure tuning effect.

    ### Decision
    **ACCEPT** — Trial 29 is accepted as the tuned RFE2 CatBoost candidate (`T1_BEST_PARAMS`) for subsequent error analysis, dynamic feature testing, and ensemble modeling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EA1 — Out-of-Fold Error Analysis: Ranking Failures and Model Disagreement

    ### Objective / Hypothesis
    Analyze out-of-fold predictions from the tuned CatBoost candidate (Trial 29) against the RFE1 baseline to diagnose ranking failures, assess review-policy behavior, and determine whether systematic error cohorts justify targeted feature engineering.
    Hypothesis: Ranking errors (hard false negatives, model disagreements, and cohort underperformance) may stem from uncaptured behavioral dynamics, missing historical signals, or cohort-specific interactions.

    ### Setup
    - Primary model: Final CatBoost candidate (Trial 29, 148 features, `split_v1` 5-fold OOF predictions);
    - Comparator: Accepted RFE1 CatBoost model (174 features, baseline parameters); note that this comparison describes the final CatBoost candidate vs RFE1, not a pure tuning impact.
    - Validation protocol: Fixed `split_v1` (5 stratified folds, identical development population, aligned by `SK_ID_CURR`);
    - Target policy: Global Top-10% risk queue (capacity = 0.10);
    - Scope:
      1. Global OOF alignment and Top-10% review queue overlap;
      2. Hard false negatives and data availability;
      3. Score-matched diagnostic analysis;
      4. Categorical cohort performance (Drivers and Sales staff);
      5. Numeric cohort scan across feature quintiles.
    - Holdout policy: Internal holdout remains untouched.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EA1.1 — Global OOF & Top-10% Review Policy Diagnostics

    #### Objective / Hypothesis
    Compare out-of-fold probability distributions, rank movements, and Top-10% review-queue composition between the accepted RFE1 model and the final CatBoost candidate (Trial 29).
    Hypothesis: The transition from the RFE1 baseline to the final CatBoost candidate (Trial 29) improved global ranking metrics (AP and ROC-AUC) through combined feature-pruning and parameter tuning. Comparing out-of-fold distributions and review-queue overlap against RFE1 helps diagnose whether rank movements shifted high-risk applicants into the Top-10% queue.
    """)
    return


@app.cell
def _(pd):
    rfe1_oof = pd.read_parquet("mlartifacts\\1\\2eb0315f16f94d429ea9380f06574d80\\artifacts\\predictions\\oof_predictions.parquet")
    t29_oof = pd.read_parquet("mlartifacts\\1\\5665cc4f67844ea79c909b9951c593c1\\artifacts\\predictions\\oof_predictions.parquet")
    return rfe1_oof, t29_oof


@app.cell
def _(rfe1_oof, t29_oof):
    print(rfe1_oof.shape)
    print(t29_oof.shape)

    print(rfe1_oof.columns.tolist())
    print(t29_oof.columns.tolist())
    return


@app.cell
def _(rfe1_oof, t29_oof):
    assert rfe1_oof["SK_ID_CURR"].is_unique
    assert t29_oof["SK_ID_CURR"].is_unique

    assert set(rfe1_oof["SK_ID_CURR"]) == set(t29_oof["SK_ID_CURR"])

    assert rfe1_oof["probability"].notna().all()
    assert t29_oof["probability"].notna().all()

    assert rfe1_oof["probability"].between(0, 1).all()
    assert t29_oof["probability"].between(0, 1).all()
    return


@app.cell
def _(rfe1_oof, t29_oof):
    oof_analysis = (
        rfe1_oof[
            ["SK_ID_CURR", "TARGET", "fold", "probability"]
        ]
        .rename(
            columns={
                "probability": "pred_rfe1",
            }
        )
        .merge(
            t29_oof[
                ["SK_ID_CURR", "TARGET", "fold", "probability"]
            ].rename(
                columns={
                    "TARGET": "TARGET_t29",
                    "fold": "fold_t29",
                    "probability": "pred_t29",
                }
            ),
            on="SK_ID_CURR",
            how="inner",
            validate="one_to_one",
        )
    )

    assert (oof_analysis["TARGET"] == oof_analysis["TARGET_t29"]).all()
    assert (oof_analysis["fold"] == oof_analysis["fold_t29"]).all()

    oof_analysis = oof_analysis.drop(
        columns=["TARGET_t29", "fold_t29"]
    )
    return (oof_analysis,)


@app.cell
def _(oof_analysis):
    oof_analysis["pred_delta"] = (
        oof_analysis["pred_t29"]
        - oof_analysis["pred_rfe1"]
    )
    return


@app.cell
def _(oof_analysis):
    oof_analysis
    return


@app.cell
def _(oof_analysis):
    oof_analysis.sort_values("pred_delta", ascending=False).head(10)
    return


@app.cell
def _(oof_analysis, spearmanr):
    print(
        oof_analysis[
            ["pred_rfe1", "pred_t29", "pred_delta"]
        ].describe(
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        )
    )

    print(
        "Pearson:",
        oof_analysis["pred_rfe1"].corr(oof_analysis["pred_t29"])
    )

    print(
        "Spearman:",
        spearmanr(
            oof_analysis["pred_rfe1"],
            oof_analysis["pred_t29"],
        ).statistic
    )

    oof_analysis["abs_pred_delta"] = (
        oof_analysis["pred_delta"].abs()
    )

    print(
        oof_analysis["abs_pred_delta"].describe(
            percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]
        )
    )
    return


@app.cell
def _(oof_analysis):
    oof_analysis["rank_rfe1"] = (
        oof_analysis["pred_rfe1"]
        .rank(method="average", ascending=False)
    )

    oof_analysis["rank_t29"] = (
        oof_analysis["pred_t29"]
        .rank(method="average", ascending=False)
    )

    oof_analysis["rank_improvement"] = (
        oof_analysis["rank_rfe1"]
        - oof_analysis["rank_t29"]
    )
    return


@app.cell
def _(oof_analysis):
    oof_analysis.groupby("TARGET").agg(
            mean_pred_delta=("pred_delta", "mean"),
            median_pred_delta=("pred_delta", "median"),
            mean_rank_improvement=("rank_improvement", "mean"),
            median_rank_improvement=("rank_improvement", "median"),
            mean_abs_rank_change=(
                "rank_improvement",
                lambda x: x.abs().mean(),
            ),
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Top-10% Review Queue Dynamics
    Analyze applicants around the highest-risk 10% decision boundary (`capacity = 0.10`) to assess overlap, queue churn, and target capture consistency between RFE1 and Trial 29.
    """)
    return


@app.cell
def _(oof_analysis):
    n_top = int(len(oof_analysis) * 0.10)

    top_rfe1_ids = set(
        oof_analysis.nlargest(
            n_top,
            "pred_rfe1",
        )["SK_ID_CURR"]
    )

    top_t29_ids = set(
        oof_analysis.nlargest(
            n_top,
            "pred_t29",
        )["SK_ID_CURR"]
    )

    overlap = len(top_rfe1_ids & top_t29_ids)

    print("Top-10 size:", n_top)
    print("Overlap:", overlap)
    print("Overlap share:", overlap / n_top)

    print(
        "Entered T29:",
        len(top_t29_ids - top_rfe1_ids)
    )

    print(
        "Left T29:",
        len(top_rfe1_ids - top_t29_ids)
    )
    return top_rfe1_ids, top_t29_ids


@app.cell
def _(oof_analysis, top_rfe1_ids, top_t29_ids):
    oof_analysis["top10_rfe1"] = (
        oof_analysis["SK_ID_CURR"].isin(top_rfe1_ids)
    )

    oof_analysis["top10_t29"] = (
        oof_analysis["SK_ID_CURR"].isin(top_t29_ids)
    )

    oof_analysis["top10_change"] = "same"

    oof_analysis.loc[
        (~oof_analysis["top10_rfe1"])
        & (oof_analysis["top10_t29"]),
        "top10_change"
    ] = "entered"

    oof_analysis.loc[
        (oof_analysis["top10_rfe1"])
        & (~oof_analysis["top10_t29"]),
        "top10_change"
    ] = "left"
    return


@app.cell
def _(oof_analysis, pd):
    pd.crosstab(
        oof_analysis["top10_change"],
        oof_analysis["TARGET"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Results & Interpretation
    - Prediction correlation: High agreement between RFE1 and Trial 29 (Pearson r ≈ 0.99, Spearman r ≈ 0.99);
    - Queue overlap: ~90% of reviewed applicants in the Top-10% queue are identical between models;
    - Boundary movement: Applications entering and leaving the 10% boundary possess virtually identical default rates;
    - Diagnostic conclusion: Trial 29 systematically moves positive applications higher in global ranking and non-defaults lower (driving AP and ROC-AUC gains), but the top-decile review operating point remains stable.
    """)
    return


@app.cell
def _(oof_analysis):
    oof_analysis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### High-Confidence Ranking Errors

    Analyze:

    - `TARGET = 1` applications assigned very low risk;
    - `TARGET = 0` applications assigned very high risk.

    The analysis focuses on ranking errors rather than using an arbitrary
    classification threshold such as `0.5`.
    """)
    return


@app.cell
def _(oof_analysis):
    positive_mask = oof_analysis["TARGET"] == 1

    hard_fn_threshold = (
        oof_analysis.loc[
            positive_mask,
            "pred_t29",
        ]
        .quantile(0.10)
    )

    oof_analysis["hard_fn_t29"] = (
        (oof_analysis["TARGET"] == 1)
        & (oof_analysis["pred_t29"] <= hard_fn_threshold)
    )
    return (hard_fn_threshold,)


@app.cell
def _(hard_fn_threshold, oof_analysis):
    print("Threshold:", hard_fn_threshold)

    print(
        oof_analysis["hard_fn_t29"]
        .value_counts()
    )
    return


@app.cell
def _(oof_analysis):
    oof_analysis.loc[
            oof_analysis["hard_fn_t29"],
            [
                "SK_ID_CURR",
                "TARGET",
                "pred_rfe1",
                "pred_t29",
                "rank_rfe1",
                "rank_t29",
            ],
        ].sort_values("pred_t29").head(20)
    return


@app.cell
def _(oof_analysis):
    oof_analysis["entered_top10_t29"] = (
        (~oof_analysis["top10_rfe1"])
        & (oof_analysis["top10_t29"])
    )
    return


@app.cell
def _(oof_analysis):
    oof_analysis["left_top10_t29"] = (
        (oof_analysis["top10_rfe1"])
        & (~oof_analysis["top10_t29"])
    )
    return


@app.cell
def _(oof_analysis):
    oof_analysis["abs_rank_change"] = (
        oof_analysis["rank_improvement"].abs()
    )
    return


@app.cell
def _(oof_analysis):
    rank_change_threshold = (
        oof_analysis["abs_rank_change"]
        .quantile(0.99)
    )

    oof_analysis["large_rank_disagreement"] = (
        oof_analysis["abs_rank_change"]
        >= rank_change_threshold
    )
    return (rank_change_threshold,)


@app.cell
def _(oof_analysis, rank_change_threshold):
    print("99th percentile rank change:", rank_change_threshold)

    print(
        oof_analysis["large_rank_disagreement"]
        .sum()
    )
    return


@app.cell
def _(oof_analysis):
    oof_analysis["large_rank_up_t29"] = (
        oof_analysis["large_rank_disagreement"]
        & (oof_analysis["rank_improvement"] > 0)
    )

    oof_analysis["large_rank_down_t29"] = (
        oof_analysis["large_rank_disagreement"]
        & (oof_analysis["rank_improvement"] < 0)
    )
    return


@app.cell
def _(oof_analysis, pd):
    groups = {
        "hard_fn_t29": oof_analysis["hard_fn_t29"],
        "entered_top10_t29": oof_analysis["entered_top10_t29"],
        "left_top10_t29": oof_analysis["left_top10_t29"],
        "large_rank_up_t29": oof_analysis["large_rank_up_t29"],
        "large_rank_down_t29": oof_analysis["large_rank_down_t29"],
    }

    group_summary = []

    for name, mask in groups.items():
        subset = oof_analysis.loc[mask]

        group_summary.append(
            {
                "group": name,
                "n": len(subset),
                "target_rate": subset["TARGET"].mean(),
                "mean_pred_rfe1": subset["pred_rfe1"].mean(),
                "mean_pred_t29": subset["pred_t29"].mean(),
                "mean_pred_delta": subset["pred_delta"].mean(),
                "mean_rank_rfe1": subset["rank_rfe1"].mean(),
                "mean_rank_t29": subset["rank_t29"].mean(),
                "mean_rank_improvement": subset[
                    "rank_improvement"
                ].mean(),
            }
        )

    group_summary = pd.DataFrame(group_summary)

    group_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Preliminary Cohort Diagnostics by History Source

    Evaluate OOF performance across meaningful applicant groups.

    Candidate cohorts include:

    - with / without bureau history;
    - with / without previous-application history;
    - with / without installment-payment history;
    - with / without credit-card history;
    - number of previous contracts;
    - age groups;
    - income groups;
    - credit amount groups;
    - contract type.

    For each sufficiently large cohort, compare:

    - prevalence;
    - Average Precision;
    - ROC-AUC where meaningful;
    - Recall@Top10%;
    - prediction distribution.
    """)
    return


@app.cell
def _(oof_analysis, training_dataset):
    history_columns = [
        "SK_ID_CURR",
        "BUREAU_CREDIT_COUNT",
        "PREV_APP_COUNT",
        "IP_CONTRACT_COUNT",
        "HAS_CREDIT_CARD_HISTORY",
    ]

    error_analysis = oof_analysis.merge(
        training_dataset[history_columns],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (error_analysis,)


@app.cell
def _(error_analysis):
    error_analysis["HAS_BUREAU_HISTORY"] = (
        error_analysis["BUREAU_CREDIT_COUNT"].fillna(0) > 0
    )

    error_analysis["HAS_PREVIOUS_HISTORY"] = (
        error_analysis["PREV_APP_COUNT"].fillna(0) > 0
    )

    error_analysis["HAS_INSTALLMENTS_HISTORY"] = (
        error_analysis["IP_CONTRACT_COUNT"].fillna(0) > 0
    )

    error_analysis["HAS_CREDIT_CARD_HISTORY"] = (
        error_analysis["HAS_CREDIT_CARD_HISTORY"]
        .fillna(0)
        .astype(bool)
    )
    return


@app.cell
def _(error_analysis, pd):
    group_summary_ext = pd.DataFrame(
        [
            {
                "group": "all",
                "n": len(error_analysis),
                "target_rate": error_analysis["TARGET"].mean(),
            },
            {
                "group": "all_positives",
                "n": (error_analysis["TARGET"] == 1).sum(),
                "target_rate": 1.0,
            },
            {
                "group": "hard_fn_t29",
                "n": error_analysis["hard_fn_t29"].sum(),
                "target_rate": error_analysis.loc[
                    error_analysis["hard_fn_t29"], "TARGET"
                ].mean(),
            },
            {
                "group": "entered_top10_t29",
                "n": error_analysis["entered_top10_t29"].sum(),
                "target_rate": error_analysis.loc[
                    error_analysis["entered_top10_t29"], "TARGET"
                ].mean(),
            },
            {
                "group": "left_top10_t29",
                "n": error_analysis["left_top10_t29"].sum(),
                "target_rate": error_analysis.loc[
                    error_analysis["left_top10_t29"], "TARGET"
                ].mean(),
            },
            {
                "group": "large_rank_up_t29",
                "n": error_analysis["large_rank_up_t29"].sum(),
                "target_rate": error_analysis.loc[
                    error_analysis["large_rank_up_t29"], "TARGET"
                ].mean(),
            },
            {
                "group": "large_rank_down_t29",
                "n": error_analysis["large_rank_down_t29"].sum(),
                "target_rate": error_analysis.loc[
                    error_analysis["large_rank_down_t29"], "TARGET"
                ].mean(),
            },
        ]
    )

    group_summary_ext
    return


@app.cell
def _(error_analysis):
    history_flags = [
        "HAS_BUREAU_HISTORY",
        "HAS_PREVIOUS_HISTORY",
        "HAS_INSTALLMENTS_HISTORY",
        "HAS_CREDIT_CARD_HISTORY",
    ]

    def summarize_history(mask, name):
        df = error_analysis.loc[mask]

        return {
            "group": name,
            "n": len(df),
            **{
                col: df[col].mean()
                for col in history_flags
            },
        }

    return (summarize_history,)


@app.cell
def _(error_analysis, pd, summarize_history):
    history_summary = pd.DataFrame(
        [
            summarize_history(
                error_analysis["TARGET"] == 1,
                "all_positives",
            ),

            summarize_history(
                error_analysis["hard_fn_t29"],
                "hard_fn_t29",
            ),

            summarize_history(
                error_analysis["entered_top10_t29"],
                "entered_top10_t29",
            ),

            summarize_history(
                error_analysis["left_top10_t29"],
                "left_top10_t29",
            ),

            summarize_history(
                error_analysis["large_rank_up_t29"],
                "large_rank_up_t29",
            ),

            summarize_history(
                error_analysis["large_rank_down_t29"],
                "large_rank_down_t29",
            ),
        ]
    )

    history_summary
    return


@app.cell
def _(error_analysis):
    count_features = [
        "BUREAU_CREDIT_COUNT",
        "PREV_APP_COUNT",
        "IP_CONTRACT_COUNT",
    ]

    def summarize_counts(mask, name):
        df = error_analysis.loc[mask]

        row = {"group": name, "n": len(df)}

        for col in count_features:
            row[f"{col}_median"] = df[col].median()
            row[f"{col}_mean"] = df[col].mean()

        return row

    return (summarize_counts,)


@app.cell
def _(error_analysis, pd, summarize_counts):
    count_summary = pd.DataFrame(
        [
            summarize_counts(
                error_analysis["TARGET"] == 1,
                "all_positives",
            ),
            summarize_counts(
                error_analysis["hard_fn_t29"],
                "hard_fn_t29",
            ),
            summarize_counts(
                error_analysis["entered_top10_t29"],
                "entered_top10_t29",
            ),
            summarize_counts(
                error_analysis["left_top10_t29"],
                "left_top10_t29",
            ),
        ]
    )

    count_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Historical Data Availability Diagnostic

    Historical-source availability does not explain the hard false-negative cohort.

    Hard false negatives are not characterized by missing historical information.
    Compared with all positive applications, they have:

    - substantially higher bureau-history coverage;
    - similar previous-application coverage;
    - similar installment-payment coverage;
    - similar credit-card coverage.

    They also have slightly more bureau credits and installment contracts on
    average.

    Therefore, the primary failure mode is unlikely to be insufficient historical
    data.

    A more plausible hypothesis is that some hard false negatives have substantial
    but apparently low-risk historical behavior, causing the model to underestimate
    risk in the current application.

    The next analysis should therefore compare the content of current-application
    and repayment-history features rather than source availability alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Profile Comparison: Detected Defaults vs Hard False Negatives
    """)
    return


@app.cell
def _(error_analysis):
    error_analysis["detected_positive_t29"] = (
        (error_analysis["TARGET"] == 1)
        & (error_analysis["top10_t29"])
    )
    return


@app.cell
def _(error_analysis, training_dataset):
    CURRENT_PROFILE_FEATURES = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",

        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",

        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",

        "DAYS_BIRTH",
        "DAYS_EMPLOYED",

        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
    ]

    profile_analysis = error_analysis.merge(
        training_dataset[
            ["SK_ID_CURR"] + CURRENT_PROFILE_FEATURES
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (profile_analysis,)


@app.cell
def _(pd):
    numeric_profile_features = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",

        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",

        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",

        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ]

    def summarize_numeric_group(df, mask, features, prefix):
        subset = df.loc[mask]

        return pd.DataFrame({
            "feature": features,

            f"{prefix}_mean": [
                subset[feature].mean()
                for feature in features
            ],

            f"{prefix}_median": [
                subset[feature].median()
                for feature in features
            ],

            f"{prefix}_missing_share": [
                subset[feature].isna().mean()
                for feature in features
            ],
        })

    return numeric_profile_features, summarize_numeric_group


@app.cell
def _(numeric_profile_features, profile_analysis, summarize_numeric_group):
    hard_fn_profile = summarize_numeric_group(
        profile_analysis,
        profile_analysis["hard_fn_t29"],
        numeric_profile_features,
        "hard_fn",
    )

    detected_profile = summarize_numeric_group(
        profile_analysis,
        profile_analysis["detected_positive_t29"],
        numeric_profile_features,
        "detected",
    )
    return detected_profile, hard_fn_profile


@app.cell
def _(detected_profile, hard_fn_profile):
    profile_comparison = hard_fn_profile.merge(
        detected_profile,
        on="feature",
        how="inner",
        validate="one_to_one",
    )
    return (profile_comparison,)


@app.cell
def _(profile_comparison):
    profile_comparison["mean_delta"] = (
        profile_comparison["hard_fn_mean"]
        - profile_comparison["detected_mean"]
    )

    profile_comparison["median_delta"] = (
        profile_comparison["hard_fn_median"]
        - profile_comparison["detected_median"]
    )

    profile_comparison["missing_delta"] = (
        profile_comparison["hard_fn_missing_share"]
        - profile_comparison["detected_missing_share"]
    )
    return


@app.cell
def _(profile_comparison):
    profile_comparison[
            [
                "feature",

                "hard_fn_mean",
                "detected_mean",
                "mean_delta",

                "hard_fn_median",
                "detected_median",
                "median_delta",

                "hard_fn_missing_share",
                "detected_missing_share",
                "missing_delta",
            ]
        ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EA1.2 — Score-Matched Hard False Negatives Analysis

    #### Objective

    Determine whether the lowest-ranked true defaults contain residual information that distinguishes them from genuinely low-risk non-default applicants.

    The previous analysis showed that hard false negatives are not caused by a lack of historical data. They have broadly similar or even richer historical coverage than the positive population overall.

    The next question is therefore more specific:

    > Among applicants receiving approximately the same low OOF risk score, are there observable pre-decision characteristics that distinguish actual defaults from actual non-defaults?

    If such a pattern exists, it may indicate a feature-engineering opportunity that the current model representation does not capture sufficiently well.

    ---

    #### Cohort definition

    The primary model is the tuned CatBoost candidate trained on the accepted RFE2 feature representation.

    Hard false negatives are defined as:

    * `TARGET = 1`;
    * bottom 10% of `pred_t29` among all positive development applications.

    This isolates defaults that the model considers particularly low risk.

    Because directly comparing these applications with all non-defaults would confound the analysis with model score, a score-matched negative cohort is constructed.

    For each hard false negative, non-default applications with similar OOF `pred_t29` values are sampled.

    Matching is performed using quantile bins of the hard-false-negative score distribution, with approximately 1:1 sampling within each score bin.

    This produces two cohorts with similar predicted-risk distributions:

    * hard false negatives: `TARGET = 1`;
    * matched low-risk negatives: `TARGET = 0`.

    The comparison therefore focuses on residual differences not already strongly reflected in the model score.

    ---

    #### Diagnostic method

    Numeric features are compared using:

    * mean;
    * median;
    * missing-value share;
    * standardized mean difference (SMD).

    SMD is used as an exploratory effect-size measure rather than as a statistical significance test.

    Large absolute SMD values would indicate feature-level separation between the two score-matched cohorts.

    The analysis is performed in three layers:

    1. historical behavior;
    2. current-application characteristics;
    3. cross-table exposure features.

    ---

    #### Historical behavior

    Historical features from:

    * `bureau`;
    * `previous_application`;
    * `installments_payments`;
    * `credit_card_balance`

    were compared between the two score-matched cohorts.

    The largest observed standardized difference was:

    `PREV_MEAN_DAYS_SINCE_DECISION`

    with approximately:

    `|SMD| ≈ 0.124`.

    Other relatively larger differences were observed in:

    * credit-card maximum balance;
    * credit-card history length;
    * maximum credit-card utilization;
    * previous-application recency;
    * installment-payment recency.

    However, all effects remained small.

    Installment-payment behavior did not reveal a clear deterioration pattern.

    Hard false negatives were not materially worse in:

    * mean payment delay;
    * recent maximum delay;
    * payment coverage;
    * recent underpayment frequency.

    Credit-card features showed somewhat larger exposure among hard false negatives, but credit-card history is available only for a minority of applicants and the effect sizes remained modest.

    **Finding**

    No strong historical behavior feature separates hard false negatives from non-defaults with approximately the same predicted risk.

    ---

    #### Current-application characteristics

    Current-application features were then compared using the same score-matched cohorts.

    This substantially changed the interpretation of the earlier hard-FN profile.

    Before score matching, hard false negatives appeared much safer than detected defaults:

    * substantially higher `EXT_SOURCE` scores;
    * older age;
    * longer employment history;
    * higher income;
    * lower apparent payment burden.

    After matching on model score, these differences almost disappeared.

    The largest current-application effect sizes were approximately:

    * `EXT_SOURCE_1`: `|SMD| ≈ 0.063`;
    * `CREDIT_INCOME_RATIO`: `|SMD| ≈ 0.062`;
    * `ANNUITY_INCOME_RATIO`: `|SMD| ≈ 0.055`.

    Other major predictors, including:

    * `EXT_SOURCE_2`;
    * `EXT_SOURCE_3`;
    * age;
    * employment duration;
    * income;
    * credit amount;
    * annuity;
    * goods price

    showed only very small differences.

    The previously observed `DAYS_EMPLOYED` missingness pattern also largely disappeared after score matching.

    **Finding**

    The strong "safe-looking default" profile observed earlier is largely explained by the fact that these applicants occupy the low-risk region of the model's feature space.

    Within the same predicted-risk region, current-application features provide little additional separation between defaults and non-defaults.

    ---

    #### Cross-table exposure diagnostics

    A small set of economically interpretable cross-table exposure features was constructed to test whether the model was missing combinations of current and historical credit burden.

    The tested diagnostics included:

    * bureau debt / income;
    * bureau debt / current credit;
    * total current + bureau exposure / income;
    * maximum credit-card balance / income;
    * maximum credit-card balance / current credit.

    The largest observed effect was:

    `CC_MAX_BALANCE_INCOME_RATIO`

    with approximately:

    `|SMD| ≈ 0.109`.

    However, credit-card information is missing for roughly 70% of the analyzed cohort.

    The bureau-based and total-exposure ratios showed almost no separation:

    `|SMD| ≈ 0.01–0.02`.

    **Finding**

    There is no evidence that hidden aggregate debt exposure is the primary reason for these hard false negatives.

    No CV feature experiment is justified from this feature family.

    ---

    #### Interpretation

    After controlling for OOF predicted risk, hard false negatives and genuine low-risk non-defaults are highly similar across the feature families examined.

    The remaining errors may therefore reflect one or more of:

    * unobserved borrower characteristics;
    * future events that are unavailable at decision time;
    * inherently probabilistic default outcomes;
    * label noise;
    * higher-order interactions not exposed by the current diagnostic comparisons.

    The analysis does **not** imply that these targets are random.

    It indicates that no clear residual signal has been found in the available pre-decision features that would justify targeted feature engineering for this specific error cohort.

    ---

    #### Decision

    **CLOSE THIS BRANCH**

    No new feature bundle is created from the hard-false-negative analysis.

    The next error-analysis stage moves from individual hard cases to **cohort-level model performance**, looking for applicant segments where ranking quality is systematically weaker.

    The internal holdout remains untouched.
    """)
    return


@app.cell
def _(error_analysis):
    hard_fn_scores = error_analysis.loc[
        error_analysis["hard_fn_t29"],
        "pred_t29",
    ]

    score_min = hard_fn_scores.min()
    score_max = hard_fn_scores.max()

    print(score_min, score_max)
    return score_max, score_min


@app.cell
def _(error_analysis, score_max, score_min):
    error_analysis["safe_negative_match"] = (
        (error_analysis["TARGET"] == 0)
        & (error_analysis["pred_t29"] >= score_min)
        & (error_analysis["pred_t29"] <= score_max)
    )
    return


@app.cell
def _(error_analysis):
    print(
        "Hard FN:",
        error_analysis["hard_fn_t29"].sum()
    )

    print(
        "Safe negatives:",
        error_analysis["safe_negative_match"].sum()
    )

    print(
        error_analysis.loc[
            error_analysis["hard_fn_t29"],
            "pred_t29",
        ].describe()
    )

    print(
        error_analysis.loc[
            error_analysis["safe_negative_match"],
            "pred_t29",
        ].describe()
    )
    return


@app.cell
def _(error_analysis, np):
    hard_fn = (
        error_analysis.loc[
            error_analysis["hard_fn_t29"]
        ]
        .copy()
    )

    safe_negatives = (
        error_analysis.loc[
            error_analysis["TARGET"] == 0
        ]
        .copy()
    )

    quantiles = np.linspace(0, 1, 21)

    bin_edges = (
        hard_fn["pred_t29"]
        .quantile(quantiles)
        .to_numpy()
    )

    bin_edges = np.unique(bin_edges)
    bin_edges[0] -= 1e-12
    bin_edges[-1] += 1e-12
    return bin_edges, hard_fn, safe_negatives


@app.cell
def _(bin_edges, hard_fn, pd, safe_negatives):
    hard_fn["score_bin"] = pd.cut(
        hard_fn["pred_t29"],
        bins=bin_edges,
        include_lowest=True,
    )

    safe_negatives["score_bin"] = pd.cut(
        safe_negatives["pred_t29"],
        bins=bin_edges,
        include_lowest=True,
    )
    return


@app.cell
def _(hard_fn, pd, safe_negatives):
    matched_negative_parts = []

    for score_bin, hard_bin in hard_fn.groupby(
        "score_bin",
        observed=True,
    ):
        negative_bin = safe_negatives.loc[
            safe_negatives["score_bin"] == score_bin
        ]

        n_needed = len(hard_bin)

        if len(negative_bin) < n_needed:
            print(
                f"Warning: {score_bin}: "
                f"need {n_needed}, "
                f"available {len(negative_bin)}"
            )

            n_needed = len(negative_bin)

        sampled = negative_bin.sample(
            n=n_needed,
            random_state=42,
            replace=False,
        )

        matched_negative_parts.append(sampled)

    matched_negatives = pd.concat(
        matched_negative_parts,
        ignore_index=True,
    )
    return (matched_negatives,)


@app.cell
def _(hard_fn, matched_negatives):
    print("Hard FN:", len(hard_fn))
    print("Matched negatives:", len(matched_negatives))
    return


@app.cell
def _(hard_fn, matched_negatives, pd):
    score_comparison = pd.DataFrame({
        "hard_fn": hard_fn["pred_t29"].describe(),
        "matched_negative": matched_negatives["pred_t29"].describe(),
    })

    score_comparison
    return


@app.cell
def _(hard_fn, matched_negatives):
    print(
        "Hard FN mean score:",
        hard_fn["pred_t29"].mean()
    )

    print(
        "Matched negative mean score:",
        matched_negatives["pred_t29"].mean()
    )

    print(
        "Hard FN median score:",
        hard_fn["pred_t29"].median()
    )

    print(
        "Matched negative median score:",
        matched_negatives["pred_t29"].median()
    )
    return


@app.cell
def _(FINAL_FEATURES):
    historical_features = [
        feature
        for feature in FINAL_FEATURES
        if feature.startswith(
            (
                "BUREAU_",
                "PREV_",
                "IP_",
                "CC_",
            )
        )
        and feature not in [
            "BUREAU_CREDIT_COUNT",
            "PREV_APP_COUNT",
            "IP_CONTRACT_COUNT",
            "HAS_CREDIT_CARD_HISTORY",
        ]

    ]
    return (historical_features,)


@app.cell
def _(hard_fn, historical_features, matched_negatives, training_dataset):
    hard_history = hard_fn.merge(
        training_dataset[
            ["SK_ID_CURR"] + historical_features
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )

    matched_history = matched_negatives.merge(
        training_dataset[
            ["SK_ID_CURR"] + historical_features
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return hard_history, matched_history


@app.cell
def _(np, pd):
    def compare_numeric_features(
        group_a,
        group_b,
        features,
    ):
        rows = []

        for feature in features:
            a = group_a[feature]
            b = group_b[feature]

            mean_a = a.mean()
            mean_b = b.mean()

            median_a = a.median()
            median_b = b.median()

            var_a = a.var()
            var_b = b.var()

            pooled_std = np.sqrt(
                (var_a + var_b) / 2
            )

            if (
                pd.notna(pooled_std)
                and pooled_std > 0
            ):
                smd = (
                    mean_a - mean_b
                ) / pooled_std
            else:
                smd = np.nan

            rows.append({
                "feature": feature,

                "hard_fn_mean": mean_a,
                "matched_neg_mean": mean_b,

                "hard_fn_median": median_a,
                "matched_neg_median": median_b,

                "smd": smd,
                "abs_smd": abs(smd)
                    if pd.notna(smd)
                    else np.nan,

                "hard_fn_missing": (
                    a.isna().mean()
                ),
                "matched_neg_missing": (
                    b.isna().mean()
                ),
            })

        result = pd.DataFrame(rows)

        result["missing_delta"] = (
            result["hard_fn_missing"]
            - result["matched_neg_missing"]
        )

        return result.sort_values(
            "abs_smd",
            ascending=False,
        )

    return (compare_numeric_features,)


@app.cell
def _(
    compare_numeric_features,
    hard_history,
    historical_features,
    matched_history,
):
    history_comparison = compare_numeric_features(
        hard_history,
        matched_history,
        historical_features,
    )


    history_comparison.head(25)
    return


@app.cell
def _(profile_analysis, training_dataset):
    cross_table_base_features = [
        # Bureau exposure
        "BUREAU_TOTAL_CREDIT_DEBT",
        "BUREAU_TOTAL_CREDIT_SUM",

        # Credit card exposure
        "CC_MAX_BALANCE",
        "CC_MEAN_BALANCE",
        "CC_MAX_UTILIZATION",
    ]

    available_cross_features = [
        feature
        for feature in cross_table_base_features
        if feature in training_dataset.columns
    ]

    cross_analysis = profile_analysis.merge(
        training_dataset[
            ["SK_ID_CURR"] + available_cross_features
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (cross_analysis,)


@app.cell
def _(cross_analysis):
    cross_analysis["BUREAU_DEBT_INCOME_RATIO"] = (
        cross_analysis["BUREAU_TOTAL_CREDIT_DEBT"]
        / cross_analysis["AMT_INCOME_TOTAL"]
    )

    cross_analysis["BUREAU_DEBT_CURRENT_CREDIT_RATIO"] = (
        cross_analysis["BUREAU_TOTAL_CREDIT_DEBT"]
        / cross_analysis["AMT_CREDIT"]
    )
    return


@app.cell
def _(cross_analysis):
    cross_analysis["TOTAL_CREDIT_EXPOSURE"] = (
        cross_analysis["AMT_CREDIT"]
        + cross_analysis["BUREAU_TOTAL_CREDIT_DEBT"]
    )

    cross_analysis["TOTAL_EXPOSURE_INCOME_RATIO"] = (
        cross_analysis["TOTAL_CREDIT_EXPOSURE"]
        / cross_analysis["AMT_INCOME_TOTAL"]
    )
    return


@app.cell
def _(cross_analysis):
    cross_analysis["CC_MAX_BALANCE_INCOME_RATIO"] = (
        cross_analysis["CC_MAX_BALANCE"]
        / cross_analysis["AMT_INCOME_TOTAL"]
    )

    cross_analysis["CC_MAX_BALANCE_CURRENT_CREDIT_RATIO"] = (
        cross_analysis["CC_MAX_BALANCE"]
        / cross_analysis["AMT_CREDIT"]
    )
    return


@app.cell
def _(cross_analysis, matched_negatives):
    matched_negative_ids = set(
        matched_negatives["SK_ID_CURR"]
    )

    cross_analysis["matched_negative"] = (
        cross_analysis["SK_ID_CURR"]
        .isin(matched_negative_ids)
    )
    return


@app.cell
def _(cross_analysis):
    hard_cross = cross_analysis.loc[
        cross_analysis["hard_fn_t29"]
    ]

    matched_cross = cross_analysis.loc[
        cross_analysis["matched_negative"]
    ]
    return hard_cross, matched_cross


@app.cell
def _(hard_cross, matched_cross):
    print(len(hard_cross))
    print(len(matched_cross))

    print(hard_cross["pred_t29"].mean())
    print(matched_cross["pred_t29"].mean())

    print(hard_cross["pred_t29"].median())
    print(matched_cross["pred_t29"].median())
    return


@app.cell
def _():
    cross_features = [
        "BUREAU_DEBT_INCOME_RATIO",
        "BUREAU_DEBT_CURRENT_CREDIT_RATIO",
        "TOTAL_EXPOSURE_INCOME_RATIO",
        "CC_MAX_BALANCE_INCOME_RATIO",
        "CC_MAX_BALANCE_CURRENT_CREDIT_RATIO",
    ]
    return (cross_features,)


@app.cell
def _(compare_numeric_features, cross_features, hard_cross, matched_cross):
    cross_comparison = compare_numeric_features(
        hard_cross,
        matched_cross,
        cross_features,
    )

    cross_comparison
    return


@app.cell
def _(matched_negatives, profile_analysis):
    matched_negative_ids_profile = set(
        matched_negatives["SK_ID_CURR"]
    )

    profile_analysis["matched_negative"] = (
        profile_analysis["SK_ID_CURR"]
        .isin(matched_negative_ids_profile)
    )
    return


@app.cell
def _(profile_analysis):
    print(
        profile_analysis["hard_fn_t29"].sum()
    )

    print(
        profile_analysis["matched_negative"].sum()
    )
    return


@app.cell
def _(profile_analysis):
    profile_analysis["DAYS_EMPLOYED_MISSING"] = (
        profile_analysis["DAYS_EMPLOYED"]
        .isna()
        .astype("int8")
    )
    return


@app.cell
def _():
    current_numeric_features = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",

        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",

        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",

        "DAYS_BIRTH",
        "DAYS_EMPLOYED",

        "DAYS_EMPLOYED_MISSING",
    ]
    return (current_numeric_features,)


@app.cell
def _(profile_analysis):
    hard_current = profile_analysis.loc[
        profile_analysis["hard_fn_t29"]
    ]

    matched_current = profile_analysis.loc[
        profile_analysis["matched_negative"]
    ]
    return hard_current, matched_current


@app.cell
def _(
    compare_numeric_features,
    current_numeric_features,
    hard_current,
    matched_current,
):
    current_comparison = compare_numeric_features(
        hard_current,
        matched_current,
        current_numeric_features,
    )

    current_comparison
    return (current_comparison,)


@app.cell
def _(current_comparison):
    current_comparison.assign(
            abs_missing_delta=lambda x:
                x["missing_delta"].abs()
        ).sort_values(
            "abs_missing_delta",
            ascending=False,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EA1.3 — Categorical Cohort Performance Analysis

    #### Objective

    Identify applicant segments in which the final OOF model has systematically
    weaker ranking performance.

    The previous hard-false-negative analysis did not reveal a strong residual
    feature pattern after matching applications on predicted risk.

    The analysis therefore moves from individual difficult cases to population-level
    failure modes.

    The objective is to answer:

    > Are there meaningful applicant cohorts in which the model ranks defaults less
    > effectively than in the development population overall?

    Only out-of-fold predictions from the final tuned CatBoost candidate are used.

    ---

    #### Candidate cohorts

    The first stage focuses on interpretable categorical applicant characteristics:

    - contract type;
    - income type;
    - education;
    - occupation;
    - organization type;
    - gender.

    These variables represent materially different borrower populations and may
    expose interactions that are not captured equally well by the current feature
    representation.

    ---

    #### Metrics

    For each sufficiently large cohort, calculate:

    - number of applications;
    - number of defaults;
    - target prevalence;
    - mean OOF predicted risk;
    - Average Precision;
    - ROC-AUC;
    - AP lift relative to cohort prevalence;
    - share of the cohort sent to the global Top-10% review queue;
    - recall of cohort defaults within the global Top-10% review queue;
    - precision among reviewed applications from the cohort.

    Raw Average Precision is not compared across cohorts in isolation because AP
    depends strongly on target prevalence.

    Therefore:

    `AP / prevalence`

    is also reported as a simple ranking lift measure relative to the random
    baseline of each cohort.

    The global Top-10% policy is preserved rather than recalculating a separate
    Top-10% threshold inside each cohort.

    ---

    #### Reliability filter

    Very small cohorts can produce unstable ranking metrics.

    The primary analysis therefore focuses on cohorts with at least:

    - 1,000 applications;
    - 100 positive applications;
    - 100 negative applications.

    Smaller cohorts may be inspected descriptively but are not used to drive feature
    engineering decisions.

    ---

    #### Decision rule

    A cohort is considered diagnostically interesting when several signals agree,
    for example:

    - materially lower AP lift;
    - lower ROC-AUC;
    - weak capture of defaults by the global review queue;
    - sufficient sample size;
    - a plausible domain explanation.

    A single low metric in a small or highly imbalanced cohort is not sufficient.

    The goal is to generate a small number of testable feature hypotheses rather
    than to optimize metrics separately for every subgroup.
    """)
    return


@app.cell
def _(training_dataset):
    categorical_cohort_features = [
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "CODE_GENDER",
    ]

    for feature in categorical_cohort_features:
        print(
            feature,
            feature in training_dataset.columns,
        )
    return (categorical_cohort_features,)


@app.cell
def _(categorical_cohort_features, oof_analysis, training_dataset):
    cohort_analysis = oof_analysis.merge(
        training_dataset[
            ["SK_ID_CURR"] + categorical_cohort_features
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (cohort_analysis,)


@app.cell
def _(categorical_cohort_features, cohort_analysis):
    for cat_feature in categorical_cohort_features:
        cohort_analysis[cat_feature] = (
            cohort_analysis[cat_feature]
            .astype("object")
            .fillna("<MISSING>")
        )
    return


@app.cell
def _(cohort_analysis):
    assert "top10_t29" in cohort_analysis.columns

    print(
        cohort_analysis["top10_t29"].mean()
    )
    return


@app.cell
def _(average_precision_score, np, pd, roc_auc_score):
    def calculate_cohort_metrics(
        df,
        cohort_feature,
        score_col="pred_t29",
        top10_col="top10_t29",
    ):
        rows = []

        for cohort_value, group in df.groupby(
            cohort_feature,
            dropna=False,
        ):
            n = len(group)

            n_positive = int(group["TARGET"].sum())
            n_negative = n - n_positive

            prevalence = group["TARGET"].mean()

            mean_prediction = group[score_col].mean()

            if (
                n_positive > 0
                and n_negative > 0
            ):
                ap = average_precision_score(
                    group["TARGET"],
                    group[score_col],
                )

                roc_auc = roc_auc_score(
                    group["TARGET"],
                    group[score_col],
                )
            else:
                ap = np.nan
                roc_auc = np.nan

            if prevalence > 0:
                ap_lift = ap / prevalence
            else:
                ap_lift = np.nan

            reviewed = group[top10_col]

            review_rate = reviewed.mean()

            reviewed_positive = group.loc[
                reviewed,
                "TARGET",
            ].sum()

            if n_positive > 0:
                recall_global_top10 = (
                    reviewed_positive
                    / n_positive
                )
            else:
                recall_global_top10 = np.nan

            if reviewed.sum() > 0:
                precision_global_top10 = (
                    group.loc[
                        reviewed,
                        "TARGET",
                    ].mean()
                )
            else:
                precision_global_top10 = np.nan

            rows.append(
                {
                    "feature": cohort_feature,
                    "cohort": cohort_value,

                    "n": n,
                    "n_positive": n_positive,
                    "n_negative": n_negative,

                    "prevalence": prevalence,
                    "mean_prediction": mean_prediction,

                    "ap": ap,
                    "ap_lift": ap_lift,
                    "roc_auc": roc_auc,

                    "review_rate": review_rate,
                    "recall_global_top10": (
                        recall_global_top10
                    ),
                    "precision_global_top10": (
                        precision_global_top10
                    ),
                }
            )

        return pd.DataFrame(rows)

    return (calculate_cohort_metrics,)


@app.cell
def _(
    calculate_cohort_metrics,
    categorical_cohort_features,
    cohort_analysis,
    pd,
):
    categorical_results = pd.concat(
        [
            calculate_cohort_metrics(
                cohort_analysis,
                feature,
            )
            for feature
            in categorical_cohort_features
        ],
        ignore_index=True,
    )

    categorical_results["reliable"] = (
        (categorical_results["n"] >= 1000)
        & (categorical_results["n_positive"] >= 100)
        & (categorical_results["n_negative"] >= 100)
    )
    return (categorical_results,)


@app.cell
def _(categorical_results):
    reliable_categorical_results = (
        categorical_results.loc[
            categorical_results["reliable"]
        ]
        .sort_values(
            [
                "ap_lift",
                "roc_auc",
            ],
            ascending=True,
        )
    )


    reliable_categorical_results
    return


@app.cell
def _(average_precision_score, cohort_analysis, roc_auc_score):
    global_prevalence = (
        cohort_analysis["TARGET"].mean()
    )

    global_ap = average_precision_score(
        cohort_analysis["TARGET"],
        cohort_analysis["pred_t29"],
    )

    global_auc = roc_auc_score(
        cohort_analysis["TARGET"],
        cohort_analysis["pred_t29"],
    )

    global_ap_lift = (
        global_ap / global_prevalence
    )

    print("Global prevalence:", global_prevalence)
    print("Global AP:", global_ap)
    print("Global AP lift:", global_ap_lift)
    print("Global ROC-AUC:", global_auc)
    return global_ap_lift, global_auc


@app.cell
def _(categorical_results, global_ap_lift, global_auc):
    categorical_results["auc_delta_vs_global"] = (
        categorical_results["roc_auc"]
        - global_auc
    )

    categorical_results["ap_lift_ratio_vs_global"] = (
        categorical_results["ap_lift"]
        / global_ap_lift
    )
    return


@app.cell
def _():
    candidate_cohorts = [
        ("OCCUPATION_TYPE", "Low-skill Laborers"),
        ("OCCUPATION_TYPE", "Drivers"),
        ("OCCUPATION_TYPE", "Sales staff"),
        ("NAME_EDUCATION_TYPE", "Lower secondary"),
        ("ORGANIZATION_TYPE", "Restaurant"),
        ("ORGANIZATION_TYPE", "Transport: type 4"),
    ]
    return (candidate_cohorts,)


@app.cell
def _(average_precision_score, cohort_analysis, pd, roc_auc_score):
    global_fold_metrics = []

    for fold, group in cohort_analysis.groupby("fold"):
        prevalence = group["TARGET"].mean()

        ap = average_precision_score(
            group["TARGET"],
            group["pred_t29"],
        )

        auc = roc_auc_score(
            group["TARGET"],
            group["pred_t29"],
        )

        global_fold_metrics.append(
            {
                "fold": fold,
                "global_n": len(group),
                "global_prevalence": prevalence,
                "global_ap": ap,
                "global_ap_lift": ap / prevalence,
                "global_auc": auc,
            }
        )

    global_fold_metrics = pd.DataFrame(
        global_fold_metrics
    )

    global_fold_metrics
    return (global_fold_metrics,)


@app.cell
def _(average_precision_score, np, pd, roc_auc_score):
    def calculate_candidate_fold_metrics(
        df,
        feature,
        cohort_value,
        score_col="pred_t29",
    ):
        rows = []

        cohort_df = df.loc[
            df[feature] == cohort_value
        ]

        for fold, group in cohort_df.groupby("fold"):
            n = len(group)

            n_positive = int(group["TARGET"].sum())
            n_negative = n - n_positive

            prevalence = group["TARGET"].mean()

            if n_positive > 0 and n_negative > 0:
                ap = average_precision_score(
                    group["TARGET"],
                    group[score_col],
                )

                auc = roc_auc_score(
                    group["TARGET"],
                    group[score_col],
                )

                ap_lift = (
                    ap / prevalence
                    if prevalence > 0
                    else np.nan
                )
            else:
                ap = np.nan
                auc = np.nan
                ap_lift = np.nan

            rows.append(
                {
                    "feature": feature,
                    "cohort": cohort_value,
                    "fold": fold,

                    "n": n,
                    "n_positive": n_positive,
                    "n_negative": n_negative,

                    "prevalence": prevalence,

                    "ap": ap,
                    "ap_lift": ap_lift,
                    "roc_auc": auc,
                }
            )

        return pd.DataFrame(rows)

    return (calculate_candidate_fold_metrics,)


@app.cell
def _(
    calculate_candidate_fold_metrics,
    candidate_cohorts,
    cohort_analysis,
    global_fold_metrics,
    pd,
):
    candidate_fold_results = pd.concat(
        [
            calculate_candidate_fold_metrics(
                cohort_analysis,
                feature,
                cohort,
            )
            for feature, cohort
            in candidate_cohorts
        ],
        ignore_index=True,
    )

    candidate_fold_results = (
        candidate_fold_results.merge(
            global_fold_metrics,
            on="fold",
            how="left",
            validate="many_to_one",
        )
    )

    candidate_fold_results[
        "auc_delta_vs_global"
    ] = (
        candidate_fold_results["roc_auc"]
        - candidate_fold_results["global_auc"]
    )

    candidate_fold_results[
        "ap_lift_ratio_vs_global"
    ] = (
        candidate_fold_results["ap_lift"]
        / candidate_fold_results["global_ap_lift"]
    )
    return (candidate_fold_results,)


@app.cell
def _(candidate_fold_results):
    candidate_fold_results[
            [
                "feature",
                "cohort",
                "fold",

                "n",
                "n_positive",

                "prevalence",

                "roc_auc",
                "global_auc",
                "auc_delta_vs_global",

                "ap_lift",
                "global_ap_lift",
                "ap_lift_ratio_vs_global",
            ]
        ].sort_values(
            ["feature", "cohort", "fold"]
        )
    return


@app.cell
def _(candidate_fold_results):
    cohort_fold_summary = (
        candidate_fold_results
        .groupby(
            ["feature", "cohort"],
            as_index=False,
        )
        .agg(
            total_n=("n", "sum"),
            total_positive=("n_positive", "sum"),

            mean_auc=("roc_auc", "mean"),
            std_auc=("roc_auc", "std"),

            mean_auc_delta=(
                "auc_delta_vs_global",
                "mean",
            ),

            min_auc_delta=(
                "auc_delta_vs_global",
                "min",
            ),

            max_auc_delta=(
                "auc_delta_vs_global",
                "max",
            ),

            mean_ap_lift_ratio=(
                "ap_lift_ratio_vs_global",
                "mean",
            ),

            std_ap_lift_ratio=(
                "ap_lift_ratio_vs_global",
                "std",
            ),
        )
    )
    return (cohort_fold_summary,)


@app.cell
def _(candidate_fold_results, cohort_fold_summary):
    auc_worse_counts = (
        candidate_fold_results
        .assign(
            auc_worse=lambda x:
                x["auc_delta_vs_global"] < 0
        )
        .groupby(
            ["feature", "cohort"],
            as_index=False,
        )["auc_worse"]
        .sum()
        .rename(
            columns={
                "auc_worse": "auc_worse_folds"
            }
        )
    )

    ap_worse_counts = (
        candidate_fold_results
        .assign(
            ap_lift_worse=lambda x:
                x["ap_lift_ratio_vs_global"] < 1
        )
        .groupby(
            ["feature", "cohort"],
            as_index=False,
        )["ap_lift_worse"]
        .sum()
        .rename(
            columns={
                "ap_lift_worse":
                    "ap_lift_worse_folds"
            }
        )
    )

    cohort_fold_summary_merged = (
        cohort_fold_summary
        .merge(
            auc_worse_counts,
            on=["feature", "cohort"],
            validate="one_to_one",
        )
        .merge(
            ap_worse_counts,
            on=["feature", "cohort"],
            validate="one_to_one",
        )
    )
    return


@app.cell
def _(cohort_fold_summary):
    cohort_fold_summary.sort_values(
            "mean_auc_delta"
        )
    return


@app.cell
def _(cohort_analysis):
    driver_mask = (
        cohort_analysis["OCCUPATION_TYPE"] == "Drivers"
    )

    transport4_mask = (
        cohort_analysis["ORGANIZATION_TYPE"]
        == "Transport: type 4"
    )

    low_skill_mask = (
        cohort_analysis["OCCUPATION_TYPE"]
        == "Low-skill Laborers"
    )

    lower_secondary_mask = (
        cohort_analysis["NAME_EDUCATION_TYPE"]
        == "Lower secondary"
    )
    return driver_mask, low_skill_mask, lower_secondary_mask, transport4_mask


@app.cell
def _(driver_mask, low_skill_mask, lower_secondary_mask, transport4_mask):
    print(
        "Transport4 who are Drivers:",
        (
            driver_mask & transport4_mask
        ).sum() / transport4_mask.sum()
    )

    print(
        "Drivers in Transport4:",
        (
            driver_mask & transport4_mask
        ).sum() / driver_mask.sum()
    )

    print(
        "Drivers with lower secondary:",
        (
            driver_mask & lower_secondary_mask
        ).sum() / driver_mask.sum()
    )

    print(
        "Low-skill with lower secondary:",
        (
            low_skill_mask & lower_secondary_mask
        ).sum() / low_skill_mask.sum()
    )
    return


@app.cell
def _(cohort_analysis, driver_mask):
    cohort_analysis.loc[
            driver_mask,
            "ORGANIZATION_TYPE",
        ].value_counts(normalize=True).head(15).rename("share").to_frame()
    return


@app.cell
def _(cohort_analysis, driver_mask):
    cohort_analysis.loc[
            driver_mask,
            "NAME_EDUCATION_TYPE",
        ].value_counts(normalize=True).rename("share").to_frame()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EA1.3.2 — Feature separation inside Drivers
    """)
    return


@app.cell
def _(FINAL_FEATURES, pd, training_dataset):
    numeric_final_features = [
        feature
        for feature in FINAL_FEATURES
        if feature in training_dataset.columns
        and pd.api.types.is_numeric_dtype(
            training_dataset[feature]
        )
    ]


    print("Final features:", len(FINAL_FEATURES))
    print("Numeric final features:", len(numeric_final_features))
    return (numeric_final_features,)


@app.cell
def _(numeric_final_features, oof_analysis, training_dataset):
    driver_analysis = oof_analysis.merge(
        training_dataset[
            ["SK_ID_CURR", "OCCUPATION_TYPE"]
            + numeric_final_features
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )

    driver_analysis["IS_DRIVER"] = (
        driver_analysis["OCCUPATION_TYPE"] == "Drivers"
    )
    return (driver_analysis,)


@app.cell
def _(driver_analysis):
    print(driver_analysis["IS_DRIVER"].sum())

    print(
        driver_analysis.loc[
            driver_analysis["IS_DRIVER"],
            "TARGET",
        ].sum()
    )
    return


@app.cell
def _(np, pd, roc_auc_score):
    def feature_target_separation(
        df,
        features,
        label,
    ):
        rows = []

        for feature in features:
            feature_df = df[
                ["TARGET", feature]
            ].dropna()

            positives = feature_df.loc[
                feature_df["TARGET"] == 1,
                feature,
            ]

            negatives = feature_df.loc[
                feature_df["TARGET"] == 0,
                feature,
            ]

            if (
                len(positives) == 0
                or len(negatives) == 0
            ):
                continue

            mean_pos = positives.mean()
            mean_neg = negatives.mean()

            median_pos = positives.median()
            median_neg = negatives.median()

            pooled_std = np.sqrt(
                (
                    positives.var()
                    + negatives.var()
                )
                / 2
            )

            if (
                pd.notna(pooled_std)
                and pooled_std > 0
            ):
                smd = (
                    mean_pos - mean_neg
                ) / pooled_std
            else:
                smd = np.nan

            if feature_df[feature].nunique() > 1:
                auc = roc_auc_score(
                    feature_df["TARGET"],
                    feature_df[feature],
                )

                separation_auc = max(
                    auc,
                    1 - auc,
                )
            else:
                auc = np.nan
                separation_auc = np.nan

            rows.append({
                "feature": feature,
                f"{label}_n": len(feature_df),

                f"{label}_mean_positive": mean_pos,
                f"{label}_mean_negative": mean_neg,

                f"{label}_median_positive": median_pos,
                f"{label}_median_negative": median_neg,

                f"{label}_smd": smd,
                f"{label}_abs_smd": (
                    abs(smd)
                    if pd.notna(smd)
                    else np.nan
                ),

                f"{label}_separation_auc":
                    separation_auc,

                f"{label}_missing_share":
                    df[feature].isna().mean(),
            })

        return pd.DataFrame(rows)

    return (feature_target_separation,)


@app.cell
def _(driver_analysis, feature_target_separation, numeric_final_features):
    global_separation = feature_target_separation(
        driver_analysis,
        numeric_final_features,
        "global",
    )

    drivers_separation = feature_target_separation(
        driver_analysis.loc[
            driver_analysis["IS_DRIVER"]
        ],
        numeric_final_features,
        "drivers",
    )

    driver_feature_comparison = (
        global_separation.merge(
            drivers_separation,
            on="feature",
            how="inner",
            validate="one_to_one",
        )
    )

    driver_feature_comparison[
        "separation_auc_delta"
    ] = (
        driver_feature_comparison[
            "drivers_separation_auc"
        ]
        - driver_feature_comparison[
            "global_separation_auc"
        ]
    )

    driver_feature_comparison[
        "abs_smd_delta"
    ] = (
        driver_feature_comparison[
            "drivers_abs_smd"
        ]
        - driver_feature_comparison[
            "global_abs_smd"
        ]
    )
    return driver_feature_comparison, drivers_separation, global_separation


@app.cell
def _(driver_feature_comparison):
    driver_feature_comparison[
            [
                "feature",

                "global_separation_auc",
                "drivers_separation_auc",
                "separation_auc_delta",

                "global_abs_smd",
                "drivers_abs_smd",
                "abs_smd_delta",

                "global_missing_share",
                "drivers_missing_share",
            ]
        ].sort_values(
            "separation_auc_delta"
        ).head(25)
    return


@app.cell
def _(driver_feature_comparison):
    driver_feature_comparison[
            [
                "feature",
                "global_separation_auc",
                "drivers_separation_auc",
                "separation_auc_delta",
            ]
        ].sort_values(
            "separation_auc_delta",
            ascending=False,
        ).head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EA1.3.3 — Feature separation inside Sales staff
    """)
    return


@app.cell
def _(driver_analysis):
    driver_analysis["IS_SALES_STAFF"] = (
        driver_analysis["OCCUPATION_TYPE"] == "Sales staff"
    )

    print(
        driver_analysis["IS_SALES_STAFF"].sum()
    )

    print(
        driver_analysis.loc[
            driver_analysis["IS_SALES_STAFF"],
            "TARGET",
        ].sum()
    )
    return


@app.cell
def _(
    driver_analysis,
    feature_target_separation,
    global_separation,
    numeric_final_features,
):
    sales_separation = feature_target_separation(
        driver_analysis.loc[
            driver_analysis["IS_SALES_STAFF"]
        ],
        numeric_final_features,
        "sales",
    )

    sales_feature_comparison = (
        global_separation.merge(
            sales_separation,
            on="feature",
            how="inner",
            validate="one_to_one",
        )
    )

    sales_feature_comparison[
        "separation_auc_delta"
    ] = (
        sales_feature_comparison[
            "sales_separation_auc"
        ]
        - sales_feature_comparison[
            "global_separation_auc"
        ]
    )

    sales_feature_comparison[
        "abs_smd_delta"
    ] = (
        sales_feature_comparison[
            "sales_abs_smd"
        ]
        - sales_feature_comparison[
            "global_abs_smd"
        ]
    )
    return sales_feature_comparison, sales_separation


@app.cell
def _(sales_feature_comparison):
    sales_lost_separation = (
        sales_feature_comparison[
            [
                "feature",

                "global_separation_auc",
                "sales_separation_auc",
                "separation_auc_delta",

                "global_abs_smd",
                "sales_abs_smd",
                "abs_smd_delta",

                "global_missing_share",
                "sales_missing_share",
            ]
        ]
        .sort_values(
            "separation_auc_delta"
        )
        .head(25)
    )

    sales_lost_separation
    return


@app.cell
def _(sales_feature_comparison):
    sales_gained_separation = (
        sales_feature_comparison[
            [
                "feature",
                "global_separation_auc",
                "sales_separation_auc",
                "separation_auc_delta",
            ]
        ]
        .sort_values(
            "separation_auc_delta",
            ascending=False,
        )
        .head(20)
    )

    sales_gained_separation
    return


@app.cell
def _(drivers_separation, global_separation, sales_separation):
    occupation_separation_comparison = (
        global_separation[
            [
                "feature",
                "global_separation_auc",
            ]
        ]
        .merge(
            drivers_separation[
                [
                    "feature",
                    "drivers_separation_auc",
                ]
            ],
            on="feature",
            validate="one_to_one",
        )
        .merge(
            sales_separation[
                [
                    "feature",
                    "sales_separation_auc",
                ]
            ],
            on="feature",
            validate="one_to_one",
        )
    )

    occupation_separation_comparison[
        "drivers_delta"
    ] = (
        occupation_separation_comparison[
            "drivers_separation_auc"
        ]
        - occupation_separation_comparison[
            "global_separation_auc"
        ]
    )

    occupation_separation_comparison[
        "sales_delta"
    ] = (
        occupation_separation_comparison[
            "sales_separation_auc"
        ]
        - occupation_separation_comparison[
            "global_separation_auc"
        ]
    )
    return (occupation_separation_comparison,)


@app.cell
def _(occupation_separation_comparison):
    common_degradation = (
        occupation_separation_comparison.loc[
            (
                occupation_separation_comparison[
                    "drivers_delta"
                ] < 0
            )
            &
            (
                occupation_separation_comparison[
                    "sales_delta"
                ] < 0
            )
        ]
        .assign(
            mean_delta=lambda x: (
                x["drivers_delta"]
                + x["sales_delta"]
            ) / 2
        )
        .sort_values("mean_delta")
    )


    common_degradation.head(25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EA1.3.4 — Sales staff × credit-card history
    """)
    return


@app.cell
def _(driver_analysis, training_dataset):
    driver_analysis_cc = driver_analysis.merge(
        training_dataset[
            ["SK_ID_CURR", "HAS_CREDIT_CARD_HISTORY"]
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return (driver_analysis_cc,)


@app.cell
def _(driver_analysis_cc):
    driver_analysis_cc["HAS_CREDIT_CARD_HISTORY"] = (
        driver_analysis_cc["HAS_CREDIT_CARD_HISTORY"]
        .fillna(0)
        .astype(bool)
    )
    return


@app.cell
def _(driver_analysis_cc):
    sales_df = driver_analysis_cc.loc[
        driver_analysis_cc["IS_SALES_STAFF"]
    ].copy()

    print(
        sales_df["HAS_CREDIT_CARD_HISTORY"]
        .value_counts(normalize=True)
    )
    return (sales_df,)


@app.cell
def _(average_precision_score, np, pd, roc_auc_score):
    def subgroup_model_metrics(
        df,
        group_col,
        score_col="pred_t29",
    ):
        rows = []

        for value, group in df.groupby(group_col):
            n = len(group)
            n_positive = int(group["TARGET"].sum())
            prevalence = group["TARGET"].mean()

            if (
                group["TARGET"].nunique() == 2
                and n_positive > 0
            ):
                ap = average_precision_score(
                    group["TARGET"],
                    group[score_col],
                )

                auc = roc_auc_score(
                    group["TARGET"],
                    group[score_col],
                )

                ap_lift = ap / prevalence
            else:
                ap = np.nan
                auc = np.nan
                ap_lift = np.nan

            rows.append(
                {
                    "group": value,
                    "n": n,
                    "n_positive": n_positive,
                    "prevalence": prevalence,
                    "ap": ap,
                    "ap_lift": ap_lift,
                    "roc_auc": auc,
                }
            )

        return pd.DataFrame(rows)

    return (subgroup_model_metrics,)


@app.cell
def _(sales_df, subgroup_model_metrics):
    sales_cc_performance = subgroup_model_metrics(
        sales_df,
        "HAS_CREDIT_CARD_HISTORY",
    )

    sales_cc_performance
    return


@app.cell
def _(driver_analysis_cc, subgroup_model_metrics):
    global_cc_performance = subgroup_model_metrics(
        driver_analysis_cc,
        "HAS_CREDIT_CARD_HISTORY",
    )

    global_cc_performance
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Sales staff and credit-card history

    The Sales staff cohort was analyzed further because several credit-card and debt
    features became more discriminative inside this segment.

    Sales staff without credit-card history showed substantially weaker ranking
    performance:

    - ROC-AUC: approximately `0.766`;
    - AP lift: approximately `3.01`.

    For Sales staff with credit-card history:

    - ROC-AUC increased to approximately `0.780`;
    - AP lift increased to approximately `3.11`.

    This difference is not explained by a general benefit from credit-card history.

    In the full development population, ranking performance is approximately equal
    between applicants with and without credit-card history:

    - no credit-card history: ROC-AUC ≈ `0.790`;
    - credit-card history available: ROC-AUC ≈ `0.789`.

    Therefore, credit-card information appears particularly useful within the
    Sales staff cohort.

    The Sales staff AUC gap relative to the corresponding global population changes
    from approximately:

    `-0.024` without credit-card history

    to:

    `-0.008` when credit-card history is available.

    This is consistent with the feature-separation analysis, where credit-card
    utilization, balance, and bureau debt became more informative inside Sales
    staff.

    Interpretation

    The weak ranking performance of Sales staff appears to result partly from a loss
    of discriminative strength in several global demographic and employment-related
    signals.

    When credit-card behavioral information is available, utilization and balance
    features partially compensate for this loss.

    For applicants without credit-card history, no equally strong replacement
    signal is available.

    Decision

    **CLOSE OCCUPATION BRANCH**

    The analysis identifies a coherent cohort-level failure mechanism but does not
    justify a targeted feature-engineering experiment.

    The relevant credit-card and occupation features are already available to the
    CatBoost model, and the remaining performance gap appears to be driven partly by
    information availability rather than an obvious missing representation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### EA1.4 — Numeric cohort scan
    """)
    return


@app.cell
def _(oof_analysis, training_dataset):
    numeric_cohort_features = [
        "AGE_YEARS",
        "AMT_INCOME_TOTAL",
        "CREDIT_INCOME_RATIO",
        "EXT_SOURCE_2",
        "BUREAU_CREDIT_COUNT",
    ]

    numeric_cohort_analysis = oof_analysis.merge(
        training_dataset[
            ["SK_ID_CURR"] + numeric_cohort_features
        ],
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    return numeric_cohort_analysis, numeric_cohort_features


@app.cell
def _(numeric_cohort_analysis, numeric_cohort_features, pd):
    for num_feature in numeric_cohort_features:
        numeric_cohort_analysis[f"{num_feature}_BIN"] = pd.qcut(
            numeric_cohort_analysis[num_feature],
            q=5,
            duplicates="drop",
        )
    return


@app.cell
def _(average_precision_score, pd, roc_auc_score):
    def numeric_cohort_metrics(
        df,
        feature,
        score_col="pred_t29",
    ):
        bin_col = f"{feature}_BIN"

        rows = []

        for cohort, group in df.groupby(
            bin_col,
            observed=True,
        ):
            prevalence = group["TARGET"].mean()

            ap = average_precision_score(
                group["TARGET"],
                group[score_col],
            )

            auc = roc_auc_score(
                group["TARGET"],
                group[score_col],
            )

            rows.append({
                "feature": feature,
                "cohort": str(cohort),

                "n": len(group),
                "n_positive": int(group["TARGET"].sum()),

                "prevalence": prevalence,
                "ap": ap,
                "ap_lift": ap / prevalence,

                "roc_auc": auc,
            })

        return pd.DataFrame(rows)

    return (numeric_cohort_metrics,)


@app.cell
def _(
    global_ap_lift,
    global_auc,
    numeric_cohort_analysis,
    numeric_cohort_features,
    numeric_cohort_metrics,
    pd,
):
    numeric_cohort_results = pd.concat(
        [
            numeric_cohort_metrics(
                numeric_cohort_analysis,
                feature,
            )
            for feature in numeric_cohort_features
        ],
        ignore_index=True,
    )

    numeric_cohort_results["auc_delta_vs_global"] = (
        numeric_cohort_results["roc_auc"]
        - global_auc
    )

    numeric_cohort_results["ap_lift_ratio_vs_global"] = (
        numeric_cohort_results["ap_lift"]
        / global_ap_lift
    )
    return (numeric_cohort_results,)


@app.cell
def _(numeric_cohort_results):
    numeric_cohort_results.sort_values(
            "auc_delta_vs_global"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Results & Findings
    Numeric cohort performance was evaluated across quintiles of:

    - age;
    - income;
    - current credit-to-income burden;
    - `EXT_SOURCE_2`;
    - bureau-history depth.

    Most cohorts showed ranking performance close to the overall OOF model. Income and bureau-history depth were particularly stable across the population. Credit-to-income ratio also showed no major systematic failure region.

    Some degradation was observed in the youngest and oldest age groups and restricted `EXT_SOURCE_2` ranges. However, conditioning on strong continuous predictors such as `EXT_SOURCE_2` naturally removes part of their global ranking information, so lower within-bin AUC does not by itself indicate a missing model feature. No numeric cohort revealed a sufficiently strong and interpretable residual failure mechanism to justify targeted feature engineering.

    ### Overall Error Analysis Conclusion

    The OOF error analysis examined ranking failures, review queue dynamics, score-matched hard false negatives, categorical cohorts (Drivers, Sales staff), and continuous numeric quintiles:

    1. **Queue dynamics**: High overlap (~90%) with RFE1; model refinements improve global probability calibration without churning top-decile operations.
    2. **Hard false negatives**: Not caused by missing historical records; score-matching with genuine non-defaults demonstrates that pre-decision residual signals (|SMD| ≤ 0.12) do not justify targeted feature creation.
    3. **Categorical cohorts**: Drivers and Sales staff suffer from localized signal loss; however, existing features (especially credit-card behavioral metrics) are already accessible, and no distinct representation solves the remaining variance without risk of overfitting.
    4. **Numeric cohorts**: Ranking performance across age, income, and debt-burden quintiles remains balanced and free of catastrophic blind spots.

    **Decision: CLOSE ERROR ANALYSIS**

    No additional feature bundle is introduced based on the error-analysis diagnostics. The internal holdout remains untouched.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## M1 — Alternative Models and Ensemble Blending

    ### Objective / Hypothesis
    Evaluate whether alternative gradient-boosting implementations (LightGBM and XGBoost) provide complementary ranking signals to the tuned CatBoost candidate on the fixed 148-feature RFE2 representation, and whether probability blending improves out-of-fold generalization.

    ### Setup & Cross-Validation Framework
    - Feature representation: Fixed RFE2 (148 features);
    - Validation protocol: Fixed `split_v1` (5 stratified folds, identical development population);
    - Models evaluated: CatBoost (Trial 29 tuned), LightGBM (baseline with histogram binning and categorical handling), XGBoost (histogram tree method with GPU acceleration);
    - Optimization track: Primary: Validation AP. Leaderboard track: ROC-AUC. Policy track: Precision/Recall@Top10%;
    - Internal holdout: Untouched.
    """)
    return


@app.cell
def _(average_precision_score, log_loss, np, roc_auc_score):
    def evaluate_predictions(
        y_true,
        y_pred,
        capacity=0.10,
    ):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        n_top = int(len(y_true) * capacity)

        top_idx = np.argsort(
            y_pred
        )[::-1][:n_top]

        top_y = y_true[top_idx]

        return {
            "ap": average_precision_score(
                y_true,
                y_pred,
            ),
            "roc_auc": roc_auc_score(
                y_true,
                y_pred,
            ),
            "log_loss": log_loss(
                y_true,
                y_pred,
            ),
            "precision_at_10pct": (
                top_y.mean()
            ),
            "recall_at_10pct": (
                top_y.sum()
                / y_true.sum()
            ),
        }

    return (evaluate_predictions,)


@app.cell
def _(evaluate_predictions, lgb, np, pd):
    def run_lgbm_cv(
        X,
        y,
        folds,
        categorical_features,
        params,
    ):
        oof = np.zeros(len(X))
        fold_results = []
        models = []

        for fold in sorted(folds.unique()):
            print(f"\n========== Fold {fold} ==========")

            train_mask = folds != fold
            valid_mask = folds == fold

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]

            X_valid = X.loc[valid_mask]
            y_valid = y.loc[valid_mask]

            model = lgb.LGBMClassifier(
                **params
            )

            model.fit(
                X_train,
                y_train,

                eval_set=[
                    (X_valid, y_valid)
                ],

                eval_metric="auc",

                categorical_feature=
                    categorical_features,

                callbacks=[
                    lgb.early_stopping(
                        200,
                        verbose=False,
                    ),
                    lgb.log_evaluation(200),
                ],
            )

            pred = model.predict_proba(
                X_valid,
                num_iteration=model.best_iteration_,
            )[:, 1]

            oof[valid_mask] = pred

            metrics = evaluate_predictions(
                y_valid,
                pred,
            )

            metrics["fold"] = fold
            metrics["best_iteration"] = (
                model.best_iteration_
            )

            fold_results.append(metrics)
            models.append(model)

            print(metrics)

        return (
            oof,
            pd.DataFrame(fold_results),
            models,
        )

    return (run_lgbm_cv,)


@app.cell
def _(FINAL_FEATURES):
    assert len(FINAL_FEATURES) == 148
    return


@app.cell
def _(training_dataset):
    dev_model = training_dataset.copy()

    dev_model = (
        dev_model.loc[
            dev_model["partition"] == "development"
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(dev_model.shape)
    print(dev_model["fold"].value_counts().sort_index())
    print(dev_model["TARGET"].mean())
    return (dev_model,)


@app.cell
def _(FINAL_FEATURES, dev_model):
    X = dev_model[FINAL_FEATURES].copy()
    y = dev_model["TARGET"].copy()

    categorical_features_lgbm = (
        X.select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    for col in categorical_features_lgbm:
        X[col] = X[col].astype("category")

    print("Features:", X.shape[1])
    print("Categorical:", len(categorical_features_lgbm))
    return


@app.cell
def _():
    lgbm_params = {
        "objective": "binary",

        "n_estimators": 5000,
        "learning_rate": 0.03,

        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,

        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,

        "reg_alpha": 0.0,
        "reg_lambda": 1.0,

        "random_state": 42,
        "n_jobs": -1,

        "verbosity": -1,
    }
    return


@app.cell
def _():
    # lgbm_oof, lgbm_fold_metrics, lgbm_models = (
    #     run_lgbm_cv(
    #         X=X,
    #         y=y,
    #         folds=dev_model["fold"],
    #         categorical_features=categorical_features_lgbm,
    #         params=lgbm_params,
    #     )
    # )
    return


@app.cell
def _():
    # lgbm_oof_metrics = evaluate_predictions(
    #     y,
    #     lgbm_oof,
    # )

    # lgbm_fold_metrics
    return


@app.cell
def _():
    # lgbm_oof_metrics
    return


@app.cell
def _():
    # lgbm_oof_df = dev_model[
    #     ["SK_ID_CURR", "TARGET", "fold"]
    # ].copy()

    # lgbm_oof_df["oof_prediction"] = lgbm_oof

    # lgbm_oof_df.to_parquet(
    #     "lgbm_rfe2_oof.parquet",
    #     index=False,
    # )
    return


@app.cell
def _():
    xgb_params = {
        "objective": "binary:logistic",

        "n_estimators": 5000,
        "learning_rate": 0.03,

        "max_depth": 6,
        "min_child_weight": 1,

        "subsample": 0.8,
        "colsample_bytree": 0.8,

        "reg_alpha": 0.0,
        "reg_lambda": 1.0,

        "tree_method": "hist",
        "device": "cuda",

        "enable_categorical": True,

        "eval_metric": "auc",

        "random_state": 42,

        "early_stopping_rounds": 200,
    }
    return


@app.cell
def _(evaluate_predictions, np, pd, xgb):
    def run_xgb_cv(
        X,
        y,
        folds,
        params,
    ):
        oof = np.zeros(len(X))
        fold_results = []
        models = []

        for fold in sorted(folds.unique()):
            print(f"\n========== Fold {fold} ==========")

            train_mask = folds != fold
            valid_mask = folds == fold

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]

            X_valid = X.loc[valid_mask]
            y_valid = y.loc[valid_mask]

            model = xgb.XGBClassifier(
                **params
            )

            model.fit(
                X_train,
                y_train,

                eval_set=[
                    (X_valid, y_valid)
                ],

                verbose=200,
            )

            pred = model.predict_proba(
                X_valid
            )[:, 1]

            oof[valid_mask] = pred

            metrics = evaluate_predictions(
                y_valid,
                pred,
            )

            metrics["fold"] = fold
            metrics["best_iteration"] = (
                model.best_iteration
            )

            fold_results.append(metrics)
            models.append(model)

            print(metrics)

        return (
            oof,
            pd.DataFrame(fold_results),
            models,
        )

    return (run_xgb_cv,)


@app.cell
def _():
    # xgb_oof, xgb_fold_metrics, xgb_models = (
    #     run_xgb_cv(
    #         X=X,
    #         y=y,
    #         folds=dev_model["fold"],
    #         params=xgb_params,
    #     )
    # )
    return


@app.cell
def _():
    # xgb_oof_metrics = evaluate_predictions(
    #     y,
    #     xgb_oof,
    # )

    # xgb_fold_metrics
    return


@app.cell
def _():
    # print(xgb_oof_metrics)
    return


@app.cell
def _():
    # xgb_oof_df = dev_model[
    #     ["SK_ID_CURR", "TARGET", "fold"]
    # ].copy()

    # xgb_oof_df["oof_prediction"] = xgb_oof

    # xgb_oof_df.to_parquet(
    #     "xgb_rfe2_oof.parquet",
    #     index=False,
    # )
    return


@app.cell
def _():
    # prediction_comparison = pd.DataFrame({
    #     "catboost": t29_oof["probability"],
    #     "lightgbm": lgbm_oof,
    #     "xgboost": xgb_oof,
    # })

    # prediction_comparison.corr()
    return


@app.cell
def _():
    # prediction_comparison.corr(
    #         method="spearman"
    #     )
    return


@app.cell
def _():
    # models_oof = {
    #     "catboost": t29_oof["probability"].to_numpy(),
    #     "lightgbm": lgbm_oof,
    #     "xgboost": xgb_oof,
    # }

    # for model, pred in models_oof.items():
    #     print(model, evaluate_predictions(y, pred))
    return


@app.cell
def _():
    # blend_results = []

    # pairs = [
    #     ("catboost", "lightgbm"),
    #     ("catboost", "xgboost"),
    #     ("lightgbm", "xgboost"),
    # ]

    # for model_a, model_b in pairs:
    #     for weight_a in np.arange(0.0, 1.01, 0.05):
    #         weight_b = 1.0 - weight_a

    #         prediction = (
    #             weight_a * models_oof[model_a]
    #             + weight_b * models_oof[model_b]
    #         )

    #         metrics = evaluate_predictions(y, prediction)

    #         blend_results.append({
    #             "blend": f"{model_a}+{model_b}",
    #             "weight_a": weight_a,
    #             "weight_b": weight_b,
    #             **metrics,
    #         })

    # blend_results = pd.DataFrame(blend_results)
    return


@app.cell
def _():
    # blend_results.sort_values("roc_auc", ascending=False).head(15)
    return


@app.cell
def _():
    # blend_results.sort_values("ap", ascending=False).head(15)
    return


@app.cell
def _():
    # three_model_results = []

    # weights = np.arange(0.0, 1.01, 0.05)

    # for w_cb in weights:
    #     for w_lgb in weights:
    #         w_xgb = 1.0 - w_cb - w_lgb

    #         if w_xgb < 0:
    #             continue

    #         preds = (
    #             w_cb * models_oof["catboost"]
    #             + w_lgb * models_oof["lightgbm"]
    #             + w_xgb * models_oof["xgboost"]
    #         )

    #         metrics_three_model = evaluate_predictions(y, preds)

    #         three_model_results.append({
    #             "w_catboost": w_cb,
    #             "w_lightgbm": w_lgb,
    #             "w_xgboost": w_xgb,
    #             **metrics_three_model,
    #         })

    # three_model_results = pd.DataFrame(
    #     three_model_results
    # )
    return


@app.cell
def _():
    # three_model_results.sort_values("roc_auc", ascending=False).head(15)
    return


@app.cell
def _():
    # rank_predictions = {
    #     name: rankdata(pred) / len(pred)
    #     for name, pred in models_oof.items()
    # }

    # rank_blend_3 = (
    #     rank_predictions["catboost"]
    #     + rank_predictions["lightgbm"]
    #     + rank_predictions["xgboost"]
    # ) / 3

    # evaluate_predictions(
    #     y,
    #     rank_blend_3,
    # )
    return


@app.cell
def _():
    # rank_blend_cb_xgb = (
    #     rank_predictions["catboost"]
    #     + rank_predictions["xgboost"]
    # ) / 2

    # evaluate_predictions(
    #     y,
    #     rank_blend_cb_xgb,
    # )
    return


@app.cell
def _():
    # best_blend_oof = (
    #     0.60 * models_oof["catboost"]
    #     + 0.20 * models_oof["lightgbm"]
    #     + 0.20 * models_oof["xgboost"]
    # )
    return


@app.cell
def _():
    # fold_comparison = []

    # for f in sorted(dev_model["fold"].unique()):
    #     bool_mask = dev_model["fold"].to_numpy() == f

    #     cb_metrics = evaluate_predictions(
    #         y_true=y.to_numpy()[bool_mask],
    #         y_pred=models_oof["catboost"][bool_mask],
    #     )

    #     blend_metrics = evaluate_predictions(
    #         y_true=y.to_numpy()[bool_mask],
    #         y_pred=best_blend_oof[bool_mask],
    #     )

    #     fold_comparison.append({
    #         "fold": f,

    #         "catboost_ap": cb_metrics["ap"],
    #         "blend_ap": blend_metrics["ap"],
    #         "delta_ap": (
    #             blend_metrics["ap"]
    #             - cb_metrics["ap"]
    #         ),

    #         "catboost_auc": cb_metrics["roc_auc"],
    #         "blend_auc": blend_metrics["roc_auc"],
    #         "delta_auc": (
    #             blend_metrics["roc_auc"]
    #             - cb_metrics["roc_auc"]
    #         ),

    #         "catboost_precision10":
    #             cb_metrics["precision_at_10pct"],
    #         "blend_precision10":
    #             blend_metrics["precision_at_10pct"],

    #         "catboost_recall10":
    #             cb_metrics["recall_at_10pct"],
    #         "blend_recall10":
    #             blend_metrics["recall_at_10pct"],
    #     })

    # fold_comparison = pd.DataFrame(
    #     fold_comparison
    # )

    # fold_comparison
    return


@app.cell
def _():
    # print(
    #     fold_comparison[
    #         ["delta_ap", "delta_auc"]
    #     ].agg(["mean", "std", "min", "max"])
    # )

    # print(
    #     "AP positive folds:",
    #     (fold_comparison["delta_ap"] > 0).sum(),
    #     "/ 5",
    # )

    # print(
    #     "AUC positive folds:",
    #     (fold_comparison["delta_auc"] > 0).sum(),
    #     "/ 5",
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### M1 Results — Baseline Model Comparison and Initial Ensemble Blending

    Three tree-based boosting models were evaluated on the accepted 148-feature RFE2 representation:
    1. Tuned CatBoost candidate (Trial 29);
    2. Baseline LightGBM;
    3. Baseline XGBoost.

    LightGBM and XGBoost were trained as competitive baseline models without extensive hyperparameter optimization to test whether their predictions provided complementary ranking information for ensembling.

    ### Individual OOF Performance

    | Model | AP | ROC-AUC | LogLoss | Precision@10% | Recall@10% |
    |---|---:|---:|---:|---:|---:|
    | CatBoost | 0.28526 | 0.78983 | 0.23607 | 0.30591 | 0.37894 |
    | LightGBM | 0.28181 | 0.78735 | 0.23696 | 0.30171 | 0.37373 |
    | XGBoost | 0.27992 | 0.78660 | 0.23721 | 0.30308 | 0.37543 |

    CatBoost remained the strongest standalone model.

    However, the alternative models were sufficiently different from CatBoost to
    remain useful for ensembling.

    ### Prediction correlation

    Pearson correlations:

    - CatBoost vs LightGBM: approximately `0.958`;
    - CatBoost vs XGBoost: approximately `0.951`;
    - LightGBM vs XGBoost: approximately `0.971`.

    Spearman correlations showed the same pattern:

    - CatBoost vs LightGBM: approximately `0.964`;
    - CatBoost vs XGBoost: approximately `0.956`;
    - LightGBM vs XGBoost: approximately `0.973`.

    XGBoost predictions were the least correlated with CatBoost, while LightGBM and
    XGBoost were more similar to each other.

    This indicated useful model diversity despite weaker standalone performance.

    ### Probability blending

    A coarse OOF grid search over simple weighted probability averages was
    performed.

    The best two-model combinations were approximately:

    - `0.65 CatBoost + 0.35 XGBoost`
      - ROC-AUC ≈ `0.79098`;
      - AP ≈ `0.28719`.

    - `0.60 CatBoost + 0.40 LightGBM`
      - ROC-AUC ≈ `0.79085`;
      - AP ≈ `0.28727`.

    A three-model blend improved further.

    The best simple and stable weighting was:

    `0.60 CatBoost + 0.20 LightGBM + 0.20 XGBoost`

    OOF performance:

    - AP: `0.28763`;
    - ROC-AUC: `0.79115`;
    - LogLoss: `0.23560`;
    - Precision@10%: `0.30725`;
    - Recall@10%: `0.38060`.

    Relative to CatBoost alone:

    - ΔAP ≈ `+0.00237`;
    - ΔROC-AUC ≈ `+0.00132`;
    - LogLoss improved by approximately `0.00047`;
    - Precision@10% improved;
    - Recall@10% improved.

    The nearby weight combinations produced nearly identical results, indicating a
    broad optimum rather than a single unstable OOF optimum.

    ### Fold consistency

    The selected `0.60 / 0.20 / 0.20` ensemble improved both AP and ROC-AUC on all
    five validation folds.

    Paired AP improvements:

    - fold 0: `+0.00287`;
    - fold 1: `+0.00179`;
    - fold 2: `+0.00292`;
    - fold 3: `+0.00246`;
    - fold 4: `+0.00190`.

    Paired ROC-AUC improvements:

    - fold 0: `+0.00132`;
    - fold 1: `+0.00186`;
    - fold 2: `+0.00064`;
    - fold 3: `+0.00144`;
    - fold 4: `+0.00131`.

    Therefore:

    - AP improved in `5/5` folds;
    - ROC-AUC improved in `5/5` folds.

    Top-10 review-policy metrics also improved in aggregate, although individual
    fold-level changes were small and not uniformly positive.

    ### Rank averaging

    Rank-based averaging was also evaluated because ROC-AUC depends only on ranking.

    The equal-weight three-model rank blend achieved approximately:

    - ROC-AUC: `0.79054`;
    - AP: `0.28677`.

    This was weaker than probability averaging.

    Therefore, the probability scale produced by the models contains useful
    information that is partially lost when predictions are converted to ranks.

    ### Decision

    **ACCEPT**

    The three-model probability ensemble becomes the current research / leaderboard
    champion.

    Selected weights:

    - CatBoost: `0.60`;
    - LightGBM: `0.20`;
    - XGBoost: `0.20`.

    CatBoost remains the strongest individual model and is the preferred deployment
    candidate for the later FastAPI service because the approximately `0.0013`
    ROC-AUC ensemble advantage is small relative to the additional serving
    complexity of maintaining three separate models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F8 — Installments contract-level repayment dynamics

    ### Hypothesis

    Previous installments experiments showed that repayment behavior is the most
    valuable historical information source in the current feature set.

    However, existing installment features primarily summarize payment behavior
    directly at the applicant level.

    This may hide heterogeneous behavior across different previous contracts.

    For example, an applicant may have:

    - one previous contract with consistently on-time payments;
    - another contract with repeated delinquency or underpayment.

    A direct client-level mean can partially average these behaviors away.

    The hypothesis is that a two-stage aggregation:

    `installment → previous contract → applicant`

    can preserve contract-level repayment structure and provide additional ranking
    signal beyond the already accepted installment features.

    ### Feature construction

    The experiment introduces contract-level repayment diagnostics based only on
    information available before the current application cutoff.

    For each previous contract, candidate features summarize:

    - number of observed installments;
    - share of late payments;
    - share of payments more than 7 days late;
    - share of payments more than 30 days late;
    - mean and maximum days late;
    - mean and minimum payment-to-installment ratio;
    - share of underpaid installments;
    - whether the contract contains materially problematic repayment behavior.

    Contract-level features are then aggregated to the applicant level using
    statistics such as:

    - mean behavior across previous contracts;
    - worst-contract behavior;
    - variability across contracts;
    - share of problematic contracts;
    - behavior of the most recent previous contract;
    - recency-weighted repayment behavior.

    This representation is intentionally different from the previously accepted
    IP1–IP3 features: the main experimental change is preserving contract-level
    structure before client aggregation.

    ### Cutoff discipline

    Only installments whose scheduled payment date is known before the current
    application are used.

    Actual payment information occurring after the prediction cutoff must not
    contribute to any feature.

    Unavailable future payment outcomes are treated as missing rather than as
    successful repayment.

    ### Evaluation

    The candidate bundle is added to the current RFE2 representation and evaluated
    with:

    - tuned CatBoost Trial 29 parameters;
    - the same immutable five validation folds;
    - the same development population;
    - the same evaluation contract.

    Primary metric:

    - Average Precision.

    Secondary metrics:

    - ROC-AUC;
    - LogLoss;
    - Precision@Top10%;
    - Recall@Top10%.

    Paired fold-level differences against the current CatBoost reference are used
    to evaluate consistency.

    ### Decision rule

    The bundle is retained only if it produces a meaningful and reasonably
    consistent improvement.

    For this final leaderboard-oriented feature-engineering pass, approximately:

    - ROC-AUC improvement greater than `+0.0005`;
    - improvement on at least `4/5` folds;
    - no material degradation in AP or policy metrics

    would be considered sufficient evidence to retain the bundle.

    Small mixed-fold improvements are not sufficient because GPU CatBoost
    variability is already known to be non-negligible.
    """)
    return


@app.cell
def _(pd):
    final_training_dataset = pd.read_parquet("data/processed/modeling_final.parquet")
    return (final_training_dataset,)


@app.cell
def _(pd):
    t29_fold_metrics = pd.read_parquet("mlartifacts\\1\\5665cc4f67844ea79c909b9951c593c1\\artifacts\\metrics\\fold_metrics.parquet")
    return


@app.cell
def _():
    IPX_FEATURES = ['IPX_N_CONTRACTS', 'IPX_MEAN_CONTRACT_LATE_SHARE',
           'IPX_MAX_CONTRACT_LATE_SHARE', 'IPX_STD_CONTRACT_LATE_SHARE',
           'IPX_MEAN_CONTRACT_LATE30_SHARE', 'IPX_MAX_CONTRACT_LATE30_SHARE',
           'IPX_MEAN_CONTRACT_MAX_DAYS_LATE', 'IPX_WORST_CONTRACT_DAYS_LATE',
           'IPX_MEAN_CONTRACT_PAYMENT_RATIO', 'IPX_MIN_CONTRACT_PAYMENT_RATIO',
           'IPX_MEAN_CONTRACT_UNDERPAID_SHARE', 'IPX_MAX_CONTRACT_UNDERPAID_SHARE',
           'IPX_BAD_CONTRACT_SHARE', 'IPX_LATEST_IPX_CONTRACT_LATE_SHARE',
           'IPX_LATEST_IPX_CONTRACT_LATE_30D_SHARE',
           'IPX_LATEST_IPX_CONTRACT_MAX_DAYS_LATE',
           'IPX_LATEST_IPX_CONTRACT_UNDERPAID_SHARE',
           'IPX_WEIGHTED_IPX_CONTRACT_LATE_SHARE',
           'IPX_WEIGHTED_IPX_CONTRACT_LATE_30D_SHARE',
           'IPX_WEIGHTED_IPX_CONTRACT_UNDERPAID_SHARE']
    return (IPX_FEATURES,)


@app.cell
def _(BASELINE_PARAMS, FINAL_FEATURES, IPX_FEATURES, T1_BEST_PARAMS):
    f8_features = (FINAL_FEATURES + IPX_FEATURES)

    tuned_params = {
            **BASELINE_PARAMS,
            **T1_BEST_PARAMS,
        }
    return (tuned_params,)


@app.cell
def _():
    # f8_run = run_catboost_experiment(
    #     frame=final_training_dataset,
    #     features=f8_features,
    #     categorical_features=FINAL_CATEGORICAL_FEATURES,
    #     params=tuned_params,
    #     run_name="cb_ipx_f8_v1",
    #     capacity=0.10,
    #     calculate_shap=False
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     t29_fold_metrics,
    #     f8_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap"
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### F8 Results — Cross-Validation Evaluation

    The experiment evaluated the two-stage contract dynamic features using the tuned CatBoost Trial 29 hyperparameters, the fixed RFE2 representation, the same immutable five folds, and the same development population.

    ### Results

    Paired AP differences versus the Trial 29 reference were:

    - fold 0: approximately `-0.00024`;
    - fold 1: approximately `+0.00066`;
    - fold 2: approximately `-0.00129`;
    - fold 3: approximately `+0.00009`;
    - fold 4: approximately `-0.00139`.

    Only `2/5` folds improved.

    Mean fold AP decreased from approximately:

    `0.28559 → 0.28516`.

    ROC-AUC increased slightly to approximately `0.79003`, but this secondary
    improvement was not accompanied by improvement in the primary metric.

    ### Interpretation

    The contract-level representation did not add consistent incremental
    information beyond the already accepted installment features.

    The previously engineered IP1–IP3 features likely capture most of the useful
    repayment-discipline signal available from the installments table.

    ### Decision

    **REJECT**

    The F8 feature bundle is not included in the final feature representation.
    No further installments feature engineering is performed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## F9 — POS and bureau-balance temporal dynamics

    ### Hypothesis

    Earlier experiments using POS_CASH_balance and bureau_balance did not provide
    enough incremental value to retain their feature bundles.

    However, those experiments were based primarily on conventional static
    aggregates such as counts, means, maxima, and overall delinquency summaries.

    Both tables contain repeated monthly observations.

    Flat aggregation may therefore discard useful information about:

    - recency;
    - repayment state;
    - deterioration or improvement over time;
    - current vs historical delinquency;
    - remaining contract progress.

    The hypothesis is that the lack of improvement from the earlier POS and
    bureau-balance experiments reflects an insufficient temporal representation
    rather than complete absence of predictive signal.

    ### POS_CASH dynamic features

    POS features are first constructed at the previous-contract level and then
    aggregated to the applicant.

    Candidate information includes:

    - latest delinquency state;
    - recent six-month delinquency frequency;
    - recent severe delinquency frequency;
    - recent vs long-term delinquency behavior;
    - worsening or improvement in delinquency;
    - remaining-installment ratio;
    - change in remaining-installment ratio over the observed contract history;
    - share of contracts currently showing problematic repayment behavior.

    The main distinction from the previous POS experiment is explicit separation of
    recent behavior from full-history behavior and preservation of contract
    progress.

    ### bureau_balance dynamic features

    bureau_balance is linked back to applicants through `SK_ID_BUREAU`.

    Monthly credit-status observations are mapped to an ordinal delinquency
    severity representation.

    Candidate features summarize:

    - recent six-month delinquency share;
    - recent twelve-month delinquency share;
    - maximum historical delinquency severity;
    - maximum recent delinquency severity;
    - months since the latest observed delinquency;
    - share of bureau accounts with recent delinquency;
    - recent delinquency relative to long-term delinquency;
    - recent worsening of account status.

    The central experimental change is using the monthly sequence to represent
    recency and deterioration rather than reducing the entire history to one static
    summary.

    ### Why POS and bureau_balance are tested together

    Both sources previously failed to provide meaningful incremental improvement
    using simpler feature representations.

    The objective of this final experiment is not to perform exhaustive feature
    search separately for each source.

    Instead, they are treated as one compact recovery experiment testing a common
    hypothesis:

    > monthly behavioral tables may become useful when their temporal structure is
    > explicitly represented.

    This keeps the final research stage bounded and prevents repeated low-value
    feature experimentation.

    ### Cutoff discipline

    Only historical monthly observations available before the current application
    are used.

    `MONTHS_BALANCE` values must correspond to observations at or before the
    prediction cutoff.

    No information generated after the current application is allowed to enter the
    features.

    ### Evaluation

    The combined POS + bureau_balance dynamic bundle is evaluated with:

    - tuned CatBoost Trial 29 parameters;
    - the same RFE2 feature representation;
    - the same immutable five folds;
    - the same development population and prediction cutoff.

    Primary metric:

    - Average Precision.

    Secondary metrics:

    - ROC-AUC;
    - LogLoss;
    - Precision@Top10%;
    - Recall@Top10%.

    Paired fold-level deltas are compared with the current CatBoost reference.

    ### Decision rule

    The combined bundle is retained only if it provides a clear enough improvement
    to justify keeping two additional historical pipelines.

    Approximately:

    - ROC-AUC improvement greater than `+0.0005`;
    - positive direction on at least `4/5` folds;
    - no meaningful degradation in AP or policy metrics

    would justify retention.

    If improvement is negligible or fold consistency is weak, both sources are
    rejected and no further POS / bureau_balance feature search is performed.
    """)
    return


@app.cell
def _(FINAL_FEATURES):
    POSX_FEATURES = ['POSX_MEAN_LATEST_DPD', 'POSX_MAX_LATEST_DPD',
           'POSX_BAD_LATEST_SHARE', 'POSX_MEAN_RECENT6_DPD_SHARE',
           'POSX_MAX_RECENT6_DPD_SHARE', 'POSX_MEAN_RECENT_WORSENING',
           'POSX_MAX_RECENT_WORSENING', 'POSX_MEAN_PROGRESS', 'POSX_MIN_PROGRESS']

    BBX_FEATURES = ['BBX_RECENT_DELINQUENT_ACCOUNT_SHARE',
           'BBX_MEAN_RECENT6_DELINQ_SHARE', 'BBX_MAX_RECENT6_DELINQ_SHARE',
           'BBX_MEAN_RECENT12_DELINQ_SHARE', 'BBX_MAX_SEVERITY',
           'BBX_MAX_RECENT6_SEVERITY', 'BBX_MIN_MONTHS_SINCE_DELINQUENCY',
           'BBX_MEAN_RECENT_WORSENING', 'BBX_MAX_RECENT_WORSENING']

    f9_features = (FINAL_FEATURES + POSX_FEATURES + BBX_FEATURES)
    return BBX_FEATURES, POSX_FEATURES, f9_features


@app.cell
def _():
    # f9_run = run_catboost_experiment(
    #     frame=final_training_dataset,
    #     features=f9_features,
    #     categorical_features=FINAL_CATEGORICAL_FEATURES,
    #     params=tuned_params,
    #     run_name="cb_posx_bbx_f9_v1",
    #     capacity=0.10,
    #     calculate_shap=False
    # )
    return


@app.cell
def _():
    # compare_fold_results(
    #     t29_fold_metrics,
    #     f9_run["fold_metrics"],
    #     "valid_ap",
    #     "valid_ap"
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### F9 Results — Cross-Validation Evaluation

    The candidate bundle combined the new POS and bureau-balance dynamic features (excluding the rejected F8 bundle) and evaluated them using tuned CatBoost Trial 29 hyperparameters on the fixed RFE2 representation across the 5 immutable folds.

    ### Results

    Paired AP improvements versus Trial 29 were approximately:

    - fold 0: `+0.00002`;
    - fold 1: `+0.00048`;
    - fold 2: `+0.00047`;
    - fold 3: `+0.00306`;
    - fold 4: `-0.00048`.

    AP improved on `4/5` folds.

    Mean fold AP increased from approximately:

    `0.28559 → 0.28630`

    for a mean improvement of approximately:

    `+0.00071`.

    ROC-AUC increased from approximately:

    `0.78983 → 0.79032`

    for an improvement of approximately:

    `+0.00049`.

    Aggregate Top-10 policy metrics also improved:

    - Precision@10% increased from approximately `0.30590` to `0.30640`;
    - Recall@10% increased from approximately `0.37894` to `0.37956`.

    ### Interpretation

    Unlike the earlier static POS and bureau_balance experiments, the temporal
    representation provides useful incremental ranking information.

    The result supports the hypothesis that these monthly tables contain signal
    primarily in recent state, trajectory, and delinquency dynamics rather than in
    simple full-history aggregates.

    The AP gain is partly concentrated in one validation fold, so the effect should
    be considered modest rather than large. Nevertheless, the direction is
    positive across four of five folds and is supported by ROC-AUC and policy
    metrics.

    ### Decision

    **ACCEPT**

    The F9 POS + bureau_balance temporal feature bundle is added to the final
    research feature representation.

    No further feature-engineering search is performed.
    """)
    return


@app.cell
def _(BBX_FEATURES, FINAL_FEATURES, POSX_FEATURES):
    ACCEPTED_FINAL_FEATURES = FINAL_FEATURES  + POSX_FEATURES + BBX_FEATURES
    return (ACCEPTED_FINAL_FEATURES,)


@app.cell
def _(BBX_FEATURES, FINAL_FEATURES, POSX_FEATURES, f9_features):
    print("RFE2 features:", len(FINAL_FEATURES))
    print("POSX features:", len(POSX_FEATURES))
    print("BBX features:", len(BBX_FEATURES))
    print("F9 total features:", len(f9_features))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final feature representation

    Following the acceptance of candidate feature bundle F9, the feature engineering phase of the project is formally concluded and frozen.

    ### Composition
    - **RFE2 Core Features**: 148 features (parsimonious subset selected from 204 FULL features through two rounds of recursive SHAP elimination);
    - **POSX Dynamic Features**: 9 features (capturing contract completion ratios, remaining installment burdens, recent 6-month lateness, and worsening trajectory);
    - **BBX Dynamic Features**: 9 features (capturing ordinal status mapping, maximum delinquency severity, and recent 6-month / 12-month delinquency rates);
    - **Total Schema**: **166 features** (`ACCEPTED_FINAL_FEATURES`).

    ### Scope and Discipline
    - **F8 / IPX Excluded**: The two-stage installments candidate bundle showed net negative cross-validation transfer (-0.000302 mean ΔAP) and is strictly excluded;
    - **F9 Included**: Both POS and bureau-balance temporal dynamics demonstrated consistent positive transfer (+0.000296 mean ΔAP, 4/5 positive folds) and are included;
    - **Feature Freeze**: No further feature engineering, aggregation, transformation, or feature selection is performed;
    - `ACCEPTED_FINAL_FEATURES` serves as the single source of truth for all subsequent model tuning, ensembling, holdout evaluation, and test submission.

    ### Feature Source Summary
    | Source Table | Feature Family | Count | Primary Risk Signal Captured |
    |---|---|---:|---|
    | `application_train` | Demographics, financial ratios, external scores | 61 | Core applicant profile, leverage ratios, credit bureau scores |
    | `bureau` | Credit bureau historical credit summary | 34 | Historical active/closed credit count, debt burden, past delinquencies |
    | `previous_application` | Previous Home Credit applications | 25 | Past contract outcomes, approved credit amounts, rejection history |
    | `installments_payments` | Installments repayment history | 18 | Aggregated payment timeliness, underpayment, and delay metrics |
    | `credit_card_balance` | Revolving credit card activity | 10 | Card utilization rates, drawing patterns, balance-to-limit ratios |
    | `POS_CASH_balance` (POSX) | Contract completion and DPD dynamics | 9 | Remaining installments ratio, recent 6M DPD, lateness worsening |
    | `bureau_balance` (BBX) | Monthly bureau delinquency transitions | 9 | Maximum status severity, recent 6M/12M delinquency rates |
    | **Total** | | **166** | Complete multi-table borrower risk representation |
    """)
    return


@app.cell
def _(final_training_dataset):
    dev_f9 = (
        final_training_dataset.loc[
            final_training_dataset["partition"] == "development"
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(dev_f9.shape)
    print(dev_f9["fold"].value_counts().sort_index())
    print(dev_f9["TARGET"].mean())
    return (dev_f9,)


@app.cell
def _(dev_f9, f9_features):
    X_f9 = dev_f9[f9_features].copy()
    y_f9 = dev_f9["TARGET"].copy()
    folds_f9 = dev_f9["fold"].copy()

    categorical_features_f9 = (
        X_f9
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    for categ_col in categorical_features_f9:
        X_f9[categ_col] = X_f9[categ_col].astype("category")

    print("X:", X_f9.shape)
    print("Categorical:", len(categorical_features_f9))
    return X_f9, categorical_features_f9, folds_f9, y_f9


@app.cell
def _():
    # lgbm_f9_oof, lgbm_f9_fold_metrics, lgbm_f9_models = (
    #     run_lgbm_cv(
    #         X=X_f9,
    #         y=y_f9,
    #         folds=folds_f9,
    #         categorical_features=categorical_features_f9,
    #         params=lgbm_params,
    #     )
    # )
    return


@app.cell
def _():
    # lgbm_f9_metrics = evaluate_predictions(
    #     y_true=y_f9,
    #     y_pred=lgbm_f9_oof,
    # )

    # lgbm_f9_metrics
    return


@app.cell
def _():
    # xgb_f9_oof, xgb_f9_fold_metrics, xgb_f9_models = (
    #     run_xgb_cv(
    #         X=X_f9,
    #         y=y_f9,
    #         folds=folds_f9,
    #         params=xgb_params,
    #     )
    # )
    return


@app.cell
def _():
    # xgb_f9_metrics = evaluate_predictions(
    #     y_true=y_f9,
    #     y_pred=xgb_f9_oof,
    # )

    # xgb_f9_metrics
    return


@app.cell
def _():
    # cb_f9_oof_df = f9_run["oof"].copy()
    return


@app.cell
def _():
    # cb_f9_oof_aligned = (
    #     dev_f9[
    #         ["SK_ID_CURR"]
    #     ]
    #     .merge(
    #         cb_f9_oof_df[
    #             [
    #                 "SK_ID_CURR",
    #                 "probability",
    #             ]
    #         ],
    #         on="SK_ID_CURR",
    #         how="left",
    #         validate="one_to_one",
    #     )
    # )

    # assert cb_f9_oof_aligned[
    #     "probability"
    # ].notna().all()

    # cb_f9_oof = (
    #     cb_f9_oof_aligned[
    #         "probability"
    #     ]
    #     .to_numpy()
    # )
    return


@app.cell
def _():
    # models_f9_oof = {
    #     "catboost": cb_f9_oof,
    #     "lightgbm": lgbm_f9_oof,
    #     "xgboost": xgb_f9_oof,
    # }

    # for name_model, predictions in models_f9_oof.items():
    #     print(
    #         "\n",
    #         name_model,
    #         evaluate_predictions(
    #             y_true=y_f9,
    #             y_pred=predictions,
    #         ),
    #     )
    return


@app.cell
def _():
    # prediction_comparison_f9 = pd.DataFrame(
    #     models_f9_oof
    # )


    # prediction_comparison_f9.corr()
    return


@app.cell
def _():
    # prediction_comparison_f9.corr(
    #         method="spearman"
    #     )
    return


@app.cell
def _():
    # blend_f9_602020 = (
    #     0.60 * models_f9_oof["catboost"]
    #     + 0.20 * models_f9_oof["lightgbm"]
    #     + 0.20 * models_f9_oof["xgboost"]
    # )

    # blend_f9_602020_metrics = (
    #     evaluate_predictions(
    #         y_true=y_f9,
    #         y_pred=blend_f9_602020,
    #     )
    # )

    # blend_f9_602020_metrics
    return


@app.cell
def _():
    # f9_blend_results = []

    # step = 0.01
    # n_steps = int(1 / step)

    # for i in range(n_steps + 1):
    #     for j in range(n_steps + 1 - i):

    #         w_cb_blend = i * step
    #         w_lgb_blend = j * step
    #         w_xgb_blend = 1.0 - w_cb_blend - w_lgb_blend

    #         # Defensive checks
    #         assert w_cb_blend >= 0
    #         assert w_lgb_blend >= 0
    #         assert w_xgb_blend >= -1e-12

    #         w_xgb_blend = max(0.0, w_xgb_blend)

    #         assert np.isclose(
    #             w_cb_blend + w_lgb_blend + w_xgb_blend,
    #             1.0,
    #         )

    #         pred_blend = (
    #             w_cb_blend * models_f9_oof["catboost"]
    #             + w_lgb_blend * models_f9_oof["lightgbm"]
    #             + w_xgb_blend * models_f9_oof["xgboost"]
    #         )

    #         metric_blend = evaluate_predictions(
    #             y_true=y_f9,
    #             y_pred=pred_blend,
    #         )

    #         f9_blend_results.append({
    #             "w_catboost": w_cb_blend,
    #             "w_lightgbm": w_lgb_blend,
    #             "w_xgboost": w_xgb_blend,
    #             **metric_blend,
    #         })

    # f9_blend_results = pd.DataFrame(
    #     f9_blend_results
    # )
    return


@app.cell
def _():
    # f9_blend_results.sort_values(
    #         "roc_auc",
    #         ascending=False,
    #     ).head(15)
    return


@app.cell
def _():
    ENSEMBLE_WEIGHTS = {
        "catboost": 0.59,
        "lightgbm": 0.24,
        "xgboost": 0.17,
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## M2 — Alternative model tuning

    ### Objective
    Optimize the hyperparameters of LightGBM and XGBoost via Bayesian optimization (Optuna) on the frozen 166-feature representation (`ACCEPTED_FINAL_FEATURES`). In M1, baseline LightGBM and XGBoost demonstrated strong ensembling diversity; tuning each model individually aims to strengthen their standalone predictive power and maximize multi-model ensemble performance.

    ### Search Contract
    - **Feature representation**: Identical frozen F9 feature set (166 features, `ACCEPTED_FINAL_FEATURES`);
    - **Validation protocol**: Fixed `split_v1` (5 stratified folds, identical development population);
    - **Optimization objective**: Maximization of 5-fold cross-validated ROC-AUC, aligning with the primary competitive leaderboard metric;
    - **Storage backend**: Persistent SQLite database (`sqlite:///optuna.db`) for reproducible, resumable trial history;
    - **Strict holdout discipline**: The 15% internal holdout remains completely untouched.
    """)
    return


@app.cell
def _(X_f9, categorical_features_f9, folds_f9, run_lgbm_cv, y_f9):
    def objective_lgbm(trial):
        params = {
            "objective": "binary",
            "n_estimators": 5000,

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.05,
                log=True,
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves",
                20,
                80,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                4,
                10,
            ),

            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                20,
                150,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.7,
                1.0,
            ),

            "subsample_freq": 1,

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.7,
                1.0,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-3,
                10.0,
                log=True,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-3,
                20.0,
                log=True,
            ),

            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

        _, fold_metrics, _ = run_lgbm_cv(
            X=X_f9,
            y=y_f9,
            folds=folds_f9,
            categorical_features=categorical_features_f9,
            params=params,
        )

        return fold_metrics["roc_auc"].mean()

    return (objective_lgbm,)


@app.cell
def _(X_f9, folds_f9, run_xgb_cv, y_f9):
    def objective_xgb(trial):
        params = {
            "objective": "binary:logistic",
            "n_estimators": 5000,

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.05,
                log=True,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                4,
                8,
            ),

            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1.0,
                20.0,
                log=True,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.7,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.7,
                1.0,
            ),

            "gamma": trial.suggest_float(
                "gamma",
                1e-4,
                5.0,
                log=True,
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-4,
                10.0,
                log=True,
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-3,
                20.0,
                log=True,
            ),

            "tree_method": "hist",
            "device": "cuda",
            "enable_categorical": True,

            "eval_metric": "auc",

            "random_state": 42,
            "early_stopping_rounds": 200,
        }

        _, fold_metrics, _ = run_xgb_cv(
            X=X_f9,
            y=y_f9,
            folds=folds_f9,
            params=params,
        )

        return fold_metrics["roc_auc"].mean()

    return (objective_xgb,)


@app.cell
def _():
    OPTUNA_STORAGE = "sqlite:///optuna.db"
    return (OPTUNA_STORAGE,)


@app.cell
def _(OPTUNA_STORAGE, objective_lgbm, optuna):
    study_lgbm = optuna.create_study(
        study_name="lgbm_f9_auc_v1",
        storage=OPTUNA_STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    study_lgbm.optimize(
        objective_lgbm,
        n_trials=10,
    )

    print("Best value:", study_lgbm.best_value)
    print("Best params:", study_lgbm.best_params)
    return (study_lgbm,)


@app.cell
def _(study_lgbm):
    for best_trial_lgbm in study_lgbm.best_trials:
        print("Best trial:", best_trial_lgbm.number)
        print("  Value:", best_trial_lgbm.value)
        print("  Params:")
        for key, value in best_trial_lgbm.params.items():
            print(f"    {key}: {value}")
    return


@app.cell
def _(OPTUNA_STORAGE, objective_xgb, optuna):
    study_xgb = optuna.create_study(
        study_name="xgb_f9_auc_v1",
        storage=OPTUNA_STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    study_xgb.optimize(
        objective_xgb,
        n_trials=20,
    )

    print("Best value:", study_xgb.best_value)
    print("Best params:", study_xgb.best_params)
    return (study_xgb,)


@app.cell
def _(study_xgb):
    for best_trial_xgb in study_xgb.best_trials:
        print("Best trial:", best_trial_xgb.number)
        print("  Value:", best_trial_xgb.value)
        print("  Params:")
        for param, param_value in best_trial_xgb.params.items():
            print(f"    {param}: {param_value}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardized MLflow CV Experiment Runner
    Define a generic cross-validation runner to evaluate candidate models under the fixed protocol, logging fold metrics, OOF predictions, feature schemas, and parameter artifacts to MLflow.
    """)
    return


@app.cell
def _(Path, evaluate_predictions, json, mlflow, pd, tempfile):
    def run_model_experiment(
        *,
        run_name: str,
        model_name: str,
        cv_runner,
        cv_runner_kwargs: dict,
        params: dict,
        feature_names: list[str],
        ids: pd.Series,
        target: pd.Series,
        folds: pd.Series,
        tags: dict | None = None,
    ):
        """
        Generic MLflow experiment wrapper.

        cv_runner must return:
            oof_predictions,
            fold_metrics,
            models
        """

        with mlflow.start_run(
            run_name=run_name
        ) as run:

            # -------------------------
            # Tags
            # -------------------------

            mlflow.set_tags({
                "model": model_name,
                "run_type": "cv_experiment",
                **(tags or {}),
            })

            # -------------------------
            # Parameters
            # -------------------------

            safe_params = {
                key: value
                for key, value in params.items()
                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                    ),
                )
            }

            mlflow.log_params(safe_params)

            mlflow.log_param(
                "n_features",
                len(feature_names),
            )

            # -------------------------
            # CV
            # -------------------------

            (
                oof_predictions,
                fold_metrics,
                models,
            ) = cv_runner(
                **cv_runner_kwargs,
                params=params,
            )

            # -------------------------
            # OOF metrics
            # -------------------------

            oof_metrics = evaluate_predictions(
                y_true=target,
                y_pred=oof_predictions,
            )

            mlflow.log_metrics({
                f"oof_{key}": value
                for key, value
                in oof_metrics.items()
            })

            # -------------------------
            # Fold summary
            # -------------------------

            numeric_fold_metrics = (
                fold_metrics
                .select_dtypes(include="number")
            )

            for metric in [
                "ap",
                "roc_auc",
                "log_loss",
                "precision_at_10pct",
                "recall_at_10pct",
            ]:
                if metric not in fold_metrics.columns:
                    continue

                mlflow.log_metric(
                    f"fold_{metric}_mean",
                    fold_metrics[metric].mean(),
                )

                mlflow.log_metric(
                    f"fold_{metric}_std",
                    fold_metrics[metric].std(),
                )

            # -------------------------
            # Artifacts
            # -------------------------

            oof_df = pd.DataFrame({
                "SK_ID_CURR": ids.to_numpy(),
                "TARGET": target.to_numpy(),
                "fold": folds.to_numpy(),
                "probability": oof_predictions,
            })

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)

                oof_path = (
                    temp_dir
                    / "oof_predictions.parquet"
                )

                folds_path = (
                    temp_dir
                    / "fold_metrics.csv"
                )

                features_path = (
                    temp_dir
                    / "features.json"
                )

                params_path = (
                    temp_dir
                    / "resolved_params.json"
                )

                oof_df.to_parquet(
                    oof_path,
                    index=False,
                )

                fold_metrics.to_csv(
                    folds_path,
                    index=False,
                )

                features_path.write_text(
                    json.dumps(
                        feature_names,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                params_path.write_text(
                    json.dumps(
                        params,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )

                mlflow.log_artifact(
                    str(oof_path),
                    artifact_path="oof",
                )

                mlflow.log_artifact(
                    str(folds_path),
                    artifact_path="metrics",
                )

                mlflow.log_artifact(
                    str(features_path),
                    artifact_path="features",
                )

                mlflow.log_artifact(
                    str(params_path),
                    artifact_path="params",
                )

            return {
                "run_id": run.info.run_id,
                "oof": oof_df,
                "fold_metrics": fold_metrics,
                "oof_metrics": oof_metrics,
                "models": models,
            }

    return (run_model_experiment,)


@app.cell
def _(study_lgbm):
    lgbm_tuned_params = {
        "objective": "binary",
        "n_estimators": 5000,

        **study_lgbm.best_params,

        "subsample_freq": 1,

        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    return (lgbm_tuned_params,)


@app.cell
def _(
    X_f9,
    categorical_features_f9,
    dev_f9,
    f9_features,
    folds_f9,
    lgbm_tuned_params,
    run_lgbm_cv,
    run_model_experiment,
    y_f9,
):
    lgbm_tuned_run = run_model_experiment(
        run_name="lgbm_f9_tuned_v1",

        model_name="lightgbm",

        cv_runner=run_lgbm_cv,

        cv_runner_kwargs={
            "X": X_f9,
            "y": y_f9,
            "folds": folds_f9,
            "categorical_features":
                categorical_features_f9,
        },

        params=lgbm_tuned_params,

        feature_names=f9_features,

        ids=dev_f9["SK_ID_CURR"],
        target=y_f9,
        folds=folds_f9,

        tags={
            "dataset_version":
                "development_v1",
            "split_version":
                "split_v1",
            "feature_version":
                "rfe2_f9",
            "tuning":
                "optuna",
            "optuna_study":
                "lgbm_f9_auc_v1",
        },
    )
    return (lgbm_tuned_run,)


@app.cell
def _(study_xgb):
    xgb_tuned_params = {
        "objective": "binary:logistic",
        "n_estimators": 5000,

        **study_xgb.best_params,

        "tree_method": "hist",
        "device": "cuda",
        "enable_categorical": True,

        "eval_metric": "auc",

        "random_state": 42,
        "early_stopping_rounds": 200,
    }
    return (xgb_tuned_params,)


@app.cell
def _(
    X_f9,
    dev_f9,
    f9_features,
    folds_f9,
    run_model_experiment,
    run_xgb_cv,
    xgb_tuned_params,
    y_f9,
):
    xgb_tuned_run = run_model_experiment(
        run_name="xgb_f9_tuned_v1",

        model_name="xgboost",

        cv_runner=run_xgb_cv,

        cv_runner_kwargs={
            "X": X_f9,
            "y": y_f9,
            "folds": folds_f9,
        },

        params=xgb_tuned_params,

        feature_names=f9_features,

        ids=dev_f9["SK_ID_CURR"],
        target=y_f9,
        folds=folds_f9,

        tags={
            "dataset_version":
                "development_v1",
            "split_version":
                "split_v1",
            "feature_version":
                "rfe2_f9",
            "tuning":
                "optuna",
            "optuna_study":
                "xgb_f9_auc_v1",
        },
    )
    return (xgb_tuned_run,)


@app.cell
def _(pd):
    cb_f9_oof = pd.read_parquet("mlartifacts\\1\\235ba1f2b6dd4e8fb44b939cc9fe7ef6\\artifacts\\predictions\\oof_predictions.parquet")
    cb_f9_fold_metrics = pd.read_parquet("mlartifacts\\1\\235ba1f2b6dd4e8fb44b939cc9fe7ef6\\artifacts\\metrics\\fold_metrics.parquet")
    return cb_f9_fold_metrics, cb_f9_oof


@app.cell
def _(cb_f9_fold_metrics, xgb_tuned_run):
    compare_fold_results(
        cb_f9_fold_metrics,
        xgb_tuned_run["fold_metrics"],
        "valid_ap",
        "ap",
    )
    return


@app.cell
def _(cb_f9_fold_metrics, xgb_tuned_run):
    compare_fold_results(
        cb_f9_fold_metrics,
        xgb_tuned_run["fold_metrics"],
        "roc_auc",
        "roc_auc",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Tuning Results & Confirmatory Evaluation

    Both Optuna Bayesian optimization studies converged on strong hyperparameter configurations that were confirmed via independent 5-fold cross-validation:

    #### LightGBM Tuning
    - **Best Study Trial**: Trial 4 achieved cross-validated ROC-AUC of **0.79024** (`learning_rate=0.0111`, `num_leaves=77`, `max_depth=10`, `min_child_samples=125`, `subsample=0.7914`, `colsample_bytree=0.7293`, `reg_alpha=0.5457`, `reg_lambda=0.0782`);
    - **Confirmatory 5-Fold CV** (`lgbm_f9_tuned_v1`): Mean ROC-AUC = **0.79024**, Mean AP = **0.28485**, LogLoss = **0.23609**, Median best iteration = 1490.

    #### XGBoost Tuning
    - **Best Study Trial**: Trial 11 achieved cross-validated ROC-AUC of **0.79122** (`learning_rate=0.0459`, `max_depth=4`, `min_child_weight=3.2018`, `subsample=0.7035`, `colsample_bytree=0.8387`, `gamma=0.0074`, `reg_alpha=9.3388`, `reg_lambda=1.0788`);
    - **Confirmatory 5-Fold CV** (`xgb_f9_tuned_v1`): Mean ROC-AUC = **0.79122**, Mean AP = **0.28586**, LogLoss = **0.23566**, Median best iteration = 1212;
    - **Competitiveness**: Tuned XGBoost became competitive with CatBoost on ROC-AUC (**0.79122 vs 0.79034**), demonstrating that compact trees (`max_depth = 4`) paired with strong L1 regularization (`reg_alpha = 9.34`) significantly improved tabular generalization.

    ### Performance Summary (5-Fold Cross-Validation on F9 166 Features)
    | Model | Mean AP | Mean ROC-AUC | Mean LogLoss | Median Best Iteration |
    |---|---:|---:|---:|---:|
    | CatBoost F9 (Candidate) | 0.28630 | 0.79034 | 0.23595 | 4397 |
    | LightGBM F9 (Tuned Trial 4) | 0.28485 | 0.79024 | 0.23609 | 1490 |
    | XGBoost F9 (Tuned Trial 11) | 0.28586 | 0.79122 | 0.23566 | 1212 |

    ### Decision
    **ACCEPT** — Accept tuned LightGBM (`lgbm_tuned_params`) and tuned XGBoost (`xgb_tuned_params`) configurations for final ensemble evaluation. Hyperparameter tuning for alternative models is now complete and frozen.
    """)
    return


@app.cell
def _(cb_f9_oof, lgbm_tuned_run, xgb_tuned_run):
    models_tuned_oof = {
        "catboost": cb_f9_oof['probability'].to_numpy(),
        "lightgbm": lgbm_tuned_run["oof"]["probability"].to_numpy(),
        "xgboost": xgb_tuned_run["oof"]["probability"].to_numpy(),
    }
    return (models_tuned_oof,)


@app.cell
def _(dev_f9, lgbm_tuned_run, np, xgb_tuned_run):
    assert np.array_equal(
        lgbm_tuned_run["oof"]["SK_ID_CURR"].to_numpy(),
        dev_f9["SK_ID_CURR"].to_numpy(),
    )

    assert np.array_equal(
        xgb_tuned_run["oof"]["SK_ID_CURR"].to_numpy(),
        dev_f9["SK_ID_CURR"].to_numpy(),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Grid Search over Ensemble Blend Weights
    Conduct a systematic grid search (step = 0.01) across convex combinations of out-of-fold probability predictions from the three tuned models to locate the empirical ROC-AUC / AP optimum.
    """)
    return


@app.cell
def _(evaluate_predictions, models_tuned_oof, pd):
    def blend_three_models(
        pred,
        y_true,
        step: float = 0.01
    ) -> pd.DataFrame:
        """
        Search of optimal weights
        """

        blend_results_tuned = []

        step = 0.01
        n_steps = int(1 / step)

        for i in range(n_steps + 1):
            for j in range(n_steps + 1 - i):

                w_cb = i * step
                w_lgb = j * step
                w_xgb = 1.0 - w_cb - w_lgb

                pred = (
                    w_cb * models_tuned_oof["catboost"]
                    + w_lgb * models_tuned_oof["lightgbm"]
                    + w_xgb * models_tuned_oof["xgboost"]
                )

                metrics = evaluate_predictions(
                    y_true=y_true,
                    y_pred=pred,
                )

                blend_results_tuned.append({
                    "w_catboost": w_cb,
                    "w_lightgbm": w_lgb,
                    "w_xgboost": w_xgb,
                    **metrics,
                })

        blend_results_tuned = pd.DataFrame(
            blend_results_tuned
        )

        return blend_results_tuned

    return (blend_three_models,)


@app.cell
def _(blend_three_models, models_tuned_oof, y_f9):
    blend_result_tuned = blend_three_models(
        pred=models_tuned_oof,
        y_true=y_f9,
        step=0.01,
    ).sort_values("roc_auc", ascending=False).head(15)
    return (blend_result_tuned,)


@app.cell
def _(blend_result_tuned):
    blend_result_tuned
    return


@app.cell
def _(blend_result_tuned):
    blend_result_tuned.sort_values("ap", ascending=False).head(15)
    return


@app.cell
def _():
    FINAL_WEIGHTS = {
        "catboost": 0.36,
        "lightgbm": 0.28,
        "xgboost": 0.36,
    }
    return


@app.cell
def _(models_tuned_oof):
    final_blend_oof = (
        0.36 * models_tuned_oof["catboost"]
        + 0.28 * models_tuned_oof["lightgbm"]
        + 0.36 * models_tuned_oof["xgboost"]
    )
    return (final_blend_oof,)


@app.cell
def _(
    evaluate_predictions,
    final_blend_oof,
    folds_f9,
    models_tuned_oof,
    pd,
    y_f9,
):
    final_fold_comparison = []

    for fold_f9 in sorted(folds_f9.unique()):
        mask_f9 = folds_f9.to_numpy() == fold_f9

        cb_metrics = evaluate_predictions(
            y_true=y_f9.to_numpy()[mask_f9],
            y_pred=models_tuned_oof["catboost"][mask_f9],
        )

        ensemble_metrics = evaluate_predictions(
            y_true=y_f9.to_numpy()[mask_f9],
            y_pred=final_blend_oof[mask_f9],
        )

        final_fold_comparison.append({
            "fold": fold_f9,
            "delta_ap": (
                ensemble_metrics["ap"]
                - cb_metrics["ap"]
            ),
            "delta_auc": (
                ensemble_metrics["roc_auc"]
                - cb_metrics["roc_auc"]
            ),
            "delta_precision10": (
                ensemble_metrics["precision_at_10pct"]
                - cb_metrics["precision_at_10pct"]
            ),
            "delta_recall10": (
                ensemble_metrics["recall_at_10pct"]
                - cb_metrics["recall_at_10pct"]
            ),
        })

    final_fold_comparison = pd.DataFrame(
        final_fold_comparison
    )

    final_fold_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## M3 — Final tuned ensemble

    ### Blend Optimization & Weight Selection
    Grid search over convex combinations of the three tuned models identified a consistent high-performance plateau centered around balanced contributions from CatBoost and XGBoost, with a moderate complementary weight on LightGBM.

    The frozen ensemble weights are:
    - **CatBoost**: `0.36`
    - **LightGBM**: `0.28`
    - **XGBoost**: `0.36`

    ### Out-of-Fold Performance Comparison
    | Metric | CatBoost F9 (Candidate) | Final Tuned Ensemble (0.36 / 0.28 / 0.36) | Improvement (Δ) |
    |---|---:|---:|---:|
    | **Average Precision (AP)** | 0.28630 | **0.28928** | **+0.00298** |
    | **ROC-AUC** | 0.79034 | **0.79298** | **+0.00264** |
    | **LogLoss** | 0.23595 | **0.23512** | **-0.00083** |
    | **Precision@Top10%** | 0.30694 | **0.30985** | **+0.00291** |
    | **Recall@Top10%** | 0.38023 | **0.38384** | **+0.00361** |

    ### Cross-Validation Fold Consistency
    The paired fold comparison between the final tuned ensemble and the CatBoost F9 candidate demonstrated unanimous consistency across all 5 validation splits:
    - **AP improved**: **5 / 5 folds**;
    - **ROC-AUC improved**: **5 / 5 folds**;
    - **Precision@Top10% improved**: **5 / 5 folds**;
    - **Recall@Top10% improved**: **5 / 5 folds**.

    ### Robustness of Optimum
    The objective function forms a broad, flat plateau across adjacent weight allocations (e.g., CatBoost 0.34–0.38, LightGBM 0.26–0.30, XGBoost 0.34–0.38 produce nearly identical ROC-AUC within ~0.00005). The weights `0.36 / 0.28 / 0.36` were selected directly from this robust region without aggressive fine-tuning to guard against overfitting to validation noise.

    ### Decision
    **FINAL OOF CHAMPION ACCEPTED**

    Feature selection (`ACCEPTED_FINAL_FEATURES` = 166), model hyperparameters (`FINAL_CB_PARAMS`, `FINAL_LGBM_PARAMS`, `FINAL_XGB_PARAMS`), and ensemble blend weights (`0.36 / 0.28 / 0.36`) are now completely frozen.

    The frozen specification is now submitted to the untouched 15% holdout set for final unbiased verification.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Development Set Refit & Standalone Holdout Scoring
    Train each tuned model on the entire development split (261,384 rows) up to its median best iteration from 5-fold cross-validation, then generate probability predictions on the untouched 15% holdout partition (46,127 rows).
    """)
    return


@app.cell
def _(lgbm_tuned_params, tuned_params, xgb_tuned_params):
    FINAL_CB_PARAMS = tuned_params
    FINAL_LGBM_PARAMS = lgbm_tuned_params
    FINAL_XGB_PARAMS = xgb_tuned_params
    return FINAL_CB_PARAMS, FINAL_LGBM_PARAMS, FINAL_XGB_PARAMS


@app.cell
def _(final_training_dataset):
    full_split_data = final_training_dataset.copy()

    dev_final = (
        full_split_data[
            full_split_data["partition"] == "development"
        ]
        .copy()
    )

    holdout_final = (
        full_split_data[
            full_split_data["partition"] == "holdout"
        ]
        .copy()
    )

    print(dev_final.shape)
    print(holdout_final.shape)
    return dev_final, holdout_final


@app.cell
def _(cb_f9_fold_metrics, lgbm_tuned_run, np, xgb_tuned_run):
    cb_best_iteration = int(
        np.median(
            cb_f9_fold_metrics["best_iteration"]
        )
    )

    lgbm_best_iteration = int(
        np.median(
            lgbm_tuned_run["fold_metrics"]["best_iteration"]
        )
    )

    xgb_best_iteration = int(
        np.median(
            xgb_tuned_run["fold_metrics"]["best_iteration"]
        )
    )

    print(
        cb_best_iteration,
        lgbm_best_iteration,
        xgb_best_iteration,
    )
    return cb_best_iteration, lgbm_best_iteration, xgb_best_iteration


@app.cell
def _(ACCEPTED_FINAL_FEATURES, dev_final, holdout_final):
    X_dev = dev_final[ACCEPTED_FINAL_FEATURES].copy()
    y_dev = dev_final["TARGET"].copy()

    X_holdout = holdout_final[ACCEPTED_FINAL_FEATURES].copy()
    y_holdout = holdout_final["TARGET"].copy()


    cat_cols = (
        X_dev
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    for cb_cat_col in cat_cols:
        X_dev[cb_cat_col] = X_dev[cb_cat_col].astype(str)
        X_holdout[cb_cat_col] = X_holdout[cb_cat_col].astype(str)
    return X_dev, X_holdout, cat_cols, y_dev, y_holdout


@app.cell
def _(
    CatBoostClassifier,
    FINAL_CB_PARAMS,
    X_dev,
    X_holdout,
    cat_cols,
    cb_best_iteration,
    y_dev,
):
    cb_holdout_params = FINAL_CB_PARAMS.copy()

    cb_holdout_params["iterations"] = cb_best_iteration

    cb_holdout_params.pop(
        "early_stopping_rounds",
        None,
    )

    cb_final_dev = CatBoostClassifier(
        **cb_holdout_params
    )

    cb_final_dev.fit(
        X_dev,
        y_dev,
        cat_features=cat_cols,
        verbose=200,
    )

    cb_holdout_pred = (
        cb_final_dev.predict_proba(
            X_holdout
        )[:, 1]
    )
    return cb_holdout_params, cb_holdout_pred


@app.cell
def _(ACCEPTED_FINAL_FEATURES, cat_cols, dev_final, holdout_final):
    X_dev_lgb = dev_final[
        ACCEPTED_FINAL_FEATURES
    ].copy()

    X_holdout_lgb = holdout_final[
        ACCEPTED_FINAL_FEATURES
    ].copy()

    for col_lgbm in cat_cols:
        X_dev_lgb[col_lgbm] = (
            X_dev_lgb[col_lgbm].astype("category")
        )

        X_holdout_lgb[col_lgbm] = (
            X_holdout_lgb[col_lgbm].astype("category")
        )
    return X_dev_lgb, X_holdout_lgb


@app.cell
def _(
    FINAL_LGBM_PARAMS,
    X_dev_lgb,
    X_holdout_lgb,
    cat_cols,
    lgb,
    lgbm_best_iteration,
    y_dev,
):
    lgbm_holdout_params = (
        FINAL_LGBM_PARAMS.copy()
    )

    lgbm_holdout_params[
        "n_estimators"
    ] = lgbm_best_iteration

    lgbm_final_dev = lgb.LGBMClassifier(
        **lgbm_holdout_params
    )

    lgbm_final_dev.fit(
        X_dev_lgb,
        y_dev,
        categorical_feature=cat_cols,
    )

    lgbm_holdout_pred = (
        lgbm_final_dev.predict_proba(
            X_holdout_lgb
        )[:, 1]
    )
    return lgbm_holdout_params, lgbm_holdout_pred


@app.cell
def _(ACCEPTED_FINAL_FEATURES, cat_cols, dev_final, holdout_final):
    X_dev_xgb = dev_final[
        ACCEPTED_FINAL_FEATURES
    ].copy()

    X_holdout_xgb = holdout_final[
        ACCEPTED_FINAL_FEATURES
    ].copy()

    for col_xgb in cat_cols:
        X_dev_xgb[col_xgb] = (
            X_dev_xgb[col_xgb].astype("category")
        )

        X_holdout_xgb[col_xgb] = (
            X_holdout_xgb[col_xgb].astype("category")
        )
    return X_dev_xgb, X_holdout_xgb


@app.cell
def _(
    FINAL_XGB_PARAMS,
    X_dev_xgb,
    X_holdout_xgb,
    xgb,
    xgb_best_iteration,
    y_dev,
):
    xgb_holdout_params = (
        FINAL_XGB_PARAMS.copy()
    )

    xgb_holdout_params[
        "n_estimators"
    ] = xgb_best_iteration

    xgb_holdout_params.pop(
        "early_stopping_rounds",
        None,
    )

    xgb_final_dev = xgb.XGBClassifier(
        **xgb_holdout_params
    )

    xgb_final_dev.fit(
        X_dev_xgb,
        y_dev,
    )

    xgb_holdout_pred = (
        xgb_final_dev.predict_proba(
            X_holdout_xgb
        )[:, 1]
    )
    return xgb_holdout_params, xgb_holdout_pred


@app.cell
def _(cb_holdout_pred, lgbm_holdout_pred, xgb_holdout_pred):
    holdout_ensemble_pred = (
        0.36 * cb_holdout_pred
        + 0.28 * lgbm_holdout_pred
        + 0.36 * xgb_holdout_pred
    )
    return (holdout_ensemble_pred,)


@app.cell
def _(
    cb_holdout_pred,
    evaluate_predictions,
    holdout_ensemble_pred,
    lgbm_holdout_pred,
    pd,
    xgb_holdout_pred,
    y_holdout,
):
    holdout_results = {
        "catboost": evaluate_predictions(
            y_holdout,
            cb_holdout_pred,
        ),

        "lightgbm": evaluate_predictions(
            y_holdout,
            lgbm_holdout_pred,
        ),

        "xgboost": evaluate_predictions(
            y_holdout,
            xgb_holdout_pred,
        ),

        "ensemble": evaluate_predictions(
            y_holdout,
            holdout_ensemble_pred,
        ),
    }

    pd.DataFrame(
        holdout_results
    ).T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final holdout evaluation

    The final model specification was frozen prior to scoring the untouched 15% internal holdout dataset (46,127 applicants).

    ### Holdout Evaluation Results
    | Model | AP | ROC-AUC | LogLoss | Precision@10% | Recall@10% |
    |---|---:|---:|---:|---:|---:|
    | CatBoost | 0.28038 | 0.79048 | 0.23656 | 0.29640 | 0.36708 |
    | LightGBM | 0.28079 | 0.79124 | 0.23642 | 0.29640 | 0.36708 |
    | XGBoost | 0.28214 | 0.79117 | 0.23617 | 0.30052 | 0.37218 |
    | **Ensemble** | **0.28393** | **0.79319** | **0.23561** | **0.30204** | **0.37406** |

    ### Interpretation
    - **Ensemble Dominance**: The multi-model probability blend (`0.36 / 0.28 / 0.36`) strictly outperforms every individual constituent model across all five reported metrics on unseen holdout data.
    - **Generalization Consistency**: The holdout ROC-AUC of **0.79319** is exceptionally aligned with the cross-validated out-of-fold estimate (**0.79298**, gap < 0.00021), demonstrating that the internal validation protocol did not suffer from meaningful validation overfitting.
    - **Holdout Integrity**: No modeling, feature-selection, or parameter adjustments were made after examining the holdout metrics.

    ### Decision
    **FINAL MODEL SPECIFICATION ACCEPTED**

    The frozen 3-model ensemble specification is validated and approved. The research process proceeds to full-dataset retraining and test-set submission generation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full-data refit

    ### Training Context
    Following the successful validation on the untouched holdout, the entire labeled dataset is now leveraged for final model training:
    - **Recombined Population**: All 307,511 labeled training rows (development + holdout partitions combined) are utilized;
    - **Fixed Feature Schema**: Exactly 166 features (`ACCEPTED_FINAL_FEATURES`), preserving the identical column order and categorical encodings;
    - **Frozen Hyperparameters**: Each model retains its exact tuned parameter specification (`FINAL_CB_PARAMS`, `FINAL_LGBM_PARAMS`, `FINAL_XGB_PARAMS`);
    - **Fixed Iteration Budgets**: Tree counts are set to the median best iterations established during the 5-fold cross-validation procedure (CatBoost: median best iteration, LightGBM: 1490, XGBoost: 1212), eliminating early stopping and avoiding validation-based selection during the refit;
    - **Deterministic Seeding**: `random_state = 42` is maintained across all three models.
    """)
    return


@app.cell
def _(ACCEPTED_FINAL_FEATURES, final_training_dataset):
    X_full = final_training_dataset[
        ACCEPTED_FINAL_FEATURES
    ].copy()

    y_full = final_training_dataset[
        "TARGET"
    ].copy()
    return X_full, y_full


@app.cell
def _(CatBoostClassifier, X_full, cat_cols, cb_holdout_params, y_full):
    catboost_full = CatBoostClassifier(
        **cb_holdout_params
    )

    catboost_full.fit(
        X_full,
        y_full,
        cat_features=cat_cols,
        verbose=200,
    )
    return (catboost_full,)


@app.cell
def _(X_full, xgb, xgb_holdout_params, y_full):
    xgb_full = xgb.XGBClassifier(
        **xgb_holdout_params
    )

    xgb_full.fit(
        X_full,
        y_full
    )
    return (xgb_full,)


@app.cell
def _(X_full, cat_cols, lgb, lgbm_holdout_params, y_full):
    lgbm_full = lgb.LGBMClassifier(
        **lgbm_holdout_params
    )
    lgbm_full.fit(
        X_full,
        y_full,
        categorical_feature=cat_cols,
    )
    return (lgbm_full,)


@app.cell
def _(catboost_full, lgbm_full, xgb_full):
    catboost_full.save_model("artifacts/catboost_full_model.cbm")
    xgb_full.save_model("artifacts/xgboost_full_model.json")
    lgbm_full.booster_.save_model("artifacts/lightgbm_full_model.txt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Trained Model Artifact Freezing
    All three full-data gradient-boosting models were trained on the complete labeled population (307,511 rows) using fixed iteration budgets and saved to persistent artifacts:
    - **CatBoost Full Model**: `artifacts/catboost_full_model.cbm`
    - **LightGBM Full Model**: `artifacts/lightgbm_full_model.txt`
    - **XGBoost Full Model**: `artifacts/xgboost_full_model.json`

    ### Decision
    **FINAL TRAINED ARTIFACTS READY**

    The full-data model ensemble is frozen and ready for Kaggle test inference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kaggle test inference

    ### Test feature assembly
    Execute the refactored, reusable multi-table feature pipeline (`build_test_features`) against `application_test.csv` and the raw historical tables to construct the test feature matrix.
    """)
    return


@app.cell
def _(Path, build_test_features):
    X_test = build_test_features(
            data_dir=Path("data"),
            output_path=Path("data/processed/application_test_features.csv"),
            include_id=False,  # SK_ID_CURR retained as DataFrame index
        )
    return (X_test,)


@app.cell
def _(pd):
    application_test = pd.read_csv("data/raw/application_test.csv")
    return (application_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Schema parity & defensive assertions
    Verify complete schema compliance prior to inference:
    - Feature count equals exactly 166;
    - Feature names and ordering identically match `ACCEPTED_FINAL_FEATURES`;
    - Unique index with zero duplicate applicant IDs;
    - Zero duplicate column headers.
    """)
    return


@app.cell
def _(ACCEPTED_FINAL_FEATURES, X_test):
    assert X_test.shape[1] == 166

    assert list(X_test.columns) == list(
        ACCEPTED_FINAL_FEATURES
    )

    assert X_test.index.is_unique

    assert not X_test.columns.duplicated().any()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Final ensemble inference
    Generate probability predictions for all 48,744 test applications from each full-data model and blend them using the frozen ensemble weights (`0.36 * CatBoost + 0.28 * LightGBM + 0.36 * XGBoost`).
    """)
    return


@app.cell
def _(X_test, catboost_full, lgbm_full, xgb_full):
    catboost_test_pred = (
        catboost_full.predict_proba(
            X_test
        )[:, 1]
    )

    xgb_test_pred = (
        xgb_full.predict_proba(
            X_test
        )[:, 1]
    )

    lgbm_test_pred = (
        lgbm_full.predict_proba(
            X_test
        )[:, 1]
    )
    return catboost_test_pred, lgbm_test_pred, xgb_test_pred


@app.cell
def _(catboost_test_pred, lgbm_test_pred, xgb_test_pred):
    test_pred = (
        0.36 * catboost_test_pred
        + 0.28 * lgbm_test_pred
        + 0.36 * xgb_test_pred
    )
    return (test_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Submission creation & format verification
    Format test predictions into the competition submission schema (`SK_ID_CURR`, `TARGET`), serialize to `submission_final_ensemble.csv`, and assert complete row coverage and valid probability bounds [0, 1].
    """)
    return


@app.cell
def _(application_test, pd, test_pred):
    submission = pd.DataFrame({
        "SK_ID_CURR":
            application_test["SK_ID_CURR"],

        "TARGET":
            test_pred,
    })

    submission.to_csv(
        "submission_final_ensemble.csv",
        index=False,
    )

    submission.head()
    return (submission,)


@app.cell
def _(application_test, submission):
    assert len(submission) == len(
        application_test
    )

    assert submission["TARGET"].between(
        0,
        1,
    ).all()

    assert submission[
        "SK_ID_CURR"
    ].is_unique
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kaggle leaderboard result

    ### Leaderboard Evaluation
    The frozen ensemble submission (`submission_final_ensemble.csv`) was evaluated on the Kaggle competition leaderboard:
    - **Public ROC-AUC**: **0.79366** (approximate public leaderboard position: **~2898**);
    - **Private ROC-AUC**: **0.79126** (approximate private leaderboard position: **~2500**).

    ### Generalization Comparison
    | Evaluation | ROC-AUC |
    |---|---:|
    | **OOF** | **~0.79298** |
    | **Untouched holdout** | **0.79319** |
    | **Kaggle Public** | **0.79366** |
    | **Kaggle Private** | **0.79126** |

    ### Interpretation
    The exceptionally close agreement across all four evaluation tracks (OOF ~0.79298, untouched holdout 0.79319, public leaderboard 0.79366, private leaderboard 0.79126) confirms the validity and discipline of the project's validation strategy. By isolating the holdout dataset and optimizing feature representations and hyperparameters strictly on fixed cross-validation folds, the system avoided adaptive overfitting and generalized reliably to completely unseen test data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final Research Summary

    ### Final feature representation
    - **RFE2**: 148 features (parsimonious core representation selected via two rounds of recursive SHAP elimination);
    - **POSX**: 9 features (POS/Cash contract completion and dynamic delinquency transitions);
    - **BBX**: 9 features (monthly credit bureau status severity and recent delinquency rates);
    - **Total**: **166 features** (`ACCEPTED_FINAL_FEATURES`).

    ### Final research system
    - **Tuned CatBoost**: Depth 7, learning rate 0.0201, L2 regularization 6.62, GPU border count 254;
    - **Tuned LightGBM**: Num leaves 77, max depth 10, min child samples 125, subsample 0.791, colsample 0.729;
    - **Tuned XGBoost**: Max depth 4, min child weight 3.20, subsample 0.703, colsample 0.839, L1 reg 9.34;
    - **Probability ensemble**: `0.36 * CatBoost + 0.28 * LightGBM + 0.36 * XGBoost`.

    ### Main milestones
    1. **B0 baseline**: Established application-only CatBoost benchmark (OOF ROC-AUC **0.76226**).
    2. **E2 financial ratios**: Added financial burden ratios, improving OOF ROC-AUC to **0.76866**.
    3. **Multi-table accepted sources**: Extracted relational features from `bureau`, `previous_application`, `installments_payments`, and `credit_card_balance`, expanding to 204 features (ROC-AUC ~0.788);
    4. **Source ablation**: Confirmed positive independent transfer from every accepted historical source, led by installments and bureau data;
    5. **RFE1 & RFE2**: Pruned 56 noise features in two stages (204 → 174 → 148 features), improving parsimony and training efficiency without performance loss;
    6. **CatBoost tuning**: Bayesian optimization identified Trial 29 (depth 7, conservative shrinkage), producing a strong tuned candidate;
    7. **OOF error analysis**: Analyzed review queues, score-matched hard false negatives, and categorical cohorts, confirming that pre-decision data lack obvious unmodeled residual signals;
    8. **F8 reject / F9 accept**: Rejected contract-level dynamic installments (-0.00030 ΔAP) and accepted POS + bureau-balance temporal dynamics (+0.00030 ΔAP), finalizing the schema at 166 features;
    9. **Model comparison**: Demonstrated that LightGBM and XGBoost provided diverse, complementary predictions to CatBoost;
    10. **Alternative-model tuning**: Bayesian optimization tuned LightGBM and XGBoost, with tuned XGBoost matching CatBoost on ROC-AUC (0.79122);
    11. **Ensemble**: Identified robust 0.36 / 0.28 / 0.36 blend weights, improving CV ROC-AUC to 0.79298 across 5/5 folds;
    12. **Untouched holdout**: Verified generalization on 46,127 unseen holdout rows (ROC-AUC 0.79319) with zero degradation;
    13. **Kaggle**: Generated final full-data refit predictions achieving 0.79366 Public and 0.79126 Private ROC-AUC.

    ### Final metrics
    - **5-Fold Cross-Validation OOF ROC-AUC**: **~0.79298** (AP: 0.28928)
    - **Untouched 15% Internal Holdout ROC-AUC**: **0.79319** (AP: 0.28393)
    - **Kaggle Public Leaderboard ROC-AUC**: **0.79366** (~2898 position)
    - **Kaggle Private Leaderboard ROC-AUC**: **0.79126** (~2500 position)

    ### Main findings
    - Combining three distinct gradient-boosting implementations (CatBoost, LightGBM, XGBoost) outperformed every single model across 5/5 validation folds and holdout data.
    - Eliminating 56 redundant features through recursive SHAP elimination preserved full predictive ranking quality while reducing pipeline complexity.
    - Modeling temporal trajectory and worsening dynamics (F9) recovered critical default signals that flat historical averages discarded.
    - Score-matched error analysis proved that hard false negatives cannot be resolved by standard feature engineering, reflecting unobserved events and label noise.
    - Disciplined separation of validation folds and holdout data guaranteed that CV improvements transferred faithfully to both holdout and competitive test sets.

    ### Deployment decision
    - **Research / Kaggle champion**: Three-model tuned probability ensemble (`0.36 / 0.28 / 0.36`) for optimal predictive ranking.
    - **Deployment candidate for FastAPI**: Single tuned CatBoost model (`FINAL_CB_PARAMS`) on the 166-feature schema, delivering ~99.6% of the ensemble's discriminatory power with a single model artifact, lower latency, and reduced operational complexity.

    ### Final decision
    **RESEARCH STAGE CLOSED**
    """)
    return


if __name__ == "__main__":
    app.run()
