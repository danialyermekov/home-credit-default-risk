import logging
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier

from home_credit.features.assemble import build_feature_dataset
from home_credit.schema_features import (
    ACCEPTED_CATEGORICAL_FEATURES,
    ACCEPTED_FINAL_FEATURES,
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        data_dir: Path,
        model_params: dict[str, Any],
        model_save_path: Path = Path("artifacts/models/catboost_cli.cbm"),
        features: tuple[str, ...] = ACCEPTED_FINAL_FEATURES,
        cat_features: tuple[str, ...] = ACCEPTED_CATEGORICAL_FEATURES,
    ) -> None:
        self.data_dir = data_dir
        self.model_params = model_params
        self.model_save_path = model_save_path
        self.features = features
        self.cat_features = cat_features

    def _get_train_dataset(self) -> tuple[pd.DataFrame, pd.Series]:
        applications = pd.read_csv(self.data_dir / "application_train.csv")
        target = applications[["TARGET", "SK_ID_CURR"]].copy()

        tables = {
            "applications": applications.drop(columns="TARGET"),
            "bureau": pd.read_csv(self.data_dir / "bureau.csv"),
            "previous_application": pd.read_csv(
                self.data_dir / "previous_application.csv"
            ),
            "installments_payments": pd.read_csv(
                self.data_dir / "installments_payments.csv"
            ),
            "credit_card_balance": pd.read_csv(
                self.data_dir / "credit_card_balance.csv"
            ),
            "pos_cash_balance": pd.read_csv(self.data_dir / "POS_CASH_balance.csv"),
            "bureau_balance": pd.read_csv(self.data_dir / "bureau_balance.csv"),
        }

        logger.info("Building train dataset")

        features = build_feature_dataset(**tables, include_id=True)
        features_and_target = pd.merge(
            features[list(self.features) + ["SK_ID_CURR"]],
            target,
            on="SK_ID_CURR",
            validate="one_to_one",
        )

        if len(features_and_target) != len(target):
            raise ValueError(
                "Some training applicants were lost during feature assembly"
            )

        X = features_and_target.drop(columns=["SK_ID_CURR", "TARGET"])
        y = features_and_target["TARGET"]

        for column in self.cat_features:
            X[column] = X[column].astype("object").fillna("__MISSING__").astype(str)

        if tuple(X.columns) != self.features:
            raise ValueError("Training feature schema does not match production schema")

        logger.info("Training rows: %s", X.shape[0])
        logger.info("Features: %s", len(self.features))
        logger.info("Categorical features: %s", len(self.cat_features))

        return X, y

    def train_model(self) -> CatBoostClassifier:
        params = {
            **self.model_params,
            "allow_writing_files": False,
        }

        model = CatBoostClassifier(**params)

        features, target = self._get_train_dataset()

        logger.info("Training CatBoost Classifier")

        model.fit(features, target, cat_features=list(self.cat_features))

        model_path = Path(self.model_save_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)

        logger.info("Model saved: %s", model_path)

        return model
