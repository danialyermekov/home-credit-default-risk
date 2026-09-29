import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from home_credit.core.exceptions import ApplicantNotFoundError
from home_credit.db.repository import FeatureRepository
from home_credit.schema_features import (
    ACCEPTED_CATEGORICAL_FEATURES,
    ACCEPTED_FINAL_FEATURES,
)


class PredictionService:
    def __init__(
        self,
        repository: FeatureRepository,
        model: CatBoostClassifier,
        expected_feature_version: str,
    ) -> None:
        self.repository = repository
        self.expected_feature_version = expected_feature_version

        self.model = model

    def predict_default_probability(
        self,
        applicant_id: int,
    ) -> float:
        features = self.repository.get_features(applicant_id)

        if features is None:
            raise ApplicantNotFoundError(f"Applicant {applicant_id} not found")

        feature_version = features.pop("feature_version")

        if feature_version != self.expected_feature_version:
            raise ValueError(
                "Feature version mismatch: "
                f"expected {self.expected_feature_version}, "
                f"got {feature_version}"
            )

        frame = self._prepare_features(features)

        probability = self.model.predict_proba(frame)[0, 1]

        return float(probability)

    def _prepare_features(
        self,
        features: dict[str, object],
    ) -> pd.DataFrame:
        row = {feature: features[feature] for feature in ACCEPTED_FINAL_FEATURES}

        frame = pd.DataFrame([row])

        categorical_features = set(ACCEPTED_CATEGORICAL_FEATURES)

        for feature in ACCEPTED_FINAL_FEATURES:
            if feature in categorical_features:
                frame[feature] = frame[feature].fillna("__MISSING__").astype(str)
            else:
                frame[feature] = pd.to_numeric(
                    frame[feature],
                    errors="coerce",
                )

        frame = frame.replace({None: np.nan})

        return frame
