import numpy as np
import pandas as pd
import pytest

from home_credit.core.exceptions import ApplicantNotFoundError
from home_credit.schema_features import (
    ACCEPTED_CATEGORICAL_FEATURES,
    ACCEPTED_FINAL_FEATURES,
)
from home_credit.services.prediction import PredictionService


class FakeRepository:
    def __init__(
        self,
        features: dict[str, object] | None,
    ) -> None:
        self.features = features

    def get_features(
        self,
        applicant_id: int,
    ) -> dict[str, object] | None:
        return self.features


class FakeModel:
    def __init__(self) -> None:
        self.received_features: pd.DataFrame | None = None

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        self.received_features = features.copy()

        return np.array([[0.27, 0.73]])


def make_features(
    feature_version: str = "v1",
) -> dict[str, object]:
    categorical_features = set(ACCEPTED_CATEGORICAL_FEATURES)

    features: dict[str, object] = {}

    for feature in ACCEPTED_FINAL_FEATURES:
        if feature in categorical_features:
            features[feature] = "TEST"
        else:
            features[feature] = 1.0

    features["feature_version"] = feature_version

    return features


def create_service(
    features: dict[str, object] | None,
    model: FakeModel | None = None,
) -> tuple[PredictionService, FakeModel]:
    model = model or FakeModel()
    repository = FakeRepository(features)

    service = PredictionService(
        repository=repository,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        expected_feature_version="v1",
    )

    return service, model


def test_predict_returns_default_probability() -> None:
    service, _ = create_service(
        make_features(),
    )

    probability = service.predict_default_probability(
        applicant_id=123,
    )

    assert probability == pytest.approx(0.73)


def test_predict_raises_for_unknown_applicant() -> None:
    service, _ = create_service(
        features=None,
    )

    with pytest.raises(ApplicantNotFoundError):
        service.predict_default_probability(
            applicant_id=123,
        )


def test_predict_raises_for_wrong_feature_version() -> None:
    service, _ = create_service(
        make_features(feature_version="v2"),
    )

    with pytest.raises(
        ValueError,
        match="Feature version mismatch",
    ):
        service.predict_default_probability(
            applicant_id=123,
        )


def test_predict_passes_expected_features_to_model() -> None:
    features = make_features()

    categorical_feature = ACCEPTED_CATEGORICAL_FEATURES[0]
    features[categorical_feature] = None

    service, model = create_service(
        features,
    )

    service.predict_default_probability(
        applicant_id=123,
    )

    assert model.received_features is not None

    assert tuple(model.received_features.columns) == ACCEPTED_FINAL_FEATURES

    assert model.received_features.shape == (
        1,
        len(ACCEPTED_FINAL_FEATURES),
    )

    assert (
        model.received_features.loc[
            0,
            categorical_feature,
        ]
        == "__MISSING__"
    )
