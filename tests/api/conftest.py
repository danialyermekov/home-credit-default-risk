from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from home_credit.core.dependencies import get_prediction_service
from home_credit.main import app


class FakePredictionService:
    def predict_proba(self, applicant_id: int) -> float:
        return 0.73


@pytest.fixture
def fake_prediction_service() -> FakePredictionService:
    return FakePredictionService()


@pytest.fixture
def client(
    fake_prediction_service: FakePredictionService,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_prediction_service] = lambda: fake_prediction_service

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
