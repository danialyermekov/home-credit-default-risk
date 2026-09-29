from fastapi.testclient import TestClient

from home_credit.core.dependencies import get_prediction_service
from home_credit.core.exceptions import ApplicantNotFoundError
from home_credit.main import app


class FakePredictionService:
    def __init__(
        self,
        probability: float | None = None,
        raise_not_found: bool = False,
    ) -> None:
        self.probability = probability
        self.raise_not_found = raise_not_found

    def predict_default_probability(
        self,
        applicant_id: int,
    ) -> float:
        if self.raise_not_found:
            raise ApplicantNotFoundError(f"Applicant {applicant_id} not found")

        assert self.probability is not None

        return self.probability


def test_predict_returns_probability(
    client: TestClient,
) -> None:
    service = FakePredictionService(
        probability=0.73,
    )

    app.dependency_overrides[get_prediction_service] = lambda: service

    response = client.post(
        "/prediction",
        json={"sk_id_curr": 123},
    )

    assert response.status_code == 200
    assert response.json() == {
        "sk_id_curr": 123,
        "probability": 0.73,
    }


def test_predict_returns_404_for_unknown_applicant(
    client: TestClient,
) -> None:
    service = FakePredictionService(
        raise_not_found=True,
    )

    app.dependency_overrides[get_prediction_service] = lambda: service

    response = client.post(
        "/prediction",
        json={"sk_id_curr": 999},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Applicant not found"}


def test_predict_returns_422_for_invalid_request(
    client: TestClient,
) -> None:
    response = client.post(
        "/prediction",
        json={"sk_id_curr": "abc"},
    )

    assert response.status_code == 422
