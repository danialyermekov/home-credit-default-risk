from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from home_credit.core.dependencies import get_prediction_service
from home_credit.core.exceptions import ApplicantNotFoundError
from home_credit.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from home_credit.services.prediction import PredictionService

router = APIRouter(
    prefix="/prediction",
    tags=["Prediction"],
)


@router.post("", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    service: Annotated[
        PredictionService,
        Depends(get_prediction_service),
    ],
) -> PredictionResponse:
    try:
        probability = service.predict_default_probability(request.sk_id_curr)

    except ApplicantNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Applicant not found",
        ) from exc

    return PredictionResponse(
        sk_id_curr=request.sk_id_curr,
        probability=probability,
    )
