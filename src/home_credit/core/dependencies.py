from functools import lru_cache
from typing import Annotated

from catboost import CatBoostClassifier
from fastapi import Depends
from sqlalchemy import Engine

from home_credit.core.config import Settings, get_settings
from home_credit.db.connection import get_engine
from home_credit.db.repository import FeatureRepository
from home_credit.services.prediction import PredictionService


@lru_cache
def get_db_engine() -> Engine:
    return get_engine()


def get_feature_repository(
    engine: Annotated[Engine, Depends(get_db_engine)],
) -> FeatureRepository:
    return FeatureRepository(engine)


@lru_cache
def get_model() -> CatBoostClassifier:
    settings = get_settings()

    model = CatBoostClassifier()
    model.load_model(settings.model_path)

    return model


def get_prediction_service(
    repository: Annotated[
        FeatureRepository,
        Depends(get_feature_repository),
    ],
    model: Annotated[
        CatBoostClassifier,
        Depends(get_model),
    ],
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> PredictionService:
    return PredictionService(
        repository=repository,
        model=model,
        expected_feature_version=settings.feature_version,
    )
