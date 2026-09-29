from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sk_id_curr: int = Field(...)


class PredictionResponse(BaseModel):
    sk_id_curr: int
    probability: float = Field(ge=0, description="Probability of default")
