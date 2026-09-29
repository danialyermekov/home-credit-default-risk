from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy import Engine, text

from home_credit.api.prediction import router as prediction_router
from home_credit.core.dependencies import get_db_engine

app = FastAPI(title="Home Credit")

app.include_router(prediction_router)


@app.get("/health")
def health(
    engine: Annotated[
        Engine,
        Depends(get_db_engine),
    ],
) -> dict[str, str]:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    return {"status": "ok"}
