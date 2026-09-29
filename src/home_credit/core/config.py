from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE = BASE_DIR / ".env"
MODELS_DIR = BASE_DIR / "artifacts/models/"


class Settings(BaseSettings):
    database_url: str
    model_path: Path = MODELS_DIR / "catboost_full_model.cbm"
    feature_version: str = "v1"

    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
