from sqlalchemy import Engine, create_engine, text

from home_credit.core.config import get_settings


def get_engine() -> Engine:
    settings = get_settings()

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )


def check_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        result = connection.execute(text("select 1"))

        assert result.scalar_one() == 1
