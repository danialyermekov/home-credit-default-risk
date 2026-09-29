FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --locked --no-default-groups --no-install-project

COPY src ./src
COPY artifacts ./artifacts

RUN uv sync --locked --no-default-groups

CMD ["uv", "run", "--no-default-groups", "uvicorn", "home_credit.main:app", "--host", "0.0.0.0", "--port", "8000"]