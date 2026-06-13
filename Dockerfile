FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY gateway ./gateway
COPY bench ./bench
COPY config ./config

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "gateway.server:app", "--host", "0.0.0.0", "--port", "8000"]
