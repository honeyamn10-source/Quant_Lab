FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY quantlab ./quantlab

RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir ".[web]"

EXPOSE 8000

CMD ["uvicorn", "quantlab.api.main:app", "--host", "0.0.0.0", "--port", "8000"]