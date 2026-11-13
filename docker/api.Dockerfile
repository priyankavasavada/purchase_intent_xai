# API image: serves the saved model with FastAPI on port 8000.
FROM python:3.12-slim

# XGBoost needs the OpenMP runtime, which slim images leave out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install packages first, so this layer is reused when only the code changes.
COPY requirements-api.txt constraints.txt ./
RUN pip install -r requirements-api.txt -c constraints.txt

COPY src ./src
COPY models ./models

# Run as a normal user, not root. It only needs to write its log file.
RUN useradd --create-home appuser && mkdir logs && chown appuser logs
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
