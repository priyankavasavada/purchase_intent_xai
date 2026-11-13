# Dashboard image: the Streamlit app on port 8501.
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

COPY requirements-api.txt requirements-dashboard.txt constraints.txt ./
RUN pip install -r requirements-dashboard.txt -c constraints.txt

# The dashboard reads the data, the saved models and the reports.
COPY src ./src
COPY dashboard ./dashboard
COPY models ./models
COPY dataset ./dataset
COPY reports ./reports

RUN useradd --create-home appuser && mkdir logs && chown appuser logs
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
