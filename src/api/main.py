"""The web API for purchase-intent scoring.

What happens to a request:
    1. When the server starts, lifespan() loads the saved model and the SHAP
       explainer once, and keeps them in app.state.scorer.
    2. A client sends a session as JSON to POST /predict.
    3. FastAPI checks the JSON against the Session schema (schemas.py).
       Anything invalid is rejected with a 422 error before our code runs.
    4. predict() hands the session to the scorer (scoring.py), which returns
       the probability, the yes/no decision and the top reasons.
    5. FastAPI checks the result against the Prediction schema and sends it
       back as JSON.

Run from the project root:
    uvicorn src.api.main:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request, Response

from src.api.schemas import Health, Prediction, Session
from src.api.scoring import Scorer
from src.config.settings import get_settings
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Code before `yield` runs once at startup; the server then handles requests."""
    setup_logging(settings.logs_dir)
    app.state.scorer = Scorer.load(settings)
    logger.info("Loaded model %s", app.state.scorer.model_version)
    yield


app = FastAPI(
    title="Purchase Intent API",
    version="1.0.0",
    description=(
        "Scores an online shopping session mid-visit: the chance it ends in a purchase, "
        "a yes/no decision, and the behaviours behind the score. Pass `threshold` to use "
        "your own operating point instead of the default."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log every request as key=value pairs, so log tools can filter and count them."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request method=%s path=%s status=%d duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health", response_model=Health)
def health() -> dict:
    """Check the server is up and see which model version it loaded."""
    return {"status": "ok", "model_version": app.state.scorer.model_version}


@app.post("/predict", response_model=Prediction)
def predict(
    session: Session,
    threshold: float | None = Query(
        default=None, gt=0, lt=1, description="Your own threshold, e.g. 0.4 (optional)"
    ),
) -> dict:
    """Score one session: probability, decision and the top 5 reasons."""
    return app.state.scorer.score(session.model_dump(), threshold)
