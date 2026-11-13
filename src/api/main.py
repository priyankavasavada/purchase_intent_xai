"""FastAPI service for purchase-intent scoring.

Run from the project root:
    uvicorn src.api.main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request

from src.api.schemas import Health, Prediction, Session
from src.api.scoring import Scorer
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

DESCRIPTION = """
Scores an online shopping session mid-visit: the chance it ends in a purchase,
a yes/no decision, and the behaviours that drove the score (SHAP).

The default threshold is a neutral starting point. Pass `threshold` to use
your own operating point.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings to use a temporary model."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the model and explainer once, when the server starts.
        setup_logging(settings.logs_dir)
        app.state.scorer = Scorer.load(settings)
        logger.info("Loaded model %s", app.state.scorer.model_version)
        yield

    app = FastAPI(
        title="Purchase Intent API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/health", response_model=Health)
    def health(request: Request) -> dict:
        """Readiness check: the model is loaded and ready to score."""
        return {"status": "ok", "model_version": request.app.state.scorer.model_version}

    @app.post("/predict", response_model=Prediction)
    def predict(
        session: Session,
        request: Request,
        threshold: float | None = Query(
            default=None, gt=0, lt=1, description="Optional threshold instead of the default"
        ),
    ) -> dict:
        """Score one session and return the probability, decision and top behaviours."""
        return request.app.state.scorer.score(session.model_dump(), threshold)

    return app


app = create_app()
