"""FastAPI recommendation serving app."""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.serving.schemas import (
    FeedbackRequest,
    RecommendationRequest,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Global pipeline instance (set during startup)
_pipeline = None
_startup_time: float = 0.0
_model_info: dict[str, str] = {}


def _load_pipeline():
    """Load all models and build the pipeline. Raises on failure."""
    from src.features.feature_builder import FeatureBuilder
    from src.features.redis_store import RedisFeatureStore
    from src.models.collaborative import MatrixFactorizationModel
    from src.models.ranker import XGBoostRanker
    from src.models.embeddings import load_embeddings, load_projection_matrix
    from src.retrieval.elastic_store import ElasticsearchItemStore
    from src.serving.pipeline import RecommendationPipeline

    cf_path = os.getenv("CF_MODEL_PATH", "models/cf")
    ranker_path = os.getenv("RANKER_MODEL_PATH", "models/ranker")

    logger.info("Loading CF model from %s ...", cf_path)
    cf_model = MatrixFactorizationModel.load(cf_path)

    logger.info("Loading XGBoost ranker from %s ...", ranker_path)
    ranker = XGBoostRanker.load(ranker_path)

    logger.info("Loading item embeddings and projection matrix...")
    embeddings = load_embeddings()
    proj = load_projection_matrix()

    redis_store = RedisFeatureStore()
    elastic_store = ElasticsearchItemStore(
        embeddings=embeddings, projection_matrix=proj
    )
    feature_builder = FeatureBuilder()

    pipeline = RecommendationPipeline(
        redis_store=redis_store,
        elastic_store=elastic_store,
        cf_model=cf_model,
        ranker=ranker,
        feature_builder=feature_builder,
    )

    global _model_info
    _model_info = {
        "cf_model_path": cf_path,
        "ranker_model_path": ranker_path,
        "cf_factors": str(cf_model.factors),
    }

    from src.monitoring.metrics import set_model_info

    set_model_info(cf_version="1.0", ranker_version="1.0")

    logger.info("Pipeline ready.")
    return pipeline, redis_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _startup_time, _redis_store
    _startup_time = time.time()
    try:
        _pipeline, _redis_store = _load_pipeline()
    except Exception as e:
        logger.error("Failed to load models: %s", e)
        raise RuntimeError(f"Model loading failed: {e}") from e
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="StreamRank RecSys API",
    description="Real-time product recommendation engine",
    version="1.0.0",
    lifespan=lifespan,
)

_redis_store = None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    try:
        return _pipeline.recommend(
            user_id=request.user_id,
            n=request.n_recommendations,
            context=request.context or {},
        )
    except Exception as e:
        logger.exception("Error during recommendation for user %s", request.user_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _pipeline is not None,
        "timestamp": time.time(),
        "uptime_seconds": time.time() - _startup_time if _startup_time else 0,
    }


@app.get("/model-info")
def model_info():
    return {"model_info": _model_info, "pipeline_version": "1.0.0"}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    if _redis_store is None:
        raise HTTPException(status_code=503, detail="Redis not ready")
    event = {
        "event_id": f"fb-{int(time.time())}",
        "user_id": request.user_id,
        "item_id": request.item_id,
        "event_type": request.event_type,
        "session_id": request.session_id or "",
        "timestamp": time.time(),
        "metadata": request.context or {},
    }
    _redis_store.update_session(request.user_id, event)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serving.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,
    )
