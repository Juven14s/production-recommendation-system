"""FastAPI application for online recommendations."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi import Path as ApiPath
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from mle_recsys.api.cache import RecommendationCache, build_cache
from mle_recsys.api.engine import EnginePaths, RecommendationEngine
from mle_recsys.api.observability import (
    CACHE_EVENTS,
    FEATURE_AGE,
    FEATURE_LATEST_TIMESTAMP,
    HTTP_EXCEPTIONS,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS,
    MODEL_INFO,
    REQUESTS_IN_PROGRESS,
    normalized_endpoint,
)
from mle_recsys.mlops.registry import LocalModelRegistry


@dataclass(frozen=True)
class ServiceSettings:
    """Environment-backed service configuration."""

    train_path: Path = Path("data/processed/ml-1m/splits/train.parquet")
    movies_path: Path = Path("data/processed/ml-1m/movies.parquet")
    bpr_dir: Path = Path("artifacts/models/bpr")
    faiss_dir: Path = Path("artifacts/retrieval/faiss")
    ranker_dir: Path = Path("artifacts/models/ranker")
    redis_url: str | None = None
    cache_ttl_seconds: int = 300
    candidate_count: int = 200
    registry_root: Path | None = None

    @classmethod
    def from_environment(cls) -> ServiceSettings:
        registry_value = os.getenv("MODEL_REGISTRY_ROOT")
        default_registry = Path("artifacts/model_registry")
        registry_root = (
            Path(registry_value)
            if registry_value
            else default_registry
            if (default_registry / "registry.json").exists()
            else None
        )
        return cls(
            train_path=Path(
                os.getenv("TRAIN_PATH", "data/processed/ml-1m/splits/train.parquet")
            ),
            movies_path=Path(
                os.getenv("MOVIES_PATH", "data/processed/ml-1m/movies.parquet")
            ),
            bpr_dir=Path(os.getenv("BPR_DIR", "artifacts/models/bpr")),
            faiss_dir=Path(os.getenv("FAISS_DIR", "artifacts/retrieval/faiss")),
            ranker_dir=Path(os.getenv("RANKER_DIR", "artifacts/models/ranker")),
            redis_url=os.getenv("REDIS_URL"),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
            candidate_count=int(os.getenv("CANDIDATE_COUNT", "200")),
            registry_root=registry_root,
        )


class Recommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float
    rank: int
    source: str


class RecommendationResponse(BaseModel):
    user_id: int
    k: int
    model_version: str
    cold_start: bool
    cached: bool
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str
    cache_backend: str
    user_count: int = Field(ge=0)
    item_count: int = Field(ge=0)


def create_app(
    settings: ServiceSettings | None = None,
    engine: RecommendationEngine | None = None,
    cache: RecommendationCache | None = None,
) -> FastAPI:
    """Create an application with injectable engine and cache for tests."""

    resolved_settings = settings or ServiceSettings.from_environment()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> Iterator[None]:
        engine_paths = EnginePaths(
            train=resolved_settings.train_path,
            movies=resolved_settings.movies_path,
            bpr_dir=resolved_settings.bpr_dir,
            faiss_dir=resolved_settings.faiss_dir,
            ranker_dir=resolved_settings.ranker_dir,
        )
        if resolved_settings.registry_root is not None:
            bundle = LocalModelRegistry(
                resolved_settings.registry_root
            ).champion_bundle()
            engine_paths = EnginePaths(
                train=bundle / "data" / "train.parquet",
                movies=bundle / "data" / "movies.parquet",
                bpr_dir=bundle / "bpr",
                faiss_dir=bundle / "faiss",
                ranker_dir=bundle / "ranker",
            )
        application.state.engine = engine or RecommendationEngine(
            engine_paths,
            candidate_count=resolved_settings.candidate_count,
        )
        application.state.cache = cache or build_cache(resolved_settings.redis_url)
        MODEL_INFO.info(
            {
                "version": application.state.engine.model_version,
                "cache_backend": application.state.cache.backend_name,
            }
        )
        latest_timestamp = getattr(
            getattr(application.state.engine, "feature_context", None),
            "reference_timestamp",
            0,
        )
        FEATURE_LATEST_TIMESTAMP.set(latest_timestamp)
        FEATURE_AGE.set(max(time.time() - latest_timestamp, 0.0) if latest_timestamp else 0.0)
        yield

    application = FastAPI(
        title="Production Recommendation System",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Any) -> Response:
        method = request.method
        endpoint = normalized_endpoint(request.url.path)
        status = 500
        start = time.perf_counter()
        REQUESTS_IN_PROGRESS.labels(method=method).inc()
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception as error:
            HTTP_EXCEPTIONS.labels(
                endpoint=endpoint, exception_type=type(error).__name__
            ).inc()
            raise
        finally:
            REQUESTS_IN_PROGRESS.labels(method=method).dec()
            HTTP_REQUESTS.labels(
                method=method, endpoint=endpoint, status=str(status)
            ).inc()
            HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(
                time.perf_counter() - start
            )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict[str, Any]:
        active_engine = request.app.state.engine
        active_cache = request.app.state.cache
        return {
            "status": "ok",
            "model_version": active_engine.model_version,
            "cache_backend": active_cache.backend_name,
            "user_count": active_engine.user_count,
            "item_count": active_engine.item_count,
        }

    @application.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.get(
        "/recommendations/{user_id}", response_model=RecommendationResponse
    )
    def recommendations(
        request: Request,
        user_id: Annotated[int, ApiPath(gt=0)],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
    ) -> dict[str, Any]:
        active_engine = request.app.state.engine
        active_cache = request.app.state.cache
        cache_key = f"recommendations:{active_engine.model_version}:{user_id}:{k}"
        cached_payload = active_cache.get(cache_key)
        if cached_payload is not None:
            CACHE_EVENTS.labels(result="hit", backend=active_cache.backend_name).inc()
            cached_payload["cached"] = True
            return cached_payload

        CACHE_EVENTS.labels(result="miss", backend=active_cache.backend_name).inc()
        result = active_engine.recommend(user_id, k)
        payload = {
            "user_id": user_id,
            "k": k,
            "model_version": active_engine.model_version,
            "cold_start": result["cold_start"],
            "cached": False,
            "recommendations": result["recommendations"],
        }
        active_cache.set(cache_key, payload, resolved_settings.cache_ttl_seconds)
        return payload

    return application


app = create_app()


def run() -> None:
    uvicorn.run("mle_recsys.api.app:app", host="0.0.0.0", port=8000, reload=False)
