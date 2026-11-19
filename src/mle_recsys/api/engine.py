"""Online two-stage recommendation engine."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd

from mle_recsys.api.observability import (
    CANDIDATE_COUNT,
    ENGINE_STAGE_DURATION,
    RANKING_SCORE,
    RECOMMENDATION_REQUESTS,
    RECOMMENDATIONS_RETURNED,
)
from mle_recsys.ranking.lightgbm_ranker import (
    build_candidate_features,
    build_feature_context,
    load_embedding_lookup,
    rank_candidates,
)
from mle_recsys.retrieval.faiss_index import FaissArtifacts, retrieve_candidates


@dataclass(frozen=True)
class EnginePaths:
    """Filesystem locations for one deployable model bundle."""

    train: Path
    movies: Path
    bpr_dir: Path
    faiss_dir: Path
    ranker_dir: Path


class RecommendationEngine:
    """Load-once inference pipeline: FAISS, features, then LightGBM."""

    def __init__(
        self,
        paths: EnginePaths,
        candidate_count: int = 200,
    ) -> None:
        self.paths = paths
        self.candidate_count = candidate_count
        self.history = pd.read_parquet(paths.train)
        self.movies = pd.read_parquet(paths.movies)
        self.lookup = load_embedding_lookup(paths.bpr_dir)
        self.retriever = FaissArtifacts(
            index=faiss.read_index(str(paths.faiss_dir / "items.index")),
            movie_ids=np.load(paths.faiss_dir / "movie_ids.npy", allow_pickle=False),
            user_ids=self.lookup.user_ids,
            user_embeddings=np.ascontiguousarray(
                self.lookup.user_embeddings, dtype=np.float32
            ),
            popularity_movie_ids=self.lookup.popularity_movie_ids,
        )
        self.ranker = lgb.Booster(model_file=str(paths.ranker_dir / "ranker.txt"))
        ranker_metrics = json.loads(
            (paths.ranker_dir / "metrics.json").read_text(encoding="utf-8")
        )
        self.blend_alpha = float(ranker_metrics["blend_alpha"])
        self.feature_context = build_feature_context(
            self.history, self.movies, self.lookup
        )
        self.history_by_user = {
            int(user_id): {int(movie_id) for movie_id in movie_ids}
            for user_id, movie_ids in self.history.groupby("user_id")["movie_id"]
        }
        self.known_users = set(int(user_id) for user_id in self.lookup.user_ids)
        self.movie_metadata = self.movies.set_index("movie_id")[["title", "genres"]].to_dict(
            orient="index"
        )
        self.model_version = self._model_version()

    @property
    def user_count(self) -> int:
        return len(self.lookup.user_ids)

    @property
    def item_count(self) -> int:
        return len(self.lookup.movie_ids)

    def _model_version(self) -> str:
        digest = hashlib.sha256()
        for path in [
            self.paths.faiss_dir / "items.index",
            self.paths.ranker_dir / "ranker.txt",
            self.paths.bpr_dir / "user_embeddings.npy",
        ]:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()[:12]

    def _format_recommendations(self, ranked: pd.DataFrame) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []
        for row in ranked.sort_values("rank").itertuples(index=False):
            metadata = self.movie_metadata.get(int(row.movie_id), {})
            recommendations.append(
                {
                    "movie_id": int(row.movie_id),
                    "title": str(metadata.get("title", "Unknown")),
                    "genres": str(metadata.get("genres", "(no genres listed)")),
                    "score": float(row.ranking_score),
                    "rank": int(row.rank),
                    "source": str(row.source),
                }
            )
        return recommendations

    def _cold_start(self, k: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, movie_id_value in enumerate(
            self.lookup.popularity_movie_ids[:k], start=1
        ):
            movie_id = int(movie_id_value)
            metadata = self.movie_metadata.get(movie_id, {})
            rows.append(
                {
                    "movie_id": movie_id,
                    "title": str(metadata.get("title", "Unknown")),
                    "genres": str(metadata.get("genres", "(no genres listed)")),
                    "score": float(1.0 / rank),
                    "rank": rank,
                    "source": "popularity_cold_start",
                }
            )
        return rows

    def recommend(self, user_id: int, k: int) -> dict[str, Any]:
        """Return ranked recommendations for one known or cold-start user."""

        if user_id not in self.known_users:
            recommendations = self._cold_start(k)
            RECOMMENDATION_REQUESTS.labels(mode="cold_start").inc()
            RECOMMENDATIONS_RETURNED.labels(mode="cold_start").inc(len(recommendations))
            return {
                "cold_start": True,
                "recommendations": recommendations,
            }

        start = time.perf_counter()
        candidates = retrieve_candidates(
            self.retriever,
            self.history,
            [user_id],
            k=self.candidate_count,
            search_batch_size=1,
            history_by_user=self.history_by_user,
        )
        ENGINE_STAGE_DURATION.labels(stage="retrieval").observe(
            time.perf_counter() - start
        )
        CANDIDATE_COUNT.observe(len(candidates))

        start = time.perf_counter()
        features = build_candidate_features(
            candidates,
            self.history,
            self.movies,
            self.lookup,
            context=self.feature_context,
        )
        ENGINE_STAGE_DURATION.labels(stage="features").observe(time.perf_counter() - start)

        start = time.perf_counter()
        ranked = rank_candidates(
            self.ranker,
            features,
            k=k,
            blend_alpha=self.blend_alpha,
        )
        ENGINE_STAGE_DURATION.labels(stage="ranking").observe(time.perf_counter() - start)
        for score in ranked["ranking_score"]:
            RANKING_SCORE.observe(float(score))
        recommendations = self._format_recommendations(ranked)
        RECOMMENDATION_REQUESTS.labels(mode="personalized").inc()
        RECOMMENDATIONS_RETURNED.labels(mode="personalized").inc(len(recommendations))
        return {
            "cold_start": False,
            "recommendations": recommendations,
        }
