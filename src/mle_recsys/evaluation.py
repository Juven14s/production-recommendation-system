"""Offline top-K evaluation utilities."""

from __future__ import annotations

import math

import pandas as pd


def evaluate_single_relevant_item(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog_size: int,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float | int]:
    """Compute Recall@K, NDCG@K and catalog coverage for one target per user."""

    if catalog_size <= 0:
        raise ValueError("catalog_size must be strictly positive")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain strictly positive integers")
    if ground_truth["user_id"].duplicated().any():
        raise ValueError("ground_truth must contain exactly one target per user")

    targets = ground_truth.set_index("user_id")["movie_id"].to_dict()
    ranked = recommendations.sort_values(["user_id", "rank"])
    by_user = ranked.groupby("user_id")["movie_id"].apply(list).to_dict()

    metrics: dict[str, float | int] = {"evaluated_user_count": len(targets)}
    for k in sorted(set(k_values)):
        recalls: list[float] = []
        ndcgs: list[float] = []
        covered_movies: set[int] = set()
        for user_id, target_movie_id in targets.items():
            top_k = by_user.get(user_id, [])[:k]
            covered_movies.update(int(movie_id) for movie_id in top_k)
            if target_movie_id in top_k:
                rank = top_k.index(target_movie_id)
                recalls.append(1.0)
                ndcgs.append(1.0 / math.log2(rank + 2))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)

        metrics[f"recall@{k}"] = float(sum(recalls) / len(recalls)) if recalls else 0.0
        metrics[f"ndcg@{k}"] = float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0
        metrics[f"catalog_coverage@{k}"] = len(covered_movies) / catalog_size
    return metrics
