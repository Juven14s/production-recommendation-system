"""Implicit item-to-item collaborative filtering baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from mle_recsys.evaluation import evaluate_single_relevant_item
from mle_recsys.models.popularity import fit_popularity


@dataclass(frozen=True)
class ItemKNNModel:
    """Sparse item similarity model and its item-index mapping."""

    movie_ids: np.ndarray
    similarities: sparse.csr_matrix
    popularity_movie_ids: np.ndarray
    neighbors_per_item: int
    shrinkage: float


def _build_user_item_matrix(
    interactions: pd.DataFrame,
    movie_ids: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    user_ids = np.sort(interactions["user_id"].unique())
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    movie_index = pd.Series(np.arange(len(movie_ids)), index=movie_ids)

    rows = interactions["user_id"].map(user_index).to_numpy(dtype=np.int32)
    columns = interactions["movie_id"].map(movie_index).to_numpy(dtype=np.int32)
    values = np.ones(len(interactions), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(user_ids), len(movie_ids)), dtype=np.float32
    )
    matrix.data[:] = 1.0
    return matrix, user_ids


def _keep_top_neighbors(
    similarities: sparse.csr_matrix,
    neighbors_per_item: int,
) -> sparse.csr_matrix:
    """Keep only the strongest neighbors in every similarity row."""

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    for row in range(similarities.shape[0]):
        start, end = similarities.indptr[row : row + 2]
        columns = similarities.indices[start:end]
        scores = similarities.data[start:end]
        if len(scores) > neighbors_per_item:
            selected = np.argpartition(scores, -neighbors_per_item)[-neighbors_per_item:]
            columns = columns[selected]
            scores = scores[selected]
        order = np.lexsort((columns, -scores))
        row_indices.extend([row] * len(order))
        column_indices.extend(columns[order].tolist())
        values.extend(scores[order].tolist())

    return sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=similarities.shape,
        dtype=np.float32,
    )


def fit_item_knn(
    train: pd.DataFrame,
    neighbors_per_item: int = 100,
    shrinkage: float = 10.0,
) -> ItemKNNModel:
    """Fit shrunk cosine similarities from positive user-item interactions."""

    if train.empty:
        raise ValueError("train must not be empty")
    if neighbors_per_item <= 0:
        raise ValueError("neighbors_per_item must be strictly positive")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")

    movie_ids = np.sort(train["movie_id"].unique())
    user_item, _ = _build_user_item_matrix(train, movie_ids)
    item_counts = np.asarray(user_item.sum(axis=0)).ravel()

    cooccurrences = (user_item.T @ user_item).tocoo()
    non_diagonal = cooccurrences.row != cooccurrences.col
    rows = cooccurrences.row[non_diagonal]
    columns = cooccurrences.col[non_diagonal]
    counts = cooccurrences.data[non_diagonal].astype(np.float32)

    denominators = np.sqrt(item_counts[rows] * item_counts[columns])
    cosine = np.divide(counts, denominators, out=np.zeros_like(counts), where=denominators > 0)
    shrunk = cosine * (counts / (counts + shrinkage))
    similarities = sparse.csr_matrix(
        (shrunk, (rows, columns)),
        shape=(len(movie_ids), len(movie_ids)),
        dtype=np.float32,
    )
    similarities.eliminate_zeros()
    similarities = _keep_top_neighbors(similarities, neighbors_per_item)

    popularity_movie_ids = fit_popularity(train)["movie_id"].to_numpy(dtype=np.int64)
    return ItemKNNModel(
        movie_ids=movie_ids.astype(np.int64),
        similarities=similarities,
        popularity_movie_ids=popularity_movie_ids,
        neighbors_per_item=neighbors_per_item,
        shrinkage=shrinkage,
    )


def recommend_item_knn(
    model: ItemKNNModel,
    train: pd.DataFrame,
    users: list[int],
    k: int,
) -> pd.DataFrame:
    """Score unseen movies from each user's positive history."""

    if k <= 0:
        raise ValueError("k must be strictly positive")

    movie_to_index = {int(movie_id): index for index, movie_id in enumerate(model.movie_ids)}
    history_by_user = train.groupby("user_id")["movie_id"].agg(list).to_dict()
    rows: list[dict[str, int | float]] = []

    for user_id in users:
        history = [int(movie_id) for movie_id in history_by_user.get(user_id, [])]
        seen = set(history)
        history_indices = [
            movie_to_index[movie_id]
            for movie_id in history
            if movie_id in movie_to_index
        ]

        scored_candidates: list[tuple[int, float]] = []
        if history_indices:
            scores = np.asarray(model.similarities[history_indices].sum(axis=0)).ravel()
            candidate_indices = np.flatnonzero(scores > 0)
            scored_candidates = [
                (int(model.movie_ids[index]), float(scores[index]))
                for index in candidate_indices
                if int(model.movie_ids[index]) not in seen
            ]
            scored_candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))

        selected = scored_candidates[:k]
        selected_movie_ids = {movie_id for movie_id, _ in selected}
        if len(selected) < k:
            for movie_id_value in model.popularity_movie_ids:
                movie_id = int(movie_id_value)
                if movie_id in seen or movie_id in selected_movie_ids:
                    continue
                selected.append((movie_id, 0.0))
                selected_movie_ids.add(movie_id)
                if len(selected) == k:
                    break

        for rank, (movie_id, score) in enumerate(selected, start=1):
            rows.append(
                {
                    "user_id": int(user_id),
                    "movie_id": movie_id,
                    "rank": rank,
                    "score": score,
                }
            )

    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rank", "score"])


def save_item_knn(model: ItemKNNModel, output_dir: Path) -> None:
    """Persist all artifacts required to reproduce Item-KNN inference."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "similarities.npz", model.similarities)
    np.save(output_dir / "movie_ids.npy", model.movie_ids, allow_pickle=False)
    np.save(
        output_dir / "popularity_movie_ids.npy",
        model.popularity_movie_ids,
        allow_pickle=False,
    )
    metadata = {
        "model": "implicit_item_knn",
        "neighbors_per_item": model.neighbors_per_item,
        "shrinkage": model.shrinkage,
        "item_count": int(len(model.movie_ids)),
        "nonzero_similarity_count": int(model.similarities.nnz),
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_item_knn_baseline(
    split_dir: Path,
    movies_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = (5, 10, 20),
    neighbors_per_item: int = 100,
    shrinkage: float = 10.0,
) -> dict[str, object]:
    """Fit Item-KNN and evaluate it on the untouched validation split."""

    train = pd.read_parquet(split_dir / "train.parquet")
    validation = pd.read_parquet(split_dir / "validation.parquet")
    movies = pd.read_parquet(movies_path)

    model = fit_item_knn(
        train,
        neighbors_per_item=neighbors_per_item,
        shrinkage=shrinkage,
    )
    users = sorted(int(user_id) for user_id in validation["user_id"].unique())
    recommendations = recommend_item_knn(model, train, users, max(k_values))
    metrics = evaluate_single_relevant_item(
        recommendations,
        validation.loc[:, ["user_id", "movie_id"]],
        catalog_size=int(movies["movie_id"].nunique()),
        k_values=k_values,
    )
    report: dict[str, object] = {
        "model": "implicit_item_knn",
        "evaluation_split": "validation",
        "neighbors_per_item": neighbors_per_item,
        "shrinkage": shrinkage,
        **metrics,
    }

    save_item_knn(model, output_dir)
    recommendations.to_parquet(
        output_dir / "validation_recommendations.parquet", index=False
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an implicit Item-KNN baseline."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--neighbors-per-item", type=int, default=100)
    parser.add_argument("--shrinkage", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_item_knn_baseline(
        split_dir=args.split_dir,
        movies_path=args.movies,
        output_dir=args.output_dir,
        k_values=tuple(args.k),
        neighbors_per_item=args.neighbors_per_item,
        shrinkage=args.shrinkage,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
