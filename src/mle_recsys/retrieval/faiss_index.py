"""FAISS candidate retrieval built from BPR embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FaissArtifacts:
    """FAISS index and stable mappings needed by retrieval inference."""

    index: faiss.Index
    movie_ids: np.ndarray
    user_ids: np.ndarray
    user_embeddings: np.ndarray
    popularity_movie_ids: np.ndarray


def _as_float32_contiguous(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32)


def augment_item_vectors(
    item_embeddings: np.ndarray,
    item_bias: np.ndarray,
) -> np.ndarray:
    """Append item bias so inner product reproduces the full BPR score."""

    if len(item_embeddings) != len(item_bias):
        raise ValueError("item_embeddings and item_bias must have the same row count")
    bias_column = np.asarray(item_bias, dtype=np.float32).reshape(-1, 1)
    return _as_float32_contiguous(np.concatenate([item_embeddings, bias_column], axis=1))


def augment_user_vectors(user_embeddings: np.ndarray) -> np.ndarray:
    """Append a constant one to pair with the augmented item-bias dimension."""

    bias_multiplier = np.ones((len(user_embeddings), 1), dtype=np.float32)
    return _as_float32_contiguous(
        np.concatenate([user_embeddings, bias_multiplier], axis=1)
    )


def build_flat_ip_index(
    item_embeddings: np.ndarray,
    item_bias: np.ndarray,
) -> faiss.IndexFlatIP:
    """Build an exact inner-product index from BPR item parameters."""

    item_vectors = augment_item_vectors(item_embeddings, item_bias)
    index = faiss.IndexFlatIP(item_vectors.shape[1])
    index.add(item_vectors)
    return index


def load_bpr_artifacts(bpr_dir: Path) -> FaissArtifacts:
    """Load portable BPR arrays and create their exact FAISS index."""

    item_embeddings = np.load(bpr_dir / "item_embeddings.npy", allow_pickle=False)
    item_bias = np.load(bpr_dir / "item_bias.npy", allow_pickle=False)
    movie_ids = np.load(bpr_dir / "movie_ids.npy", allow_pickle=False)
    user_embeddings = np.load(bpr_dir / "user_embeddings.npy", allow_pickle=False)
    user_ids = np.load(bpr_dir / "user_ids.npy", allow_pickle=False)
    popularity_movie_ids = np.load(
        bpr_dir / "popularity_movie_ids.npy", allow_pickle=False
    )
    if len(item_embeddings) != len(movie_ids):
        raise ValueError("BPR item embeddings and movie IDs are inconsistent")
    if len(user_embeddings) != len(user_ids):
        raise ValueError("BPR user embeddings and user IDs are inconsistent")

    return FaissArtifacts(
        index=build_flat_ip_index(item_embeddings, item_bias),
        movie_ids=movie_ids.astype(np.int64),
        user_ids=user_ids.astype(np.int64),
        user_embeddings=_as_float32_contiguous(user_embeddings),
        popularity_movie_ids=popularity_movie_ids.astype(np.int64),
    )


def retrieve_candidates(
    artifacts: FaissArtifacts,
    train: pd.DataFrame,
    users: list[int],
    k: int,
    search_batch_size: int = 256,
    history_by_user: dict[int, set[int]] | None = None,
) -> pd.DataFrame:
    """Retrieve top-K unseen candidates, with popularity fallback for cold start."""

    if k <= 0 or search_batch_size <= 0:
        raise ValueError("k and search_batch_size must be strictly positive")

    user_to_index = {
        int(user_id): index for index, user_id in enumerate(artifacts.user_ids)
    }
    history_by_user = history_by_user or {
        int(user_id): {int(movie_id) for movie_id in movie_ids}
        for user_id, movie_ids in train.groupby("user_id")["movie_id"]
    }
    rows: list[dict[str, int | float | str]] = []

    for batch_start in range(0, len(users), search_batch_size):
        batch_users = users[batch_start : batch_start + search_batch_size]
        known_users = [user_id for user_id in batch_users if user_id in user_to_index]
        retrieved_by_user: dict[int, list[tuple[int, float]]] = {}

        if known_users:
            internal_indices = [user_to_index[user_id] for user_id in known_users]
            query_vectors = augment_user_vectors(artifacts.user_embeddings[internal_indices])
            maximum_seen = max(len(history_by_user.get(user_id, set())) for user_id in known_users)
            search_k = min(artifacts.index.ntotal, k + maximum_seen)
            scores, positions = artifacts.index.search(query_vectors, search_k)

            for row_index, user_id in enumerate(known_users):
                seen = history_by_user.get(user_id, set())
                candidates: list[tuple[int, float]] = []
                for position, score in zip(
                    positions[row_index], scores[row_index], strict=True
                ):
                    if position < 0:
                        continue
                    movie_id = int(artifacts.movie_ids[position])
                    if movie_id in seen:
                        continue
                    candidates.append((movie_id, float(score)))
                    if len(candidates) == k:
                        break
                retrieved_by_user[user_id] = candidates

        for user_id in batch_users:
            selected = retrieved_by_user.get(user_id, [])
            selected_ids = {movie_id for movie_id, _ in selected}
            seen = history_by_user.get(user_id, set())
            faiss_count = len(selected)
            if len(selected) < k:
                for movie_id_value in artifacts.popularity_movie_ids:
                    movie_id = int(movie_id_value)
                    if movie_id in seen or movie_id in selected_ids:
                        continue
                    selected.append((movie_id, 0.0))
                    selected_ids.add(movie_id)
                    if len(selected) == k:
                        break

            for rank, (movie_id, score) in enumerate(selected, start=1):
                rows.append(
                    {
                        "user_id": int(user_id),
                        "movie_id": movie_id,
                        "rank": rank,
                        "retrieval_score": score,
                        "source": "faiss" if rank <= faiss_count else "popularity",
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["user_id", "movie_id", "rank", "retrieval_score", "source"],
    )


def evaluate_candidate_retrieval(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    catalog_size: int,
    k_values: tuple[int, ...],
) -> dict[str, float | int]:
    """Evaluate whether the held-out relevant item survives candidate retrieval."""

    if catalog_size <= 0:
        raise ValueError("catalog_size must be strictly positive")
    if ground_truth["user_id"].duplicated().any():
        raise ValueError("ground_truth must contain one target per user")

    targets = ground_truth.set_index("user_id")["movie_id"].to_dict()
    ranked = candidates.sort_values(["user_id", "rank"])
    by_user = ranked.groupby("user_id")["movie_id"].apply(list).to_dict()
    metrics: dict[str, float | int] = {"evaluated_user_count": len(targets)}

    for k in sorted(set(k_values)):
        hit_count = 0
        covered_movies: set[int] = set()
        for user_id, target_movie_id in targets.items():
            top_k = by_user.get(user_id, [])[:k]
            covered_movies.update(int(movie_id) for movie_id in top_k)
            hit_count += int(target_movie_id in top_k)
        metrics[f"candidate_recall@{k}"] = hit_count / len(targets) if targets else 0.0
        metrics[f"candidate_coverage@{k}"] = len(covered_movies) / catalog_size
    return metrics


def benchmark_index(
    index: faiss.Index,
    user_embeddings: np.ndarray,
    k: int,
    sample_size: int = 500,
    seed: int = 42,
) -> dict[str, float | int]:
    """Benchmark single-query FAISS latency after one warm-up search."""

    if len(user_embeddings) == 0:
        raise ValueError("user_embeddings must not be empty")
    random_generator = np.random.default_rng(seed)
    sample_indices = random_generator.choice(
        len(user_embeddings), size=min(sample_size, len(user_embeddings)), replace=False
    )
    queries = augment_user_vectors(user_embeddings[sample_indices])
    search_k = min(k, index.ntotal)
    index.search(queries[:1], search_k)

    latencies_ms: list[float] = []
    for query in queries:
        start = time.perf_counter()
        index.search(query.reshape(1, -1), search_k)
        latencies_ms.append((time.perf_counter() - start) * 1_000)

    return {
        "latency_sample_count": len(latencies_ms),
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "latency_p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_index(
    artifacts: FaissArtifacts,
    bpr_dir: Path,
    output_dir: Path,
) -> None:
    """Persist the index with mappings and source-artifact fingerprints."""

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(artifacts.index, str(output_dir / "items.index"))
    np.save(output_dir / "movie_ids.npy", artifacts.movie_ids, allow_pickle=False)
    metadata = {
        "index_type": "IndexFlatIP",
        "exact_search": True,
        "item_count": int(artifacts.index.ntotal),
        "index_dimension": int(artifacts.index.d),
        "faiss_version": faiss.__version__,
        "score": "user_embedding dot item_embedding + item_bias",
        "source_sha256": {
            name: _sha256(bpr_dir / name)
            for name in [
                "item_embeddings.npy",
                "item_bias.npy",
                "movie_ids.npy",
                "user_embeddings.npy",
                "user_ids.npy",
            ]
        },
    }
    with (output_dir / "index_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_faiss_retrieval(
    bpr_dir: Path,
    split_dir: Path,
    movies_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = (50, 100, 200),
    search_batch_size: int = 256,
) -> dict[str, object]:
    """Build, persist, benchmark and evaluate an exact FAISS index."""

    train = pd.read_parquet(split_dir / "train.parquet")
    validation = pd.read_parquet(split_dir / "validation.parquet")
    movies = pd.read_parquet(movies_path)
    artifacts = load_bpr_artifacts(bpr_dir)

    users = sorted(int(user_id) for user_id in validation["user_id"].unique())
    candidates = retrieve_candidates(
        artifacts,
        train,
        users,
        k=max(k_values),
        search_batch_size=search_batch_size,
    )
    metrics = evaluate_candidate_retrieval(
        candidates,
        validation.loc[:, ["user_id", "movie_id"]],
        catalog_size=int(movies["movie_id"].nunique()),
        k_values=k_values,
    )
    latency = benchmark_index(
        artifacts.index, artifacts.user_embeddings, k=max(k_values)
    )
    report: dict[str, object] = {
        "retriever": "faiss_index_flat_ip",
        "evaluation_split": "validation",
        "index_type": "IndexFlatIP",
        "exact_search": True,
        "candidate_k_values": list(k_values),
        **metrics,
        **latency,
    }

    save_index(artifacts, bpr_dir, output_dir)
    candidates.to_parquet(output_dir / "validation_candidates.parquet", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate an exact FAISS index from BPR embeddings."
    )
    parser.add_argument("--bpr-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--search-batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_faiss_retrieval(
        bpr_dir=args.bpr_dir,
        split_dir=args.split_dir,
        movies_path=args.movies,
        output_dir=args.output_dir,
        k_values=tuple(args.k),
        search_batch_size=args.search_batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
