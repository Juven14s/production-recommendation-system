"""Leakage-safe LightGBM ranker over FAISS candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pandas as pd

from mle_recsys.evaluation import evaluate_single_relevant_item
from mle_recsys.retrieval.faiss_index import (
    FaissArtifacts,
    build_flat_ip_index,
    retrieve_candidates,
)

if TYPE_CHECKING:
    from mle_recsys.models.bpr import BPRArtifacts

FEATURE_COLUMNS = [
    "retrieval_score",
    "reciprocal_retrieval_rank",
    "user_activity_log1p",
    "movie_popularity_log1p",
    "movie_mean_rating",
    "movie_recency_days",
    "genre_affinity",
    "release_year_gap",
    "user_embedding_norm",
    "item_embedding_norm",
]


@dataclass(frozen=True)
class EmbeddingLookup:
    """Portable embeddings and biases used for candidate features."""

    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_embeddings: np.ndarray
    item_embeddings: np.ndarray
    item_bias: np.ndarray
    popularity_movie_ids: np.ndarray


@dataclass(frozen=True)
class RankerTrainingSplit:
    """Internal history and one subsequent positive target per user."""

    history: pd.DataFrame
    targets: pd.DataFrame


@dataclass(frozen=True)
class FeatureContext:
    """Precomputed online features shared by all inference requests."""

    user_activity: dict[int, int]
    movie_popularity: dict[int, int]
    movie_mean_rating: dict[int, float]
    movie_last_timestamp: dict[int, int]
    reference_timestamp: int
    genre_by_movie: dict[int, tuple[str, ...]]
    genre_profiles: dict[int, Counter[str]]
    year_by_movie: dict[int, float]
    mean_year_by_user: dict[int, float]
    user_norm_by_id: dict[int, float]
    item_norm_by_id: dict[int, float]


def make_ranker_training_split(train: pd.DataFrame) -> RankerTrainingSplit:
    """Hold out the latest training interaction for ranker supervision."""

    ordered = train.sort_values(
        ["user_id", "timestamp", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)
    per_user_count = ordered.groupby("user_id")["movie_id"].transform("size")
    reverse_position = ordered.groupby("user_id").cumcount(ascending=False)
    target_mask = per_user_count.ge(2) & reverse_position.eq(0)
    return RankerTrainingSplit(
        history=ordered.loc[~target_mask].reset_index(drop=True),
        targets=ordered.loc[target_mask, ["user_id", "movie_id", "timestamp"]].reset_index(
            drop=True
        ),
    )


def embedding_lookup_from_bpr(artifacts: BPRArtifacts) -> EmbeddingLookup:
    """Extract NumPy arrays from an in-memory BPR model."""

    return EmbeddingLookup(
        user_ids=artifacts.user_ids,
        movie_ids=artifacts.movie_ids,
        user_embeddings=artifacts.model.user_embeddings.weight.detach().numpy(),
        item_embeddings=artifacts.model.item_embeddings.weight.detach().numpy(),
        item_bias=artifacts.model.item_bias.weight.detach().numpy(),
        popularity_movie_ids=artifacts.popularity_movie_ids,
    )


def load_embedding_lookup(bpr_dir: Path) -> EmbeddingLookup:
    """Load portable embeddings produced by the BPR training step."""

    return EmbeddingLookup(
        user_ids=np.load(bpr_dir / "user_ids.npy", allow_pickle=False),
        movie_ids=np.load(bpr_dir / "movie_ids.npy", allow_pickle=False),
        user_embeddings=np.load(bpr_dir / "user_embeddings.npy", allow_pickle=False),
        item_embeddings=np.load(bpr_dir / "item_embeddings.npy", allow_pickle=False),
        item_bias=np.load(bpr_dir / "item_bias.npy", allow_pickle=False),
        popularity_movie_ids=np.load(
            bpr_dir / "popularity_movie_ids.npy", allow_pickle=False
        ),
    )


def faiss_artifacts_from_embeddings(lookup: EmbeddingLookup) -> FaissArtifacts:
    """Build an in-memory exact FAISS retriever from an embedding lookup."""

    return FaissArtifacts(
        index=build_flat_ip_index(lookup.item_embeddings, lookup.item_bias),
        movie_ids=lookup.movie_ids,
        user_ids=lookup.user_ids,
        user_embeddings=np.ascontiguousarray(lookup.user_embeddings, dtype=np.float32),
        popularity_movie_ids=lookup.popularity_movie_ids,
    )


def select_retrieved_training_groups(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    lookup: EmbeddingLookup,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep groups whose positive target naturally survived retrieval."""

    known_movies = set(int(movie_id) for movie_id in lookup.movie_ids)
    known_users = set(int(user_id) for user_id in lookup.user_ids)
    eligible_targets = targets.loc[
        targets["movie_id"].isin(known_movies) & targets["user_id"].isin(known_users)
    ].copy()
    target_pairs = set(
        eligible_targets.loc[:, ["user_id", "movie_id"]].itertuples(
            index=False, name=None
        )
    )
    retrieved_pairs = set(
        candidates.loc[:, ["user_id", "movie_id"]].itertuples(index=False, name=None)
    )
    retrieved_target_pairs = target_pairs & retrieved_pairs
    retrieved_users = {int(user_id) for user_id, _ in retrieved_target_pairs}
    selected = candidates.loc[candidates["user_id"].isin(retrieved_users)].copy()
    report = {
        "eligible_target_count": int(len(eligible_targets)),
        "cold_target_count": int(len(targets) - len(eligible_targets)),
        "retrieved_positive_group_count": int(len(retrieved_users)),
        "dropped_missing_positive_count": int(len(eligible_targets) - len(retrieved_users)),
    }
    return selected, report


def _movie_genres(movies: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    return {
        int(row.movie_id): tuple(
            genre for genre in str(row.genres).split("|") if genre != "(no genres listed)"
        )
        for row in movies.itertuples(index=False)
    }


def _user_genre_profiles(
    history: pd.DataFrame,
    genre_by_movie: dict[int, tuple[str, ...]],
) -> dict[int, Counter[str]]:
    profiles: dict[int, Counter[str]] = {}
    for user_id, movie_id in history.loc[:, ["user_id", "movie_id"]].itertuples(
        index=False
    ):
        profile = profiles.setdefault(int(user_id), Counter())
        profile.update(genre_by_movie.get(int(movie_id), ()))
    return profiles


def build_feature_context(
    history: pd.DataFrame,
    movies: pd.DataFrame,
    lookup: EmbeddingLookup,
) -> FeatureContext:
    """Precompute serving-time feature maps once per model version."""

    user_stats = history.groupby("user_id").agg(user_activity=("movie_id", "size"))
    movie_stats = history.groupby("movie_id").agg(
        movie_popularity=("user_id", "size"),
        movie_mean_rating=("rating", "mean"),
        movie_last_timestamp=("timestamp", "max"),
    )
    genre_by_movie = _movie_genres(movies)
    genre_profiles = _user_genre_profiles(history, genre_by_movie)

    release_year = pd.to_numeric(
        movies["title"].str.extract(r"\((\d{4})\)$", expand=False), errors="coerce"
    )
    year_by_movie = {
        int(movie_id): float(year)
        for movie_id, year in zip(movies["movie_id"], release_year, strict=True)
        if pd.notna(year)
    }
    history_years = history["movie_id"].map(year_by_movie)
    mean_year = history.assign(release_year=history_years).groupby("user_id")[
        "release_year"
    ].mean()

    user_norms = np.linalg.norm(lookup.user_embeddings, axis=1)
    item_norms = np.linalg.norm(lookup.item_embeddings, axis=1)
    return FeatureContext(
        user_activity={
            int(key): int(value) for key, value in user_stats["user_activity"].items()
        },
        movie_popularity={
            int(key): int(value) for key, value in movie_stats["movie_popularity"].items()
        },
        movie_mean_rating={
            int(key): float(value) for key, value in movie_stats["movie_mean_rating"].items()
        },
        movie_last_timestamp={
            int(key): int(value)
            for key, value in movie_stats["movie_last_timestamp"].items()
        },
        reference_timestamp=int(history["timestamp"].max()),
        genre_by_movie=genre_by_movie,
        genre_profiles=genre_profiles,
        year_by_movie=year_by_movie,
        mean_year_by_user={
            int(key): float(value) for key, value in mean_year.dropna().items()
        },
        user_norm_by_id=dict(
            zip(lookup.user_ids.tolist(), user_norms.tolist(), strict=True)
        ),
        item_norm_by_id=dict(
            zip(lookup.movie_ids.tolist(), item_norms.tolist(), strict=True)
        ),
    )


def build_candidate_features(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    movies: pd.DataFrame,
    lookup: EmbeddingLookup,
    targets: pd.DataFrame | None = None,
    context: FeatureContext | None = None,
) -> pd.DataFrame:
    """Build features available at serving time for every user/movie candidate."""

    features = candidates.copy()
    context = context or build_feature_context(history, movies, lookup)

    features["reciprocal_retrieval_rank"] = 1.0 / features["rank"].clip(lower=1)
    features["user_activity_log1p"] = np.log1p(
        features["user_id"].map(context.user_activity).fillna(0)
    )
    features["movie_popularity_log1p"] = np.log1p(
        features["movie_id"].map(context.movie_popularity).fillna(0)
    )
    features["movie_mean_rating"] = (
        features["movie_id"].map(context.movie_mean_rating).fillna(0.0)
    )
    last_timestamp = features["movie_id"].map(context.movie_last_timestamp)
    features["movie_recency_days"] = (
        (context.reference_timestamp - last_timestamp)
        .clip(lower=0)
        .fillna(context.reference_timestamp)
        / 86_400.0
    )

    features["genre_affinity"] = [
        sum(
            context.genre_profiles.get(int(user_id), Counter()).get(genre, 0)
            for genre in context.genre_by_movie.get(int(movie_id), ())
        )
        / max(int(context.user_activity.get(user_id, 0)), 1)
        for user_id, movie_id in features.loc[:, ["user_id", "movie_id"]].itertuples(
            index=False
        )
    ]

    candidate_year = features["movie_id"].map(context.year_by_movie)
    user_mean_year = features["user_id"].map(context.mean_year_by_user)
    features["release_year_gap"] = (candidate_year - user_mean_year).abs().fillna(0.0)

    features["user_embedding_norm"] = (
        features["user_id"].map(context.user_norm_by_id).fillna(0.0)
    )
    features["item_embedding_norm"] = (
        features["movie_id"].map(context.item_norm_by_id).fillna(0.0)
    )

    features["label"] = 0
    if targets is not None:
        positive_pairs = set(
            targets.loc[:, ["user_id", "movie_id"]].itertuples(index=False, name=None)
        )
        features["label"] = [
            int((user_id, movie_id) in positive_pairs)
            for user_id, movie_id in features.loc[:, ["user_id", "movie_id"]].itertuples(
                index=False, name=None
            )
        ]

    numeric_columns = FEATURE_COLUMNS + ["label"]
    if features[numeric_columns].isna().any().any():
        raise ValueError("candidate feature table contains null numeric values")
    return features


def fit_ranker(
    training_features: pd.DataFrame,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
) -> lgb.LGBMRanker:
    """Fit a LambdaRank model using user candidate sets as query groups."""

    ordered = training_features.sort_values(["user_id", "rank"], kind="mergesort")
    group_sizes = ordered.groupby("user_id", sort=False).size().to_numpy()
    positive_per_group = ordered.groupby("user_id", sort=False)["label"].sum()
    if not positive_per_group.eq(1).all():
        raise ValueError("every ranker training group must contain exactly one positive")

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(
        ordered.loc[:, FEATURE_COLUMNS],
        ordered["label"],
        group=group_sizes,
    )
    return model


def rank_candidates(
    model: lgb.LGBMRanker,
    features: pd.DataFrame,
    k: int,
    blend_alpha: float = 1.0,
) -> pd.DataFrame:
    """Blend ranker and retrieval percentiles, then return top-K candidates."""

    if not 0.0 <= blend_alpha <= 1.0:
        raise ValueError("blend_alpha must be between zero and one")

    ranked = features.copy()
    ranked["lightgbm_score"] = model.predict(ranked.loc[:, FEATURE_COLUMNS])
    group_sizes = ranked.groupby("user_id")["movie_id"].transform("size")
    denominator = (group_sizes - 1).clip(lower=1)
    lightgbm_position = (
        ranked.groupby("user_id")["lightgbm_score"].rank(
            method="first", ascending=False
        )
        - 1
    )
    retrieval_position = ranked["rank"] - 1
    lightgbm_percentile = 1.0 - lightgbm_position / denominator
    retrieval_percentile = 1.0 - retrieval_position / denominator
    ranked["ranking_score"] = (
        blend_alpha * lightgbm_percentile
        + (1.0 - blend_alpha) * retrieval_percentile
    )
    ranked = ranked.sort_values(
        ["user_id", "ranking_score", "rank", "movie_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    ranked["rank"] = ranked.groupby("user_id").cumcount() + 1
    return ranked.loc[ranked["rank"] <= k].reset_index(drop=True)


def run_lightgbm_ranker(
    split_dir: Path,
    movies_path: Path,
    bpr_dir: Path,
    validation_candidates_path: Path,
    output_dir: Path,
    candidate_count: int = 200,
    k_values: tuple[int, ...] = (5, 10, 20),
    internal_bpr_epochs: int = 10,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
) -> dict[str, object]:
    """Train a leakage-safe ranker and evaluate it on validation candidates."""

    from mle_recsys.models.bpr import fit_bpr, save_bpr

    train = pd.read_parquet(split_dir / "train.parquet")
    validation = pd.read_parquet(split_dir / "validation.parquet")
    movies = pd.read_parquet(movies_path)
    internal_split = make_ranker_training_split(train)

    internal_bpr = fit_bpr(
        internal_split.history,
        embedding_dim=64,
        epochs=internal_bpr_epochs,
        batch_size=4096,
        learning_rate=0.01,
        regularization=1e-4,
        seed=seed,
        device="cpu",
    )
    internal_lookup = embedding_lookup_from_bpr(internal_bpr)
    internal_retriever = faiss_artifacts_from_embeddings(internal_lookup)
    training_users = sorted(int(user_id) for user_id in internal_split.targets["user_id"])
    training_candidates = retrieve_candidates(
        internal_retriever,
        internal_split.history,
        training_users,
        k=candidate_count,
    )
    training_candidates, selection_report = select_retrieved_training_groups(
        training_candidates,
        internal_split.targets,
        internal_lookup,
    )
    training_features = build_candidate_features(
        training_candidates,
        internal_split.history,
        movies,
        internal_lookup,
        targets=internal_split.targets,
    )
    model = fit_ranker(
        training_features,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        seed=seed,
    )

    validation_candidates = pd.read_parquet(validation_candidates_path)
    validation_candidates = validation_candidates.loc[
        validation_candidates["rank"] <= candidate_count
    ].copy()
    production_lookup = load_embedding_lookup(bpr_dir)
    validation_features = build_candidate_features(
        validation_candidates,
        train,
        movies,
        production_lookup,
    )
    ground_truth = validation.loc[:, ["user_id", "movie_id"]]
    catalog_size = int(movies["movie_id"].nunique())
    blend_sweep: list[dict[str, float]] = []
    ranked_by_alpha: dict[float, pd.DataFrame] = {}
    for blend_alpha in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        candidate_ranking = rank_candidates(
            model, validation_features, max(k_values), blend_alpha=blend_alpha
        )
        candidate_metrics = evaluate_single_relevant_item(
            candidate_ranking,
            ground_truth,
            catalog_size=catalog_size,
            k_values=k_values,
        )
        ranked_by_alpha[blend_alpha] = candidate_ranking
        blend_sweep.append(
            {
                "blend_alpha": blend_alpha,
                "ndcg@10": float(candidate_metrics.get("ndcg@10", 0.0)),
                "recall@10": float(candidate_metrics.get("recall@10", 0.0)),
            }
        )
    best_blend = max(
        blend_sweep, key=lambda result: (result["ndcg@10"], result["recall@10"])
    )
    blend_alpha = best_blend["blend_alpha"]
    ranked = ranked_by_alpha[blend_alpha]
    metrics = evaluate_single_relevant_item(
        ranked,
        ground_truth,
        catalog_size=catalog_size,
        k_values=k_values,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_bpr(internal_bpr, output_dir / "training_retriever")
    model.booster_.save_model(str(output_dir / "ranker.txt"))
    ranked.to_parquet(output_dir / "validation_recommendations.parquet", index=False)

    importance = {
        feature: int(value)
        for feature, value in zip(
            FEATURE_COLUMNS,
            model.booster_.feature_importance(importance_type="gain"),
            strict=True,
        )
    }
    with (output_dir / "feature_importance.json").open("w", encoding="utf-8") as stream:
        json.dump(importance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "blend_sweep.json").open("w", encoding="utf-8") as stream:
        json.dump(blend_sweep, stream, indent=2, sort_keys=True)
        stream.write("\n")

    report: dict[str, object] = {
        "model": "lightgbm_lambdarank",
        "evaluation_split": "validation",
        "candidate_count": candidate_count,
        "training_group_count": int(training_features["user_id"].nunique()),
        "training_row_count": int(len(training_features)),
        "internal_bpr_epochs": internal_bpr_epochs,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "blend_alpha": blend_alpha,
        "seed": seed,
        "lightgbm_version": lgb.__version__,
        **selection_report,
        **metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a LightGBM LambdaRank model."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--bpr-dir", type=Path, required=True)
    parser.add_argument("--validation-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=200)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--internal-bpr-epochs", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_lightgbm_ranker(
        split_dir=args.split_dir,
        movies_path=args.movies,
        bpr_dir=args.bpr_dir,
        validation_candidates_path=args.validation_candidates,
        output_dir=args.output_dir,
        candidate_count=args.candidate_count,
        k_values=tuple(args.k),
        internal_bpr_epochs=args.internal_bpr_epochs,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
