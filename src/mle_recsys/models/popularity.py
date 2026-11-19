"""Global popularity baseline for recommendation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mle_recsys.evaluation import evaluate_single_relevant_item


def fit_popularity(train: pd.DataFrame) -> pd.DataFrame:
    """Rank movies by positive interaction count and mean rating."""

    model = (
        train.groupby("movie_id", as_index=False)
        .agg(interaction_count=("user_id", "size"), mean_rating=("rating", "mean"))
        .sort_values(
            ["interaction_count", "mean_rating", "movie_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    model["popularity_rank"] = model.index + 1
    return model


def recommend_popular(
    model: pd.DataFrame,
    train: pd.DataFrame,
    users: list[int],
    k: int,
) -> pd.DataFrame:
    """Recommend popular unseen movies to each requested user."""

    if k <= 0:
        raise ValueError("k must be strictly positive")

    seen_by_user = train.groupby("user_id")["movie_id"].agg(set).to_dict()
    candidates = model.loc[:, ["movie_id", "interaction_count"]].itertuples(index=False)
    candidate_rows = list(candidates)
    rows: list[dict[str, int | float]] = []

    for user_id in users:
        seen = seen_by_user.get(user_id, set())
        rank = 0
        for candidate in candidate_rows:
            if candidate.movie_id in seen:
                continue
            rank += 1
            rows.append(
                {
                    "user_id": int(user_id),
                    "movie_id": int(candidate.movie_id),
                    "rank": rank,
                    "score": float(candidate.interaction_count),
                }
            )
            if rank == k:
                break
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rank", "score"])


def run_popularity_baseline(
    split_dir: Path,
    movies_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, object]:
    """Fit the baseline and evaluate it on validation only."""

    train = pd.read_parquet(split_dir / "train.parquet")
    validation = pd.read_parquet(split_dir / "validation.parquet")
    movies = pd.read_parquet(movies_path)

    model = fit_popularity(train)
    max_k = max(k_values)
    users = sorted(int(user_id) for user_id in validation["user_id"].unique())
    recommendations = recommend_popular(model, train, users, max_k)
    metrics = evaluate_single_relevant_item(
        recommendations,
        validation.loc[:, ["user_id", "movie_id"]],
        catalog_size=int(movies["movie_id"].nunique()),
        k_values=k_values,
    )

    report: dict[str, object] = {
        "model": "global_popularity",
        "evaluation_split": "validation",
        **metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to_parquet(output_dir / "popularity_model.parquet", index=False)
    recommendations.to_parquet(
        output_dir / "validation_recommendations.parquet", index=False
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the global popularity baseline."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_popularity_baseline(
        split_dir=args.split_dir,
        movies_path=args.movies,
        output_dir=args.output_dir,
        k_values=tuple(args.k),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
