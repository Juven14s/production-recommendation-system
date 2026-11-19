"""Leakage-safe temporal splits for implicit recommendation experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mle_recsys.data.contracts import validate_ratings


@dataclass(frozen=True)
class TemporalSplit:
    """Positive interactions split into train, validation and test tables."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    report: dict[str, object]


def leave_two_out_split(
    ratings: pd.DataFrame,
    positive_threshold: float = 4.0,
    minimum_positive_interactions: int = 3,
) -> TemporalSplit:
    """Reserve each eligible user's two latest positive events.

    The latest positive interaction becomes test data and the preceding one
    becomes validation data. Users with too little positive history remain in
    training and are excluded from offline ranking evaluation.
    """

    validate_ratings(ratings)
    if not 0.5 <= positive_threshold <= 5.0:
        raise ValueError("positive_threshold must be between 0.5 and 5.0")
    if minimum_positive_interactions < 3:
        raise ValueError("minimum_positive_interactions must be at least 3")

    positives = ratings.loc[ratings["rating"] >= positive_threshold].copy()
    positives = positives.sort_values(
        ["user_id", "timestamp", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)

    per_user_count = positives.groupby("user_id")["movie_id"].transform("size")
    eligible = per_user_count >= minimum_positive_interactions
    reverse_position = positives.groupby("user_id").cumcount(ascending=False)

    test_mask = eligible & reverse_position.eq(0)
    validation_mask = eligible & reverse_position.eq(1)
    train_mask = ~(test_mask | validation_mask)

    train = positives.loc[train_mask].reset_index(drop=True)
    validation = positives.loc[validation_mask].reset_index(drop=True)
    test = positives.loc[test_mask].reset_index(drop=True)

    eligible_user_count = int(validation["user_id"].nunique())
    report: dict[str, object] = {
        "strategy": "per_user_leave_two_out",
        "positive_rating_threshold": positive_threshold,
        "minimum_positive_interactions": minimum_positive_interactions,
        "source_interaction_count": int(len(ratings)),
        "positive_interaction_count": int(len(positives)),
        "eligible_user_count": eligible_user_count,
        "train_interaction_count": int(len(train)),
        "validation_interaction_count": int(len(validation)),
        "test_interaction_count": int(len(test)),
        "train_user_count": int(train["user_id"].nunique()),
        "train_movie_count": int(train["movie_id"].nunique()),
    }
    return TemporalSplit(train=train, validation=validation, test=test, report=report)


def write_split(split: TemporalSplit, output_dir: Path) -> None:
    """Persist a temporal split and its metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_parquet(output_dir / "train.parquet", index=False)
    split.validation.to_parquet(output_dir / "validation.parquet", index=False)
    split.test.to_parquet(output_dir / "test.parquet", index=False)
    with (output_dir / "split_report.json").open("w", encoding="utf-8") as stream:
        json.dump(split.report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a per-user temporal split from normalized ratings."
    )
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--minimum-positive-interactions", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ratings = pd.read_parquet(args.ratings)
    split = leave_two_out_split(
        ratings,
        positive_threshold=args.positive_threshold,
        minimum_positive_interactions=args.minimum_positive_interactions,
    )
    write_split(split, args.output_dir)
    print(json.dumps(split.report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
