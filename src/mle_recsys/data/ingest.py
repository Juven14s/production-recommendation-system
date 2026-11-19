"""Ingest and validate MovieLens datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

import pandas as pd

from mle_recsys.data.contracts import validate_movies, validate_ratings

CSV_RENAMES: Final = {
    "userId": "user_id",
    "movieId": "movie_id",
}


def _read_csv_dataset(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(input_dir / "ratings.csv")
    movies = pd.read_csv(input_dir / "movies.csv")
    return ratings.rename(columns=CSV_RENAMES), movies.rename(columns=CSV_RENAMES)


def _read_dat_dataset(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        input_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        input_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def read_movielens(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read a supported MovieLens directory and return normalized tables."""

    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    csv_files = (input_dir / "ratings.csv", input_dir / "movies.csv")
    dat_files = (input_dir / "ratings.dat", input_dir / "movies.dat")

    if all(path.is_file() for path in csv_files):
        ratings, movies = _read_csv_dataset(input_dir)
        source_format = "csv"
    elif all(path.is_file() for path in dat_files):
        ratings, movies = _read_dat_dataset(input_dir)
        source_format = "dat"
    else:
        raise FileNotFoundError(
            "Expected ratings.csv and movies.csv, or ratings.dat and movies.dat "
            f"in {input_dir}"
        )

    return normalize_ratings(ratings), normalize_movies(movies), source_format


def normalize_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the canonical ratings schema and coerce stable dtypes."""

    required = ["user_id", "movie_id", "rating", "timestamp"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"ratings: cannot normalize missing columns: {missing}")

    normalized = frame.loc[:, required].copy()
    normalized["user_id"] = pd.to_numeric(normalized["user_id"], errors="raise").astype(
        "int64"
    )
    normalized["movie_id"] = pd.to_numeric(normalized["movie_id"], errors="raise").astype(
        "int64"
    )
    normalized["rating"] = pd.to_numeric(normalized["rating"], errors="raise").astype(
        "float32"
    )
    normalized["timestamp"] = pd.to_numeric(
        normalized["timestamp"], errors="raise"
    ).astype("int64")
    return normalized


def normalize_movies(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the canonical movies schema and coerce stable dtypes."""

    required = ["movie_id", "title", "genres"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"movies: cannot normalize missing columns: {missing}")

    normalized = frame.loc[:, required].copy()
    normalized["movie_id"] = pd.to_numeric(
        normalized["movie_id"], errors="raise"
    ).astype("int64")
    normalized["title"] = normalized["title"].astype("string")
    normalized["genres"] = normalized["genres"].astype("string")
    return normalized


def ingest_movielens(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Read, validate and persist a MovieLens dataset."""

    ratings, movies, source_format = read_movielens(input_dir)
    ratings_summary = validate_ratings(ratings)
    movies_summary = validate_movies(movies)

    unknown_movie_count = int((~ratings["movie_id"].isin(movies["movie_id"])).sum())
    if unknown_movie_count:
        raise ValueError(
            f"ratings: {unknown_movie_count} rows reference a movie absent from movies"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(output_dir / "ratings.parquet", index=False)
    movies.to_parquet(output_dir / "movies.parquet", index=False)

    report: dict[str, object] = {
        "source_format": source_format,
        "ratings": {
            **asdict(ratings_summary),
            "user_count": int(ratings["user_id"].nunique()),
            "movie_count": int(ratings["movie_id"].nunique()),
            "minimum_timestamp": int(ratings["timestamp"].min()),
            "maximum_timestamp": int(ratings["timestamp"].max()),
            "mean_rating": round(float(ratings["rating"].mean()), 6),
        },
        "movies": {
            **asdict(movies_summary),
            "catalog_size": int(movies["movie_id"].nunique()),
        },
        "referential_integrity": {"unknown_movie_count": unknown_movie_count},
    }
    with (output_dir / "quality_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and convert a MovieLens dataset to Parquet."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = ingest_movielens(args.input_dir, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
