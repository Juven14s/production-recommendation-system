"""Data contracts for normalized MovieLens tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input table violates the MovieLens data contract."""


@dataclass(frozen=True)
class ValidationSummary:
    """Small, serializable summary produced after validation."""

    row_count: int
    null_count: int
    duplicate_count: int


RATINGS_COLUMNS = ("user_id", "movie_id", "rating", "timestamp")
MOVIES_COLUMNS = ("movie_id", "title", "genres")


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], table: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataValidationError(f"{table}: missing required columns: {missing}")


def _require_no_nulls(frame: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    nulls = frame.loc[:, list(columns)].isna().sum()
    invalid = {column: int(count) for column, count in nulls.items() if count > 0}
    if invalid:
        raise DataValidationError(f"{table}: null values found: {invalid}")


def validate_ratings(frame: pd.DataFrame) -> ValidationSummary:
    """Validate the normalized ratings table."""

    _require_columns(frame, RATINGS_COLUMNS, "ratings")
    _require_no_nulls(frame, RATINGS_COLUMNS, "ratings")

    if frame.empty:
        raise DataValidationError("ratings: table is empty")
    if (frame["user_id"] <= 0).any():
        raise DataValidationError("ratings: user_id must be strictly positive")
    if (frame["movie_id"] <= 0).any():
        raise DataValidationError("ratings: movie_id must be strictly positive")
    if not frame["rating"].between(0.5, 5.0, inclusive="both").all():
        raise DataValidationError("ratings: rating must be between 0.5 and 5.0")
    if (frame["timestamp"] <= 0).any():
        raise DataValidationError("ratings: timestamp must be strictly positive")

    duplicate_count = int(frame.duplicated().sum())
    return ValidationSummary(
        row_count=len(frame),
        null_count=int(frame.loc[:, list(RATINGS_COLUMNS)].isna().sum().sum()),
        duplicate_count=duplicate_count,
    )


def validate_movies(frame: pd.DataFrame) -> ValidationSummary:
    """Validate the normalized movies table."""

    _require_columns(frame, MOVIES_COLUMNS, "movies")
    _require_no_nulls(frame, MOVIES_COLUMNS, "movies")

    if frame.empty:
        raise DataValidationError("movies: table is empty")
    if (frame["movie_id"] <= 0).any():
        raise DataValidationError("movies: movie_id must be strictly positive")
    if (frame["title"].str.strip() == "").any():
        raise DataValidationError("movies: title must not be empty")
    if frame["movie_id"].duplicated().any():
        raise DataValidationError("movies: movie_id must be unique")

    duplicate_count = int(frame.duplicated().sum())
    return ValidationSummary(
        row_count=len(frame),
        null_count=int(frame.loc[:, list(MOVIES_COLUMNS)].isna().sum().sum()),
        duplicate_count=duplicate_count,
    )
