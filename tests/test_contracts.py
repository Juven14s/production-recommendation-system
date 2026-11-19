import pandas as pd
import pytest

from mle_recsys.data.contracts import DataValidationError, validate_movies, validate_ratings


def test_validate_ratings_rejects_out_of_range_value() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1],
            "movie_id": [10],
            "rating": [6.0],
            "timestamp": [1_000_000_000],
        }
    )

    with pytest.raises(DataValidationError, match="between 0.5 and 5.0"):
        validate_ratings(ratings)


def test_validate_movies_rejects_duplicate_ids() -> None:
    movies = pd.DataFrame(
        {
            "movie_id": [10, 10],
            "title": ["First", "Second"],
            "genres": ["Drama", "Comedy"],
        }
    )

    with pytest.raises(DataValidationError, match="must be unique"):
        validate_movies(movies)
