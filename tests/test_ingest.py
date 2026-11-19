import json
from pathlib import Path

import pandas as pd
import pytest

from mle_recsys.data.ingest import ingest_movielens


@pytest.fixture(params=["csv", "dat"])
def raw_dataset(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    if request.param == "csv":
        (raw_dir / "ratings.csv").write_text(
            "userId,movieId,rating,timestamp\n1,10,4.0,1000000000\n2,20,3.5,1000000100\n",
            encoding="utf-8",
        )
        (raw_dir / "movies.csv").write_text(
            "movieId,title,genres\n10,First Movie,Drama\n20,Second Movie,Comedy\n",
            encoding="utf-8",
        )
    else:
        (raw_dir / "ratings.dat").write_text(
            "1::10::4::1000000000\n2::20::3.5::1000000100\n",
            encoding="latin-1",
        )
        (raw_dir / "movies.dat").write_text(
            "10::First Movie::Drama\n20::Second Movie::Comedy\n",
            encoding="latin-1",
        )
    return raw_dir


def test_ingest_writes_valid_parquet_and_report(raw_dataset: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"

    report = ingest_movielens(raw_dataset, output_dir)

    ratings = pd.read_parquet(output_dir / "ratings.parquet")
    movies = pd.read_parquet(output_dir / "movies.parquet")
    saved_report = json.loads((output_dir / "quality_report.json").read_text("utf-8"))

    assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
    assert list(movies.columns) == ["movie_id", "title", "genres"]
    assert ratings["user_id"].dtype == "int64"
    assert ratings["rating"].dtype == "float32"
    assert report["ratings"]["row_count"] == 2
    assert report["movies"]["catalog_size"] == 2
    assert saved_report == report


def test_ingest_rejects_unknown_movie(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n1,999,4.0,1000000000\n",
        encoding="utf-8",
    )
    (raw_dir / "movies.csv").write_text(
        "movieId,title,genres\n10,First Movie,Drama\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from movies"):
        ingest_movielens(raw_dir, tmp_path / "processed")
