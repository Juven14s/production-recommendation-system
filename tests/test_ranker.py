import numpy as np
import pandas as pd

from mle_recsys.ranking.lightgbm_ranker import (
    FEATURE_COLUMNS,
    EmbeddingLookup,
    build_candidate_features,
    make_ranker_training_split,
    select_retrieved_training_groups,
)


def test_ranker_training_split_holds_out_latest_event() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "movie_id": [10, 20, 30, 40],
            "rating": [4.0, 5.0, 4.5, 5.0],
            "timestamp": [100, 300, 200, 100],
        }
    )

    split = make_ranker_training_split(train)

    assert split.targets[["user_id", "movie_id"]].values.tolist() == [[1, 20]]
    assert set(split.history.loc[split.history["user_id"] == 1, "movie_id"]) == {10, 30}
    assert split.history.loc[split.history["user_id"] == 1, "timestamp"].max() < 300


def test_candidate_features_are_complete_and_label_target() -> None:
    history = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "movie_id": [10, 20, 10],
            "rating": [5.0, 4.0, 4.5],
            "timestamp": [100, 200, 150],
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 30],
            "title": ["A (1990)", "B (2000)", "C (1995)"],
            "genres": ["Drama", "Comedy|Drama", "Comedy"],
        }
    )
    candidates = pd.DataFrame(
        {
            "user_id": [1, 1],
            "movie_id": [20, 30],
            "rank": [1, 2],
            "retrieval_score": [0.8, 0.4],
            "source": ["faiss", "faiss"],
        }
    )
    lookup = EmbeddingLookup(
        user_ids=np.array([1, 2]),
        movie_ids=np.array([10, 20, 30]),
        user_embeddings=np.ones((2, 2), dtype=np.float32),
        item_embeddings=np.ones((3, 2), dtype=np.float32),
        item_bias=np.zeros((3, 1), dtype=np.float32),
        popularity_movie_ids=np.array([10, 20, 30]),
    )
    targets = pd.DataFrame({"user_id": [1], "movie_id": [30]})

    features = build_candidate_features(candidates, history, movies, lookup, targets)

    assert not features[FEATURE_COLUMNS].isna().any().any()
    assert features["label"].tolist() == [0, 1]
    assert features.loc[1, "genre_affinity"] > 0


def test_ranker_training_keeps_only_naturally_retrieved_positives() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [10, 20, 30, 40],
            "rank": [1, 2, 1, 2],
        }
    )
    targets = pd.DataFrame({"user_id": [1, 2], "movie_id": [20, 99]})
    lookup = EmbeddingLookup(
        user_ids=np.array([1, 2]),
        movie_ids=np.array([10, 20, 30, 40]),
        user_embeddings=np.ones((2, 2), dtype=np.float32),
        item_embeddings=np.ones((4, 2), dtype=np.float32),
        item_bias=np.zeros((4, 1), dtype=np.float32),
        popularity_movie_ids=np.array([10, 20, 30, 40]),
    )

    selected, report = select_retrieved_training_groups(candidates, targets, lookup)

    assert selected["user_id"].unique().tolist() == [1]
    assert report["retrieved_positive_group_count"] == 1
    assert report["cold_target_count"] == 1
