import pandas as pd
import pytest

from mle_recsys.evaluation import evaluate_single_relevant_item
from mle_recsys.models.popularity import fit_popularity, recommend_popular


def test_popularity_recommendations_exclude_seen_movies() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 3],
            "movie_id": [10, 20, 10, 10, 30],
            "rating": [5.0, 4.0, 4.5, 4.0, 5.0],
            "timestamp": [1, 2, 1, 1, 2],
        }
    )

    model = fit_popularity(train)
    recommendations = recommend_popular(model, train, users=[1], k=2)

    assert recommendations["movie_id"].tolist() == [30]
    assert not set(recommendations["movie_id"]).intersection({10, 20})


def test_top_k_metrics_use_rank_and_catalog_size() -> None:
    recommendations = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [10, 20, 30, 40],
            "rank": [1, 2, 1, 2],
            "score": [5.0, 4.0, 5.0, 4.0],
        }
    )
    targets = pd.DataFrame({"user_id": [1, 2], "movie_id": [20, 99]})

    metrics = evaluate_single_relevant_item(
        recommendations, targets, catalog_size=10, k_values=(1, 2)
    )

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 0.5
    assert metrics["ndcg@2"] == pytest.approx(0.3154648768)
    assert metrics["catalog_coverage@2"] == 0.4
