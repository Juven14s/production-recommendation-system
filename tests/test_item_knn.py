import pandas as pd

from mle_recsys.models.item_knn import fit_item_knn, recommend_item_knn


def test_item_knn_recommends_cooccurring_unseen_movie() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4],
            "movie_id": [10, 20, 10, 30, 10, 30, 40],
            "rating": [5.0, 4.0, 5.0, 4.0, 4.5, 5.0, 4.0],
            "timestamp": [1, 2, 1, 2, 1, 2, 1],
        }
    )

    model = fit_item_knn(train, neighbors_per_item=10, shrinkage=0.0)
    recommendations = recommend_item_knn(model, train, users=[1], k=2)

    assert recommendations.iloc[0]["movie_id"] == 30
    assert not set(recommendations["movie_id"]).intersection({10, 20})
    assert model.similarities.shape == (4, 4)
    assert model.similarities.diagonal().sum() == 0


def test_item_knn_uses_popularity_fallback_for_unknown_user() -> None:
    train = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "movie_id": [10, 10, 20],
            "rating": [5.0, 4.0, 5.0],
            "timestamp": [1, 1, 1],
        }
    )

    model = fit_item_knn(train)
    recommendations = recommend_item_knn(model, train, users=[999], k=2)

    assert recommendations["movie_id"].tolist() == [10, 20]
    assert recommendations["score"].tolist() == [0.0, 0.0]
