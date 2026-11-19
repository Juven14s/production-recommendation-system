import pandas as pd

from mle_recsys.models.bpr import fit_bpr, recommend_bpr


def _small_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "movie_id": [10, 20, 10, 30, 20, 40, 30, 40],
            "rating": [5.0, 4.0, 5.0, 4.0, 4.5, 5.0, 4.0, 5.0],
            "timestamp": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )


def test_bpr_training_is_reproducible_and_recommendations_exclude_seen() -> None:
    train = _small_training_frame()
    first = fit_bpr(
        train, embedding_dim=4, epochs=2, batch_size=4, seed=7, device="cpu"
    )
    second = fit_bpr(
        train, embedding_dim=4, epochs=2, batch_size=4, seed=7, device="cpu"
    )

    recommendations = recommend_bpr(first, train, users=[1], k=2)

    assert first.training_history == second.training_history
    assert len(first.training_history) == 2
    assert not set(recommendations["movie_id"]).intersection({10, 20})


def test_bpr_unknown_user_receives_popularity_fallback() -> None:
    train = _small_training_frame()
    artifacts = fit_bpr(
        train, embedding_dim=4, epochs=1, batch_size=4, seed=7, device="cpu"
    )

    recommendations = recommend_bpr(artifacts, train, users=[999], k=2)

    assert recommendations["movie_id"].tolist() == [10, 40]
    assert recommendations["score"].tolist() == [0.0, 0.0]
