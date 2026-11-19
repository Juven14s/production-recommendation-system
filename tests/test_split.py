import pandas as pd

from mle_recsys.data.split import leave_two_out_split


def test_leave_two_out_split_uses_latest_events_without_leakage() -> None:
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2],
            "movie_id": [10, 20, 30, 40, 10, 20],
            "rating": [4.0, 5.0, 3.0, 4.5, 5.0, 4.0],
            "timestamp": [100, 200, 250, 300, 100, 200],
        }
    )

    split = leave_two_out_split(ratings, positive_threshold=4.0)

    assert split.train.loc[split.train["user_id"] == 1, "movie_id"].tolist() == [10]
    assert split.validation["movie_id"].tolist() == [20]
    assert split.test["movie_id"].tolist() == [40]
    assert set(split.train.loc[split.train["user_id"] == 2, "movie_id"]) == {10, 20}
    assert split.report["eligible_user_count"] == 1
