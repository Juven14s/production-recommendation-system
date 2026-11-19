import numpy as np
import pandas as pd

from mle_recsys.retrieval.faiss_index import (
    FaissArtifacts,
    augment_item_vectors,
    augment_user_vectors,
    build_flat_ip_index,
    evaluate_candidate_retrieval,
    retrieve_candidates,
)


def test_augmented_inner_product_reproduces_bpr_score() -> None:
    items = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    biases = np.array([[0.2], [0.8]], dtype=np.float32)
    user = np.array([[0.5, 0.25]], dtype=np.float32)

    expected = user @ items.T + biases.T
    actual = augment_user_vectors(user) @ augment_item_vectors(items, biases).T

    np.testing.assert_allclose(actual, expected)


def test_faiss_retrieval_excludes_seen_and_uses_bias() -> None:
    item_embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32
    )
    item_bias = np.array([[0.0], [0.8], [0.0]], dtype=np.float32)
    artifacts = FaissArtifacts(
        index=build_flat_ip_index(item_embeddings, item_bias),
        movie_ids=np.array([10, 20, 30]),
        user_ids=np.array([1]),
        user_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        popularity_movie_ids=np.array([10, 20, 30]),
    )
    train = pd.DataFrame({"user_id": [1], "movie_id": [10]})

    candidates = retrieve_candidates(artifacts, train, users=[1], k=2)

    assert candidates["movie_id"].tolist() == [20, 30]
    assert 10 not in candidates["movie_id"].tolist()


def test_candidate_recall_measures_target_survival() -> None:
    candidates = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [10, 20, 30, 40],
            "rank": [1, 2, 1, 2],
        }
    )
    targets = pd.DataFrame({"user_id": [1, 2], "movie_id": [20, 99]})

    metrics = evaluate_candidate_retrieval(candidates, targets, 10, (1, 2))

    assert metrics["candidate_recall@1"] == 0.0
    assert metrics["candidate_recall@2"] == 0.5
    assert metrics["candidate_coverage@2"] == 0.4
