from typing import Any

from fastapi.testclient import TestClient

from mle_recsys.api.app import ServiceSettings, create_app
from mle_recsys.api.cache import InMemoryTTLCache


class FakeEngine:
    model_version = "test-version"
    user_count = 10
    item_count = 20

    def __init__(self) -> None:
        self.call_count = 0

    def recommend(self, user_id: int, k: int) -> dict[str, Any]:
        self.call_count += 1
        return {
            "cold_start": user_id > 10,
            "recommendations": [
                {
                    "movie_id": rank,
                    "title": f"Movie {rank}",
                    "genres": "Drama",
                    "score": 1.0 / rank,
                    "rank": rank,
                    "source": "fake",
                }
                for rank in range(1, k + 1)
            ],
        }


def test_health_and_recommendation_cache() -> None:
    engine = FakeEngine()
    app = create_app(
        settings=ServiceSettings(cache_ttl_seconds=60),
        engine=engine,  # type: ignore[arg-type]
        cache=InMemoryTTLCache(),
    )

    with TestClient(app) as client:
        health = client.get("/health")
        first = client.get("/recommendations/1?k=3")
        second = client.get("/recommendations/1?k=3")
        metrics = client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["cache_backend"] == "memory"
    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert len(first.json()["recommendations"]) == 3
    assert second.json()["cached"] is True
    assert engine.call_count == 1
    assert metrics.status_code == 200
    assert "recsys_http_requests_total" in metrics.text
    assert "recsys_cache_events_total" in metrics.text


def test_api_validates_inputs_and_marks_cold_start() -> None:
    app = create_app(
        engine=FakeEngine(),  # type: ignore[arg-type]
        cache=InMemoryTTLCache(),
    )

    with TestClient(app) as client:
        cold_start = client.get("/recommendations/999?k=2")
        invalid_user = client.get("/recommendations/0")
        invalid_k = client.get("/recommendations/1?k=51")

    assert cold_start.status_code == 200
    assert cold_start.json()["cold_start"] is True
    assert invalid_user.status_code == 422
    assert invalid_k.status_code == 422
