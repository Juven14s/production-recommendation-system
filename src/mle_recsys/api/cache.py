"""Recommendation cache backends."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import redis


class RecommendationCache(Protocol):
    """Minimal cache contract used by the API."""

    backend_name: str

    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None: ...


@dataclass
class _MemoryEntry:
    value: dict[str, Any]
    expires_at: float


class InMemoryTTLCache:
    """Thread-safe process-local fallback cache."""

    backend_name = "memory"

    def __init__(self) -> None:
        self._entries: dict[str, _MemoryEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                self._entries.pop(key, None)
                return None
            return dict(entry.value)

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        with self._lock:
            self._entries[key] = _MemoryEntry(
                value=dict(value), expires_at=time.monotonic() + ttl_seconds
            )


class RedisRecommendationCache:
    """Redis-backed JSON cache."""

    backend_name = "redis"

    def __init__(self, url: str) -> None:
        self._client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        self._client.ping()

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._client.get(key)
        return json.loads(value) if value is not None else None

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self._client.setex(key, ttl_seconds, json.dumps(value))


def build_cache(redis_url: str | None) -> RecommendationCache:
    """Use Redis when reachable, otherwise degrade to a local TTL cache."""

    if redis_url:
        try:
            return RedisRecommendationCache(redis_url)
        except redis.RedisError:
            pass
    return InMemoryTTLCache()
