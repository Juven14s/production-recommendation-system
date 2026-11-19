"""Prometheus metrics for API, cache and ML inference behavior."""

from prometheus_client import Counter, Gauge, Histogram, Info

HTTP_REQUESTS = Counter(
    "recsys_http_requests_total",
    "HTTP requests handled by the recommendation service.",
    ["method", "endpoint", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "recsys_http_request_duration_seconds",
    "End-to-end HTTP request duration.",
    ["method", "endpoint"],
    buckets=(0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
)
HTTP_EXCEPTIONS = Counter(
    "recsys_http_exceptions_total",
    "Unhandled exceptions raised by endpoint and exception type.",
    ["endpoint", "exception_type"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "recsys_http_requests_in_progress",
    "Currently executing HTTP requests.",
    ["method"],
)
CACHE_EVENTS = Counter(
    "recsys_cache_events_total",
    "Recommendation cache hits and misses.",
    ["result", "backend"],
)
RECOMMENDATION_REQUESTS = Counter(
    "recsys_recommendation_requests_total",
    "Recommendation requests by personalization mode.",
    ["mode"],
)
RECOMMENDATIONS_RETURNED = Counter(
    "recsys_recommendations_returned_total",
    "Number of movie recommendations returned.",
    ["mode"],
)
ENGINE_STAGE_DURATION = Histogram(
    "recsys_engine_stage_duration_seconds",
    "Duration of individual online inference stages.",
    ["stage"],
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)
CANDIDATE_COUNT = Histogram(
    "recsys_candidate_count",
    "Number of candidates produced before ranking.",
    buckets=(10, 25, 50, 100, 200, 500, 1000),
)
RANKING_SCORE = Histogram(
    "recsys_ranking_score",
    "Distribution of final blended ranking scores.",
    buckets=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0),
)
MODEL_INFO = Info("recsys_model", "Active recommendation model bundle.")
FEATURE_LATEST_TIMESTAMP = Gauge(
    "recsys_feature_latest_timestamp_seconds",
    "Unix timestamp of the most recent interaction used by online features.",
)
FEATURE_AGE = Gauge(
    "recsys_feature_age_seconds",
    "Age of the most recent online feature interaction at service startup.",
)


def normalized_endpoint(path: str) -> str:
    """Avoid high-cardinality metric labels for user IDs."""

    if path.startswith("/recommendations/"):
        return "/recommendations/{user_id}"
    return path
