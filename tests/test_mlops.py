from pathlib import Path

import pytest

from mle_recsys.mlops.gates import evaluate_promotion_gates
from mle_recsys.mlops.registry import LocalModelRegistry
from mle_recsys.mlops.validate_bundle import validate_bundle


def _passing_metrics() -> dict[str, float]:
    return {
        "recall@10": 0.08,
        "ndcg@10": 0.04,
        "catalog_coverage@10": 0.50,
        "candidate_recall@200": 0.55,
    }


def test_promotion_gates_reject_regression_against_champion() -> None:
    champion = _passing_metrics()
    candidate = {**champion, "recall@10": champion["recall@10"] * 0.95}

    result = evaluate_promotion_gates(candidate, champion)

    assert result.passed is False
    assert result.checks["relative:recall@10"]["passed"] is False


def test_registry_promote_and_rollback_are_auditable(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")
    for version in ["v1", "v2"]:
        (registry.bundles_dir / version).mkdir()
        registry.register(
            version,
            {
                "bundle_path": f"bundles/{version}",
                "status": "candidate",
                "metrics": _passing_metrics(),
                "mlflow_model_version": version[-1],
            },
        )

    registry.promote("v1")
    registry.promote("v2")
    rolled_back = registry.rollback()

    state = registry.load()
    assert rolled_back == "v1"
    assert state["champion"] == "v1"
    assert state["previous_champion"] == "v2"
    assert state["history"][-1]["action"] == "rollback"
    assert registry.champion_bundle() == (registry.bundles_dir / "v1").resolve()


def test_registry_rejects_rollback_without_history(tmp_path: Path) -> None:
    registry = LocalModelRegistry(tmp_path / "registry")

    with pytest.raises(ValueError, match="no previous champion"):
        registry.rollback()


def test_bundle_validator_rejects_incomplete_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required files"):
        validate_bundle(tmp_path / "empty-bundle")
