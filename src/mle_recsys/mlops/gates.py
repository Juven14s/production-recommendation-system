"""Quality gates for automatic model promotion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: dict[str, dict[str, float | bool | str]]


ABSOLUTE_THRESHOLDS = {
    "recall@10": 0.07,
    "ndcg@10": 0.035,
    "catalog_coverage@10": 0.40,
    "candidate_recall@200": 0.50,
}

RELATIVE_TOLERANCES = {
    "recall@10": 0.99,
    "ndcg@10": 0.99,
    "catalog_coverage@10": 0.90,
    "candidate_recall@200": 0.99,
}


def evaluate_promotion_gates(
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float] | None = None,
) -> GateResult:
    """Require absolute quality and prevent material champion regressions."""

    checks: dict[str, dict[str, float | bool | str]] = {}
    passed = True
    for metric, threshold in ABSOLUTE_THRESHOLDS.items():
        value = float(candidate_metrics.get(metric, float("-inf")))
        check_passed = value >= threshold
        checks[f"absolute:{metric}"] = {
            "kind": "absolute_minimum",
            "value": value,
            "threshold": threshold,
            "passed": check_passed,
        }
        passed = passed and check_passed

    if champion_metrics:
        for metric, tolerance in RELATIVE_TOLERANCES.items():
            champion_value = float(champion_metrics.get(metric, 0.0))
            threshold = champion_value * tolerance
            value = float(candidate_metrics.get(metric, float("-inf")))
            check_passed = value >= threshold
            checks[f"relative:{metric}"] = {
                "kind": "relative_to_champion",
                "value": value,
                "threshold": threshold,
                "champion_value": champion_value,
                "passed": check_passed,
            }
            passed = passed and check_passed
    return GateResult(passed=passed, checks=checks)
