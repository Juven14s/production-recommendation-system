"""Inspect and roll back the deployment registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from mle_recsys.mlops.registry import LocalModelRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local model deployment aliases.")
    parser.add_argument(
        "--registry-root", type=Path, default=Path("artifacts/model_registry")
    )
    parser.add_argument(
        "--tracking-uri", default="sqlite:///artifacts/mlflow/mlflow.db"
    )
    parser.add_argument("--registered-model-name", default="movie-recommender")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--version")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = LocalModelRegistry(args.registry_root)
    if args.command == "status":
        print(json.dumps(registry.load(), indent=2, sort_keys=True))
        return

    target = registry.rollback(args.version)
    state = registry.load()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    mlflow_version = state["versions"][target]["mlflow_model_version"]
    MlflowClient().set_registered_model_alias(
        args.registered_model_name, "champion", mlflow_version
    )
    print(json.dumps({"rolled_back_to": target}, indent=2))
