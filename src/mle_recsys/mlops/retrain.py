"""End-to-end versioned retraining, evaluation and promotion pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from mle_recsys.mlops.gates import evaluate_promotion_gates
from mle_recsys.mlops.mlflow_model import RecommendationPyfuncModel
from mle_recsys.mlops.registry import LocalModelRegistry, utc_now
from mle_recsys.models.bpr import fit_bpr, save_bpr
from mle_recsys.ranking.lightgbm_ranker import run_lightgbm_ranker
from mle_recsys.retrieval.faiss_index import run_faiss_retrieval


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_version() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(3)}"


def _champion_metrics(registry: LocalModelRegistry) -> dict[str, float] | None:
    state = registry.load()
    champion = state["champion"]
    if champion is None:
        return None
    return {
        key: float(value)
        for key, value in state["versions"][champion]["metrics"].items()
    }


def _combined_metrics(
    ranking_metrics: dict[str, Any], retrieval_metrics: dict[str, Any]
) -> dict[str, float]:
    keys = [
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "catalog_coverage@5",
        "catalog_coverage@10",
        "catalog_coverage@20",
    ]
    metrics = {key: float(ranking_metrics[key]) for key in keys}
    for key in [
        "candidate_recall@50",
        "candidate_recall@100",
        "candidate_recall@200",
        "latency_p95_ms",
    ]:
        metrics[key] = float(retrieval_metrics[key])
    return metrics


def _register_mlflow_model(
    run_id: str,
    bundle: Path,
    model_name: str,
) -> str:
    model_info = mlflow.pyfunc.log_model(
        name="recommendation_model",
        python_model=RecommendationPyfuncModel(),
        artifacts={
            "train": str(bundle / "data" / "train.parquet"),
            "movies": str(bundle / "data" / "movies.parquet"),
            "bpr": str(bundle / "bpr"),
            "faiss": str(bundle / "faiss"),
            "ranker": str(bundle / "ranker"),
        },
        input_example=pd.DataFrame({"user_id": [1], "k": [10]}),
    )
    registered = mlflow.register_model(model_info.model_uri, model_name)
    client = MlflowClient()
    client.set_model_version_tag(
        model_name, registered.version, "source_run_id", run_id
    )
    return str(registered.version)


def run_retraining_pipeline(
    split_dir: Path,
    movies_path: Path,
    registry_root: Path,
    tracking_uri: str,
    experiment_name: str = "production-recommendation-system",
    registered_model_name: str = "movie-recommender",
    bpr_epochs: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a candidate bundle, evaluate gates and promote when safe."""

    registry = LocalModelRegistry(registry_root)
    version = _new_version()
    bundle = registry.bundles_dir / version
    bundle.mkdir(parents=True, exist_ok=False)
    data_dir = bundle / "data"
    data_dir.mkdir()
    shutil.copy2(split_dir / "train.parquet", data_dir / "train.parquet")
    shutil.copy2(movies_path, data_dir / "movies.parquet")

    if tracking_uri.startswith("sqlite:///"):
        database_path = Path(tracking_uri.removeprefix("sqlite:///"))
        database_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = (registry_root.parent / "mlflow" / "artifacts").resolve()
        artifact_location.mkdir(parents=True, exist_ok=True)
        client.create_experiment(
            experiment_name, artifact_location=artifact_location.as_uri()
        )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"retrain-{version}") as active_run:
        run_id = active_run.info.run_id
        mlflow.set_tags(
            {
                "pipeline": "full_retraining",
                "bundle_version": version,
                "dataset": "movielens-1m",
            }
        )
        mlflow.log_params(
            {
                "bpr_embedding_dim": 64,
                "bpr_epochs": bpr_epochs,
                "bpr_batch_size": 4096,
                "ranker_candidate_count": 200,
                "ranker_estimators": 300,
                "seed": seed,
            }
        )

        train = pd.read_parquet(data_dir / "train.parquet")
        bpr = fit_bpr(
            train,
            embedding_dim=64,
            epochs=bpr_epochs,
            batch_size=4096,
            learning_rate=0.01,
            regularization=1e-4,
            seed=seed,
            device="cpu",
        )
        save_bpr(bpr, bundle / "bpr")

        retrieval_report = run_faiss_retrieval(
            bpr_dir=bundle / "bpr",
            split_dir=split_dir,
            movies_path=movies_path,
            output_dir=bundle / "faiss",
            k_values=(50, 100, 200),
        )
        ranking_report = run_lightgbm_ranker(
            split_dir=split_dir,
            movies_path=movies_path,
            bpr_dir=bundle / "bpr",
            validation_candidates_path=bundle
            / "faiss"
            / "validation_candidates.parquet",
            output_dir=bundle / "ranker",
            candidate_count=200,
            k_values=(5, 10, 20),
            internal_bpr_epochs=10,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            seed=seed,
        )
        metrics = _combined_metrics(ranking_report, retrieval_report)
        gates = evaluate_promotion_gates(metrics, _champion_metrics(registry))
        mlflow.log_metrics(
            {metric.replace("@", "_at_"): value for metric, value in metrics.items()}
        )
        mlflow.log_metrics(
            {"promotion_gate_passed": 1.0 if gates.passed else 0.0}
        )

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "version": version,
            "created_at": utc_now(),
            "run_id": run_id,
            "dataset_fingerprints": {
                "train.parquet": _sha256(data_dir / "train.parquet"),
                "movies.parquet": _sha256(data_dir / "movies.parquet"),
            },
            "metrics": metrics,
            "promotion": {"passed": gates.passed, "checks": gates.checks},
            "parameters": {
                "bpr_epochs": bpr_epochs,
                "seed": seed,
                "candidate_count": 200,
            },
        }
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(manifest_path), artifact_path="governance")
        registered_version = _register_mlflow_model(
            run_id, bundle, registered_model_name
        )

        client = MlflowClient()
        validation_status = "approved" if gates.passed else "rejected"
        client.set_model_version_tag(
            registered_model_name,
            registered_version,
            "validation_status",
            validation_status,
        )
        metadata = {
            "bundle_path": (Path("bundles") / version).as_posix(),
            "created_at": manifest["created_at"],
            "run_id": run_id,
            "mlflow_model_version": registered_version,
            "status": "candidate",
            "metrics": metrics,
            "promotion_checks": gates.checks,
        }
        registry.register(version, metadata)
        if gates.passed:
            registry.promote(version)
            client.set_registered_model_alias(
                registered_model_name, "champion", registered_version
            )

        result = {
            "version": version,
            "run_id": run_id,
            "mlflow_model_version": registered_version,
            "promoted": gates.passed,
            "metrics": metrics,
            "checks": gates.checks,
            "bundle": str(bundle),
        }
        (bundle / "pipeline_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain, track, register and conditionally promote the recommender."
    )
    parser.add_argument(
        "--split-dir", type=Path, default=Path("data/processed/ml-1m/splits")
    )
    parser.add_argument(
        "--movies",
        type=Path,
        default=Path("data/processed/ml-1m/movies.parquet"),
    )
    parser.add_argument(
        "--registry-root", type=Path, default=Path("artifacts/model_registry")
    )
    parser.add_argument(
        "--tracking-uri", default="sqlite:///artifacts/mlflow/mlflow.db"
    )
    parser.add_argument("--experiment-name", default="production-recommendation-system")
    parser.add_argument("--registered-model-name", default="movie-recommender")
    parser.add_argument("--bpr-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_retraining_pipeline(
        split_dir=args.split_dir,
        movies_path=args.movies,
        registry_root=args.registry_root,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        bpr_epochs=args.bpr_epochs,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
