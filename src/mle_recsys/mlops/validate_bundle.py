"""Validate the integrity and inference behavior of a deployable bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mle_recsys.api.engine import EnginePaths, RecommendationEngine

REQUIRED_FILES = (
    "manifest.json",
    "data/train.parquet",
    "data/movies.parquet",
    "bpr/user_ids.npy",
    "bpr/movie_ids.npy",
    "bpr/user_embeddings.npy",
    "bpr/item_embeddings.npy",
    "bpr/popularity_movie_ids.npy",
    "faiss/items.index",
    "faiss/movie_ids.npy",
    "ranker/ranker.txt",
    "ranker/metrics.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(bundle: Path) -> dict[str, Any]:
    """Fail closed unless the bundle is complete, approved and serves predictions."""

    bundle = bundle.resolve()
    missing = [relative for relative in REQUIRED_FILES if not (bundle / relative).is_file()]
    if missing:
        raise ValueError(f"bundle is missing required files: {', '.join(missing)}")

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported bundle manifest schema")
    if not manifest.get("promotion", {}).get("passed", False):
        raise ValueError("bundle did not pass its promotion gates")

    fingerprint_paths = {
        "train.parquet": bundle / "data" / "train.parquet",
        "movies.parquet": bundle / "data" / "movies.parquet",
    }
    expected_fingerprints = manifest.get("dataset_fingerprints", {})
    for name, path in fingerprint_paths.items():
        expected = expected_fingerprints.get(name)
        if not expected or _sha256(path) != expected:
            raise ValueError(f"dataset fingerprint mismatch: {name}")

    paths = EnginePaths(
        train=bundle / "data" / "train.parquet",
        movies=bundle / "data" / "movies.parquet",
        bpr_dir=bundle / "bpr",
        faiss_dir=bundle / "faiss",
        ranker_dir=bundle / "ranker",
    )
    engine = RecommendationEngine(paths)
    train = pd.read_parquet(paths.train, columns=["user_id", "movie_id"])
    known_user = int(train.iloc[0]["user_id"])
    cold_user = int(train["user_id"].max()) + 1_000_000
    personalized = engine.recommend(known_user, 5)
    cold_start = engine.recommend(cold_user, 3)

    if personalized["cold_start"] or len(personalized["recommendations"]) != 5:
        raise ValueError("personalized inference smoke test failed")
    if not cold_start["cold_start"] or len(cold_start["recommendations"]) != 3:
        raise ValueError("cold-start inference smoke test failed")
    seen = set(train.loc[train["user_id"] == known_user, "movie_id"].astype(int))
    recommended = {
        int(item["movie_id"]) for item in personalized["recommendations"]
    }
    if seen & recommended:
        raise ValueError("personalized recommendations contain already seen items")

    return {
        "valid": True,
        "version": str(manifest["version"]),
        "model_version": engine.model_version,
        "known_user_smoke_test": known_user,
        "cold_start_smoke_test": cold_user,
        "user_count": engine.user_count,
        "item_count": engine.item_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a deployable model bundle.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.bundle), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
