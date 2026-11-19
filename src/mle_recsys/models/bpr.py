"""Bayesian Personalized Ranking with PyTorch embeddings."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as functional

from mle_recsys.evaluation import evaluate_single_relevant_item
from mle_recsys.models.popularity import fit_popularity


class BPRModel(nn.Module):
    """User/item dot-product model optimized with the BPR objective."""

    def __init__(self, user_count: int, item_count: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(user_count, embedding_dim)
        self.item_embeddings = nn.Embedding(item_count, embedding_dim)
        self.item_bias = nn.Embedding(item_count, 1)
        nn.init.normal_(self.user_embeddings.weight, std=0.05)
        nn.init.normal_(self.item_embeddings.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def item_scores(self, users: torch.Tensor) -> torch.Tensor:
        """Score every known item for a batch of internal user indices."""

        user_vectors = self.user_embeddings(users)
        return user_vectors @ self.item_embeddings.weight.T + self.item_bias.weight.T

    def pairwise_scores(
        self,
        users: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score paired positive and negative examples."""

        user_vectors = self.user_embeddings(users)
        positive_vectors = self.item_embeddings(positive_items)
        negative_vectors = self.item_embeddings(negative_items)
        positive_scores = (
            (user_vectors * positive_vectors).sum(dim=1)
            + self.item_bias(positive_items).squeeze(1)
        )
        negative_scores = (
            (user_vectors * negative_vectors).sum(dim=1)
            + self.item_bias(negative_items).squeeze(1)
        )
        return positive_scores, negative_scores


@dataclass(frozen=True)
class BPRArtifacts:
    """Trained model plus stable mappings required for inference."""

    model: BPRModel
    user_ids: np.ndarray
    movie_ids: np.ndarray
    popularity_movie_ids: np.ndarray
    training_history: list[dict[str, float | int]]
    config: dict[str, float | int | str]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def _prepare_training_arrays(
    train: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_ids = np.sort(train["user_id"].unique()).astype(np.int64)
    movie_ids = np.sort(train["movie_id"].unique()).astype(np.int64)
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    movie_index = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    users = train["user_id"].map(user_index).to_numpy(dtype=np.int64)
    positive_items = train["movie_id"].map(movie_index).to_numpy(dtype=np.int64)

    seen = np.zeros((len(user_ids), len(movie_ids)), dtype=np.bool_)
    seen[users, positive_items] = True
    return user_ids, movie_ids, users, positive_items, seen


def _sample_negatives(
    users: np.ndarray,
    seen: np.ndarray,
    random_generator: np.random.Generator,
) -> np.ndarray:
    negatives = random_generator.integers(0, seen.shape[1], size=len(users), dtype=np.int64)
    conflicts = seen[users, negatives]
    while conflicts.any():
        negatives[conflicts] = random_generator.integers(
            0, seen.shape[1], size=int(conflicts.sum()), dtype=np.int64
        )
        conflicts = seen[users, negatives]
    return negatives


def fit_bpr(
    train: pd.DataFrame,
    embedding_dim: int = 64,
    epochs: int = 15,
    batch_size: int = 4096,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> BPRArtifacts:
    """Train deterministic BPR embeddings with uniform negative sampling."""

    if train.empty:
        raise ValueError("train must not be empty")
    if embedding_dim <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("embedding_dim, epochs and batch_size must be positive")
    if learning_rate <= 0 or regularization < 0:
        raise ValueError("learning_rate must be positive and regularization non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    resolved_device = _resolve_device(device)
    user_ids, movie_ids, users, positive_items, seen = _prepare_training_arrays(train)
    random_generator = np.random.default_rng(seed)

    model = BPRModel(len(user_ids), len(movie_ids), embedding_dim).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        permutation = random_generator.permutation(len(users))
        epoch_loss = 0.0
        example_count = 0
        model.train()

        for start in range(0, len(permutation), batch_size):
            selected = permutation[start : start + batch_size]
            batch_users = users[selected]
            batch_positives = positive_items[selected]
            batch_negatives = _sample_negatives(batch_users, seen, random_generator)

            user_tensor = torch.from_numpy(batch_users).to(resolved_device)
            positive_tensor = torch.from_numpy(batch_positives).to(resolved_device)
            negative_tensor = torch.from_numpy(batch_negatives).to(resolved_device)

            positive_scores, negative_scores = model.pairwise_scores(
                user_tensor, positive_tensor, negative_tensor
            )
            ranking_loss = -functional.logsigmoid(positive_scores - negative_scores).mean()
            regularization_loss = regularization * (
                model.user_embeddings(user_tensor).square().mean()
                + model.item_embeddings(positive_tensor).square().mean()
                + model.item_embeddings(negative_tensor).square().mean()
            )
            loss = ranking_loss + regularization_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = len(selected)
            epoch_loss += float(loss.detach().cpu()) * count
            example_count += count

        history.append({"epoch": epoch, "loss": epoch_loss / example_count})

    model = model.cpu().eval()
    popularity_movie_ids = fit_popularity(train)["movie_id"].to_numpy(dtype=np.int64)
    config: dict[str, float | int | str] = {
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "seed": seed,
        "training_device": str(resolved_device),
    }
    return BPRArtifacts(
        model=model,
        user_ids=user_ids,
        movie_ids=movie_ids,
        popularity_movie_ids=popularity_movie_ids,
        training_history=history,
        config=config,
    )


def recommend_bpr(
    artifacts: BPRArtifacts,
    train: pd.DataFrame,
    users: list[int],
    k: int,
    inference_batch_size: int = 512,
) -> pd.DataFrame:
    """Generate top-K unseen recommendations from learned embeddings."""

    if k <= 0 or inference_batch_size <= 0:
        raise ValueError("k and inference_batch_size must be strictly positive")

    user_to_index = {
        int(user_id): index for index, user_id in enumerate(artifacts.user_ids)
    }
    movie_to_index = {
        int(movie_id): index for index, movie_id in enumerate(artifacts.movie_ids)
    }
    history_by_user = train.groupby("user_id")["movie_id"].agg(set).to_dict()
    rows: list[dict[str, int | float]] = []

    for batch_start in range(0, len(users), inference_batch_size):
        batch_users = users[batch_start : batch_start + inference_batch_size]
        known_positions = [
            position for position, user_id in enumerate(batch_users) if user_id in user_to_index
        ]
        known_user_ids = [batch_users[position] for position in known_positions]

        ranked_by_user: dict[int, list[tuple[int, float]]] = {}
        if known_user_ids:
            internal_users = torch.tensor(
                [user_to_index[user_id] for user_id in known_user_ids], dtype=torch.long
            )
            with torch.no_grad():
                scores = artifacts.model.item_scores(internal_users)

            for row_index, user_id in enumerate(known_user_ids):
                seen = history_by_user.get(user_id, set())
                seen_indices = [
                    movie_to_index[int(movie_id)]
                    for movie_id in seen
                    if int(movie_id) in movie_to_index
                ]
                if seen_indices:
                    scores[row_index, seen_indices] = -torch.inf
                top_scores, top_indices = torch.topk(
                    scores[row_index], k=min(k, scores.shape[1] - len(seen_indices))
                )
                ranked_by_user[user_id] = [
                    (int(artifacts.movie_ids[index]), float(score))
                    for index, score in zip(
                        top_indices.cpu().tolist(), top_scores.cpu().tolist(), strict=True
                    )
                ]

        for user_id in batch_users:
            selected = ranked_by_user.get(user_id, [])
            selected_ids = {movie_id for movie_id, _ in selected}
            seen = history_by_user.get(user_id, set())
            if len(selected) < k:
                for movie_id_value in artifacts.popularity_movie_ids:
                    movie_id = int(movie_id_value)
                    if movie_id in seen or movie_id in selected_ids:
                        continue
                    selected.append((movie_id, 0.0))
                    selected_ids.add(movie_id)
                    if len(selected) == k:
                        break

            for rank, (movie_id, score) in enumerate(selected, start=1):
                rows.append(
                    {
                        "user_id": int(user_id),
                        "movie_id": movie_id,
                        "rank": rank,
                        "score": score,
                    }
                )

    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rank", "score"])


def save_bpr(artifacts: BPRArtifacts, output_dir: Path) -> None:
    """Persist a checkpoint and portable NumPy embeddings for retrieval."""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts.model.state_dict(), output_dir / "model_state.pt")
    np.save(output_dir / "user_ids.npy", artifacts.user_ids, allow_pickle=False)
    np.save(output_dir / "movie_ids.npy", artifacts.movie_ids, allow_pickle=False)
    np.save(
        output_dir / "user_embeddings.npy",
        artifacts.model.user_embeddings.weight.detach().numpy(),
        allow_pickle=False,
    )
    np.save(
        output_dir / "item_embeddings.npy",
        artifacts.model.item_embeddings.weight.detach().numpy(),
        allow_pickle=False,
    )
    np.save(
        output_dir / "item_bias.npy",
        artifacts.model.item_bias.weight.detach().numpy(),
        allow_pickle=False,
    )
    np.save(
        output_dir / "popularity_movie_ids.npy",
        artifacts.popularity_movie_ids,
        allow_pickle=False,
    )
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump(artifacts.training_history, stream, indent=2)
        stream.write("\n")

    metadata = {
        "model": "bpr_matrix_factorization",
        "user_count": int(len(artifacts.user_ids)),
        "item_count": int(len(artifacts.movie_ids)),
        "torch_version": torch.__version__,
        "final_training_loss": artifacts.training_history[-1]["loss"],
        **artifacts.config,
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_bpr_baseline(
    split_dir: Path,
    movies_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = (5, 10, 20),
    embedding_dim: int = 64,
    epochs: int = 15,
    batch_size: int = 4096,
    learning_rate: float = 0.01,
    regularization: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, object]:
    """Fit BPR and evaluate the learned retrieval scores on validation."""

    train = pd.read_parquet(split_dir / "train.parquet")
    validation = pd.read_parquet(split_dir / "validation.parquet")
    movies = pd.read_parquet(movies_path)
    artifacts = fit_bpr(
        train,
        embedding_dim=embedding_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        regularization=regularization,
        seed=seed,
        device=device,
    )
    users = sorted(int(user_id) for user_id in validation["user_id"].unique())
    recommendations = recommend_bpr(artifacts, train, users, max(k_values))
    metrics = evaluate_single_relevant_item(
        recommendations,
        validation.loc[:, ["user_id", "movie_id"]],
        catalog_size=int(movies["movie_id"].nunique()),
        k_values=k_values,
    )
    report: dict[str, object] = {
        "model": "bpr_matrix_factorization",
        "evaluation_split": "validation",
        **artifacts.config,
        **metrics,
    }

    save_bpr(artifacts, output_dir)
    recommendations.to_parquet(
        output_dir / "validation_recommendations.parquet", index=False
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate BPR embeddings.")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--movies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_bpr_baseline(
        split_dir=args.split_dir,
        movies_path=args.movies,
        output_dir=args.output_dir,
        k_values=tuple(args.k),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
