"""MLflow pyfunc wrapper for a complete recommendation bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from mle_recsys.api.engine import EnginePaths, RecommendationEngine


class RecommendationPyfuncModel(mlflow.pyfunc.PythonModel):
    """Portable MLflow flavor that loads the production inference engine."""

    def load_context(self, context: Any) -> None:
        self.engine = RecommendationEngine(
            EnginePaths(
                train=Path(context.artifacts["train"]),
                movies=Path(context.artifacts["movies"]),
                bpr_dir=Path(context.artifacts["bpr"]),
                faiss_dir=Path(context.artifacts["faiss"]),
                ranker_dir=Path(context.artifacts["ranker"]),
            )
        )

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        results = []
        for row in model_input.itertuples(index=False):
            user_id = int(row.user_id)
            k = int(getattr(row, "k", 10))
            prediction = self.engine.recommend(user_id, k)
            results.append(
                {
                    "user_id": user_id,
                    "cold_start": prediction["cold_start"],
                    "recommendations_json": json.dumps(
                        prediction["recommendations"], sort_keys=True
                    ),
                }
            )
        return pd.DataFrame(results)
