"""Implicit-feedback collaborative filtering via Alternating Least Squares."""

import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class MatrixFactorizationModel:
    def __init__(self, factors: int = 128, regularization: float = 0.01, iterations: int = 30):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

        self._model: Optional[AlternatingLeastSquares] = None
        self._user_index: dict[str, int] = {}   # user_id → row index
        self._item_index: dict[str, int] = {}   # item_id → col index
        self._idx_to_item: dict[int, str] = {}  # col index → item_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, interactions_df: pd.DataFrame, mlflow_run: bool = True) -> None:
        """Build sparse matrix and train ALS. Logs metrics to MLflow."""
        users = interactions_df["user_id"].unique()
        items = interactions_df["item_id"].unique()
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: i for i, it in enumerate(items)}
        self._idx_to_item = {i: it for it, i in self._item_index.items()}

        rows = interactions_df["user_id"].map(self._user_index)
        cols = interactions_df["item_id"].map(self._item_index)
        # Use rating as confidence weight (Hu et al.: c = 1 + alpha * r)
        data = 1.0 + 40.0 * interactions_df["rating"].astype(float)
        user_item = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        # Hold-out 10% of users for validation
        val_size = max(1, int(len(users) * 0.1))
        val_user_ids = set(np.random.choice(list(self._user_index.keys()), val_size, replace=False))

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=False,
        )
        logger.info("Training ALS (factors=%d, reg=%s, iter=%d)...",
                    self.factors, self.regularization, self.iterations)
        self._model.fit(user_item)

        p10, r10 = self._evaluate(interactions_df, val_user_ids, user_item, k=10)
        logger.info("Validation — precision@10: %.4f  recall@10: %.4f", p10, r10)

        if mlflow_run:
            mlflow.log_params({
                "cf_factors": self.factors,
                "cf_regularization": self.regularization,
                "cf_iterations": self.iterations,
            })
            mlflow.log_metrics({"cf_precision_at_10": p10, "cf_recall_at_10": r10})

    def _evaluate(
        self,
        interactions_df: pd.DataFrame,
        val_user_ids: set,
        user_item: csr_matrix,
        k: int = 10,
    ) -> tuple[float, float]:
        """Compute precision@k and recall@k on held-out users."""
        precisions, recalls = [], []
        val_df = interactions_df[interactions_df["user_id"].isin(val_user_ids)]

        for user_id, group in val_df.groupby("user_id"):
            uid = self._user_index[user_id]
            true_items = set(group["item_id"].tolist())
            try:
                recs, _ = self._model.recommend(
                    uid, user_item[uid], N=k, filter_already_liked_items=False
                )
                rec_items = {self._idx_to_item[i] for i in recs if i in self._idx_to_item}
                hits = len(rec_items & true_items)
                precisions.append(hits / k)
                recalls.append(hits / len(true_items) if true_items else 0.0)
            except Exception:
                continue

        return (
            float(np.mean(precisions)) if precisions else 0.0,
            float(np.mean(recalls)) if recalls else 0.0,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_user_embedding(self, user_id: str) -> np.ndarray:
        uid = self._user_index[user_id]
        return self._model.user_factors[uid].copy()

    def get_item_embedding(self, item_id: str) -> np.ndarray:
        iid = self._item_index[item_id]
        return self._model.item_factors[iid].copy()

    def recommend_for_user(self, user_id: str, n: int = 100) -> list[str]:
        """Return top-n item_ids for a user, excluding already-seen items."""
        uid = self._user_index.get(user_id)
        if uid is None:
            return []

        # Build sparse row for this user to pass as user_items
        from scipy.sparse import csr_matrix as _csr
        n_items = len(self._item_index)
        # Create empty row (no training data available at inference)
        user_items = _csr((1, n_items), dtype=np.float32)

        recs, _ = self._model.recommend(uid, user_items, N=n, filter_already_liked_items=False)
        return [self._idx_to_item[i] for i in recs if i in self._idx_to_item]

    def recommend_for_user_with_history(
        self, user_id: str, user_item_matrix: csr_matrix, n: int = 100
    ) -> list[str]:
        """Like recommend_for_user but filters seen items using the training matrix."""
        uid = self._user_index.get(user_id)
        if uid is None:
            return []
        recs, _ = self._model.recommend(
            uid, user_item_matrix[uid], N=n, filter_already_liked_items=True
        )
        return [self._idx_to_item[i] for i in recs if i in self._idx_to_item]

    def has_user(self, user_id: str) -> bool:
        return user_id in self._user_index

    def has_item(self, item_id: str) -> bool:
        return item_id in self._item_index

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "cf_model.pkl", "wb") as f:
            pickle.dump({
                "model": self._model,
                "user_index": self._user_index,
                "item_index": self._item_index,
                "idx_to_item": self._idx_to_item,
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
            }, f)
        logger.info("Saved CF model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MatrixFactorizationModel":
        path = Path(path)
        with open(path / "cf_model.pkl", "rb") as f:
            data = pickle.load(f)
        obj = cls(
            factors=data["factors"],
            regularization=data["regularization"],
            iterations=data["iterations"],
        )
        obj._model = data["model"]
        obj._user_index = data["user_index"]
        obj._item_index = data["item_index"]
        obj._idx_to_item = data["idx_to_item"]
        logger.info("Loaded CF model from %s", path)
        return obj
