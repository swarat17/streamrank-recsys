"""XGBoost pointwise ranker — second stage of the two-stage recommendation pipeline."""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")

# Feature columns fed to XGBoost (matches FeatureBuilder output minus string fields)
FEATURE_COLS = [
    "session_length",
    "session_age_seconds",
    "n_recent_views",
    "candidate_in_recent_views",
    "candidate_avg_rating",
    "candidate_price",
    "candidate_review_count",
    "time_of_day",
    "device_mobile",
]


class XGBoostRanker:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self._model: Optional[xgb.XGBClassifier] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        mlflow_run: bool = True,
    ) -> None:
        """Train XGBoost binary classifier and log metrics + importances to MLflow."""
        X = features_df[FEATURE_COLS].astype(float)
        y = labels.values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )

        self._model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            eval_metric="auc",
            early_stopping_rounds=20,
            verbosity=0,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        train_auc = roc_auc_score(y_train, self._model.predict_proba(X_train)[:, 1])
        val_auc = roc_auc_score(y_val, self._model.predict_proba(X_val)[:, 1])
        logger.info("XGBoost — train AUC: %.4f  val AUC: %.4f", train_auc, val_auc)

        importance = dict(zip(FEATURE_COLS, self._model.feature_importances_.tolist()))

        if mlflow_run:
            mlflow.log_params({
                "xgb_n_estimators": self.n_estimators,
                "xgb_max_depth": self.max_depth,
                "xgb_learning_rate": self.learning_rate,
            })
            mlflow.log_metrics({"xgb_train_auc": train_auc, "xgb_val_auc": val_auc})
            mlflow.log_dict(importance, "feature_importances.json")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_scores(self, features: list[dict]) -> list[float]:
        """Return probability scores for a list of candidate feature dicts."""
        if not features:
            return []
        df = pd.DataFrame(features)[FEATURE_COLS].astype(float)
        return self._model.predict_proba(df)[:, 1].tolist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "ranker.pkl", "wb") as f:
            pickle.dump({
                "model": self._model,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            }, f)
        logger.info("Saved XGBoost ranker to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostRanker":
        path = Path(path)
        with open(path / "ranker.pkl", "rb") as f:
            data = pickle.load(f)
        obj = cls(
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
            learning_rate=data["learning_rate"],
            seed=data["seed"],
        )
        obj._model = data["model"]
        logger.info("Loaded XGBoost ranker from %s", path)
        return obj


# ------------------------------------------------------------------
# Training data construction
# ------------------------------------------------------------------

def build_training_data(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    cf_model,
    n_users: int = 5_000,
    neg_ratio: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build XGBoost training data from CF candidates.

    Returns:
        (features_df, labels) where labels is a binary pd.Series.
    """
    from src.features.feature_builder import FeatureBuilder

    rng = np.random.default_rng(seed)
    items_lookup = items_df.set_index("item_id").to_dict("index")
    builder = FeatureBuilder()

    # Sample users (cap at n_users)
    all_users = interactions_df["user_id"].unique()
    sampled_users = rng.choice(
        all_users, size=min(n_users, len(all_users)), replace=False
    )

    feature_rows: list[dict] = []
    label_rows: list[int] = []

    logger.info("Building training data for %d users...", len(sampled_users))
    for user_id in sampled_users:
        if not cf_model.has_user(user_id):
            continue

        positive_items = set(
            interactions_df[interactions_df["user_id"] == user_id]["item_id"].tolist()
        )
        candidates = cf_model.recommend_for_user(user_id, n=100)
        negatives = [c for c in candidates if c not in positive_items]

        # Default session features for training (no live Redis)
        session = {
            "user_id": user_id,
            "recent_views": list(positive_items)[:5],
            "session_length": len(positive_items),
            "session_age_seconds": 300.0,
            "has_cart_item": False,
        }

        for item_id in positive_items:
            if item_id not in items_lookup:
                continue
            feat = builder.build(session, {"item_id": item_id, **items_lookup[item_id]})
            feature_rows.append(feat)
            label_rows.append(1)

        # Sample negatives at neg_ratio × positives
        n_neg = min(len(negatives), neg_ratio * len(positive_items))
        for item_id in rng.choice(negatives, size=n_neg, replace=False) if negatives else []:
            if item_id not in items_lookup:
                continue
            feat = builder.build(session, {"item_id": item_id, **items_lookup[item_id]})
            feature_rows.append(feat)
            label_rows.append(0)

    features_df = pd.DataFrame(feature_rows)
    labels = pd.Series(label_rows, name="label")
    logger.info(
        "Training data: %d rows (%d positives, %d negatives)",
        len(features_df), sum(label_rows), len(label_rows) - sum(label_rows),
    )
    return features_df, labels
