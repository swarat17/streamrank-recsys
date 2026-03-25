"""Train collaborative filtering + XGBoost ranker and register both in MLflow."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import os

import mlflow
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))

    with mlflow.start_run(run_name="streamrank-training") as run:
        logger.info("MLflow run id: %s", run.info.run_id)

        # ── 1. Load data ──────────────────────────────────────────────────────
        from src.data.loader import load_interactions, load_items
        logger.info("Loading data...")
        interactions_df = load_interactions()
        items_df = load_items()
        logger.info("Interactions: %s  Items: %s", interactions_df.shape, items_df.shape)

        np.random.seed(args.seed)

        # ── 2. Train CF model ─────────────────────────────────────────────────
        from src.models.collaborative import MatrixFactorizationModel
        logger.info("Training collaborative filter (factors=%d)...", args.cf_factors)
        cf_model = MatrixFactorizationModel(
            factors=args.cf_factors, regularization=0.01, iterations=30
        )
        cf_model.train(interactions_df, mlflow_run=True)
        cf_model.save(Path("models/cf"))
        mlflow.log_artifact("models/cf/cf_model.pkl", artifact_path="cf_model")

        # ── 3. Generate item embeddings ───────────────────────────────────────
        from src.models.embeddings import ItemEmbeddingGenerator, EMBEDDINGS_PATH, PROJECTION_PATH
        logger.info("Generating 192-dim item embeddings...")
        gen = ItemEmbeddingGenerator()
        embeddings = gen.build(cf_model, items_df)
        gen.save(embeddings)
        mlflow.log_artifact(str(EMBEDDINGS_PATH), artifact_path="embeddings")
        mlflow.log_artifact(str(PROJECTION_PATH), artifact_path="embeddings")
        mlflow.log_metric("n_item_embeddings", len(embeddings))

        # ── 4. Build XGBoost training data ────────────────────────────────────
        from src.models.ranker import XGBoostRanker, build_training_data
        logger.info("Building XGBoost training data...")
        features_df, labels = build_training_data(
            interactions_df, items_df, cf_model, n_users=5_000, seed=args.seed
        )

        # ── 5. Train XGBoost ranker ───────────────────────────────────────────
        logger.info("Training XGBoost ranker (n_estimators=%d)...", args.xgb_estimators)
        ranker = XGBoostRanker(
            n_estimators=args.xgb_estimators, max_depth=6, learning_rate=0.05, seed=args.seed
        )
        ranker.train(features_df, labels, mlflow_run=True)
        ranker.save(Path("models/ranker"))
        mlflow.log_artifact("models/ranker/ranker.pkl", artifact_path="ranker_model")

        # ── 6. Register both models in MLflow ─────────────────────────────────
        cf_uri = f"runs:/{run.info.run_id}/cf_model"
        ranker_uri = f"runs:/{run.info.run_id}/ranker_model"

        try:
            mlflow.register_model(cf_uri, "collaborative-filter")
            mlflow.register_model(ranker_uri, "xgboost-ranker")
        except Exception as e:
            logger.warning("Model registry skipped (%s). Models are saved locally.", e)

        mlflow.set_tags({
            "stage": "production",
            "cf_model_type": "ALS",
            "ranker_model_type": "XGBoost",
        })

    logger.info("\n✅ Training complete.")
    logger.info("CF model URI:     %s", cf_uri)
    logger.info("Ranker model URI: %s", ranker_uri)
    logger.info("Run 'mlflow ui' to inspect metrics and artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train recsys models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cf-factors", type=int, default=128)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    main(parser.parse_args())
