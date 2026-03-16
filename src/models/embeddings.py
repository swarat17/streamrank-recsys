"""Item embedding generation: CF (128d) + text TF-IDF/SVD (64d) = 192d."""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.models.collaborative import MatrixFactorizationModel

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
EMBEDDINGS_PATH = MODELS_DIR / "item_embeddings.pkl"
PROJECTION_PATH = MODELS_DIR / "projection_matrix.npy"

CF_DIM = 128
TEXT_DIM = 64
TOTAL_DIM = CF_DIM + TEXT_DIM  # 192


class ItemEmbeddingGenerator:
    """Combines CF and text embeddings into 192-dimensional item vectors."""

    def __init__(self):
        self._tfidf = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
        self._svd = TruncatedSVD(n_components=TEXT_DIM, random_state=42)

    def build(
        self,
        cf_model: MatrixFactorizationModel,
        items_df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Generate 192-dim embeddings for all items in items_df.

        Returns:
            dict mapping item_id → 192d numpy array.
        """
        logger.info("Building text corpus...")
        items_df = items_df.copy()
        items_df["text"] = (
            items_df["title"].fillna("") + " " + items_df["description_snippet"].fillna("")
        )

        logger.info("Fitting TF-IDF on %d items...", len(items_df))
        tfidf_matrix = self._tfidf.fit_transform(items_df["text"])

        # Cap SVD components to vocabulary size (small datasets may have fewer features)
        n_components = min(TEXT_DIM, tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        logger.info("Reducing to %d dims via TruncatedSVD...", n_components)
        text_vecs_raw = self._svd.fit_transform(tfidf_matrix)
        # Pad to TEXT_DIM if vocabulary was smaller than TEXT_DIM
        if n_components < TEXT_DIM:
            pad = np.zeros((text_vecs_raw.shape[0], TEXT_DIM - n_components), dtype=text_vecs_raw.dtype)
            text_vecs = np.hstack([text_vecs_raw, pad])
        else:
            text_vecs = text_vecs_raw
        text_vecs = normalize(text_vecs, norm="l2")  # unit-norm

        embeddings: dict[str, np.ndarray] = {}
        missing_cf = 0

        for i, row in items_df.iterrows():
            item_id = row["item_id"]
            text_vec = text_vecs[items_df.index.get_loc(i)]  # 64d

            if cf_model.has_item(item_id):
                raw_cf = cf_model.get_item_embedding(item_id)
                # Pad to CF_DIM if model was trained with fewer factors (e.g. in tests)
                if raw_cf.shape[0] < CF_DIM:
                    cf_vec = np.pad(raw_cf, (0, CF_DIM - raw_cf.shape[0])).astype(np.float32)
                else:
                    cf_vec = raw_cf.astype(np.float32)
            else:
                cf_vec = np.zeros(CF_DIM, dtype=np.float32)
                missing_cf += 1

            combined = np.concatenate([cf_vec, text_vec.astype(np.float32)])
            embeddings[item_id] = combined

        logger.info(
            "Built %d embeddings (%d without CF — used zero-vector). dim=%d",
            len(embeddings), missing_cf, TOTAL_DIM,
        )
        return embeddings

    def save(
        self,
        embeddings: dict[str, np.ndarray],
        embeddings_path: Path = EMBEDDINGS_PATH,
        projection_path: Path = PROJECTION_PATH,
    ) -> None:
        """Save embeddings dict and user→item projection matrix."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info("Saved %d item embeddings to %s", len(embeddings), embeddings_path)

        # Projection matrix: maps 128d user CF embedding → 192d space
        # Strategy: pad CF embedding with 64 zeros (text part unknown for users).
        # Shape (192, 128): when applied as M @ user_vec produces 192d output.
        proj = np.zeros((TOTAL_DIM, CF_DIM), dtype=np.float32)
        proj[:CF_DIM, :CF_DIM] = np.eye(CF_DIM, dtype=np.float32)
        np.save(projection_path, proj)
        logger.info("Saved projection matrix to %s", projection_path)


def load_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_projection_matrix(path: Path = PROJECTION_PATH) -> np.ndarray:
    return np.load(path)


def project_user_embedding(user_cf_vec: np.ndarray, proj_matrix: np.ndarray) -> np.ndarray:
    """Project a 128d user CF embedding to 192d for Elasticsearch kNN."""
    result = proj_matrix @ user_cf_vec.astype(np.float32)
    norm = np.linalg.norm(result)
    return result / norm if norm > 0 else result
