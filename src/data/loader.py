"""Download and preprocess the Amazon Electronics dataset from HuggingFace."""

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"
ITEMS_PATH = PROCESSED_DIR / "items.parquet"

MIN_USER_INTERACTIONS = 5
MIN_ITEM_REVIEWS = 10


def _download_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Downloading Amazon Reviews 2023 — Electronics subset...")
    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Electronics",
        split="full",
        trust_remote_code=True,
    )
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Electronics",
        split="full",
        trust_remote_code=True,
    )
    # Select only needed columns before to_pandas() to avoid OOM on 43M rows
    reviews_df = reviews.select_columns(
        ["user_id", "asin", "rating", "timestamp", "verified_purchase"]
    ).to_pandas()
    meta_df = meta.select_columns(
        ["parent_asin", "title", "categories", "price", "average_rating", "rating_number", "description"]
    ).to_pandas()
    logger.info("Downloaded %d reviews and %d items.", len(reviews_df), len(meta_df))
    return reviews_df, meta_df


def _build_interactions(reviews_df: pd.DataFrame) -> pd.DataFrame:
    df = reviews_df[["user_id", "asin", "rating", "timestamp", "verified_purchase"]].copy()
    df = df.rename(columns={"asin": "item_id"})
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    # Two-pass filter: users >= MIN_USER_INTERACTIONS, items >= MIN_ITEM_REVIEWS
    for _ in range(2):
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= MIN_ITEM_REVIEWS].index)]

    df = df.reset_index(drop=True)
    logger.info(
        "Interactions: %d rows, %d unique users, %d unique items.",
        len(df), df["user_id"].nunique(), df["item_id"].nunique(),
    )
    return df


def _safe_price(p) -> float:
    if p is None:
        return float("nan")
    if isinstance(p, (int, float)):
        return float(p)
    s = str(p).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _build_items(meta_df: pd.DataFrame, valid_item_ids: set) -> pd.DataFrame:
    rows = []
    for row in meta_df.to_dict("records"):
        asin = row.get("parent_asin") or row.get("asin")
        if asin not in valid_item_ids:
            continue
        categories = row.get("categories")
        if not isinstance(categories, (list, tuple)) or len(categories) == 0:
            categories = []
        category = categories[0] if categories else "Unknown"
        desc_parts = row.get("description")
        if not isinstance(desc_parts, (list, tuple)):
            desc_parts = []
        description = " ".join(str(p) for p in desc_parts)
        rows.append({
            "item_id": asin,
            "title": str(row.get("title") or ""),
            "category": category,
            "price": _safe_price(row.get("price")),
            "avg_rating": float(row.get("average_rating") or 0),
            "review_count": int(row.get("rating_number") or 0),
            "description_snippet": description[:200],
        })

    df = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    logger.info("Items DataFrame: %d rows.", len(df))
    return df


def build_dataset(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download, process, and save both DataFrames. Skip if already on disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not force and INTERACTIONS_PATH.exists() and ITEMS_PATH.exists():
        logger.info("Processed files already exist — loading from disk.")
        return load_interactions(), load_items()

    reviews_df, meta_df = _download_raw()
    interactions_df = _build_interactions(reviews_df)
    items_df = _build_items(meta_df, set(interactions_df["item_id"].unique()))

    interactions_df.to_parquet(INTERACTIONS_PATH, index=False)
    items_df.to_parquet(ITEMS_PATH, index=False)
    logger.info("Saved to %s and %s.", INTERACTIONS_PATH, ITEMS_PATH)
    return interactions_df, items_df


def load_interactions() -> pd.DataFrame:
    """Load the preprocessed interactions DataFrame from disk."""
    return pd.read_parquet(INTERACTIONS_PATH)


def load_items() -> pd.DataFrame:
    """Load the preprocessed items DataFrame from disk."""
    return pd.read_parquet(ITEMS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    interactions, items = build_dataset()
    print(f"Interactions: {interactions.shape}")
    print(f"Items: {items.shape}")
