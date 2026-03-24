"""Bootstrap script that redirects ALL caches to G: drive before downloading the dataset."""
import os
import sys

# Must be set BEFORE any huggingface/datasets imports
_CACHE_ROOT = "G:/hf-cache"
os.environ["HF_HOME"] = _CACHE_ROOT
os.environ["HF_DATASETS_CACHE"] = f"{_CACHE_ROOT}/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_CACHE_ROOT}/hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Redirect Python temp dir so extraction doesn't hit C: drive
_TMP = "G:/tmp"
os.makedirs(_TMP, exist_ok=True)
os.environ["TEMP"] = _TMP
os.environ["TMP"] = _TMP
os.environ["TMPDIR"] = _TMP

import tempfile
tempfile.tempdir = _TMP  # override in-process tempfile default

# Now safe to import
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from src.data.loader import build_dataset

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    build_dataset()
    print("Dataset download and processing complete.")
