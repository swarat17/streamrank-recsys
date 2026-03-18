"""Send realistic traffic to the FastAPI /recommend endpoint for Grafana demo."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import random
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.55, 0.35, 0.10]


def load_user_ids(max_users: int = 500) -> tuple[list[str], list[str]]:
    """Load real user IDs split into known (with CF embeddings) and cold-start pools."""
    try:
        from src.data.loader import load_interactions
        df = load_interactions()
        all_users = df["user_id"].unique().tolist()
        random.shuffle(all_users)
        known = all_users[: max_users // 2]
        cold = [f"cold_user_{i}" for i in range(max_users // 2)]
        logger.info("Loaded %d known users + %d cold-start users.", len(known), len(cold))
        return known, cold
    except Exception as e:
        logger.warning("Could not load real users (%s) — using synthetic IDs.", e)
        known = [f"known_user_{i}" for i in range(250)]
        cold = [f"cold_user_{i}" for i in range(250)]
        return known, cold


def simulate(base_url: str, rps: float, duration: int, cold_start_ratio: float = 0.2) -> None:
    known_users, cold_users = load_user_ids()
    delay = 1.0 / rps
    t_end = time.time() + duration
    n_success = n_error = n_cold = 0

    logger.info("Simulating at %.1f RPS for %ds → %s", rps, duration, base_url)

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        while time.time() < t_end:
            t0 = time.time()
            is_cold = random.random() < cold_start_ratio
            user_id = random.choice(cold_users if is_cold else known_users)
            device = random.choices(DEVICES, weights=DEVICE_WEIGHTS, k=1)[0]

            try:
                resp = client.post("/recommend", json={
                    "user_id": user_id,
                    "n_recommendations": random.randint(5, 15),
                    "context": {"device": device},
                })
                if resp.status_code == 200:
                    n_success += 1
                    recs = resp.json().get("recommendations", [])

                    # Occasionally send feedback for a returned item
                    if recs and random.random() < 0.3:
                        item_id = random.choice(recs)["item_id"]
                        event = random.choice(["click", "add_to_cart", "purchase"])
                        client.post("/feedback", json={
                            "user_id": user_id,
                            "item_id": item_id,
                            "event_type": event,
                            "context": {"device": device},
                        })
                else:
                    n_error += 1
                if is_cold:
                    n_cold += 1
            except Exception as e:
                n_error += 1
                logger.debug("Request error: %s", e)

            elapsed = time.time() - t0
            sleep_for = max(0.0, delay - elapsed)
            time.sleep(sleep_for)

    total = n_success + n_error
    logger.info(
        "Done. Sent %d requests — success=%d error=%d cold_start=%d (%.0f%%)",
        total, n_success, n_error, n_cold,
        100 * n_cold / total if total else 0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate recommendation traffic")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=float, default=10.0)
    parser.add_argument("--duration", type=int, default=60, help="Seconds to run")
    parser.add_argument("--cold-ratio", type=float, default=0.2,
                        help="Fraction of cold-start requests")
    args = parser.parse_args()
    simulate(args.url, args.rps, args.duration, args.cold_ratio)
