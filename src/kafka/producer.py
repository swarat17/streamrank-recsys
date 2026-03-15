"""Kafka producer for clickstream events."""

import json
import time
import logging
import os
import argparse

from confluent_kafka import Producer

logger = logging.getLogger(__name__)


class ClickstreamProducer:
    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str = "clickstream-events",
    ):
        servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self._topic = topic
        self._producer = Producer({"bootstrap.servers": servers})
        logger.info("ClickstreamProducer connected to %s, topic=%s", servers, topic)

    def _delivery_report(self, err, msg):
        if err:
            logger.error("Delivery failed for %s: %s", msg.key(), err)

    def send_event(self, event: dict) -> None:
        """Serialize and send a single event to Kafka."""
        self._producer.produce(
            self._topic,
            key=event.get("user_id", ""),
            value=json.dumps(event).encode("utf-8"),
            callback=self._delivery_report,
        )
        self._producer.poll(0)

    def send_batch(self, events: list[dict]) -> None:
        """Send a batch of events and log throughput."""
        t0 = time.time()
        for event in events:
            self.send_event(event)
        self._producer.flush()
        elapsed = time.time() - t0
        rate = len(events) / elapsed if elapsed > 0 else float("inf")
        logger.info("Sent %d events in %.2fs (%.0f events/sec)", len(events), elapsed, rate)

    def simulate_live(self, events_per_second: float = 10.0) -> None:
        """Continuously produce events using the ClickstreamSimulator."""
        from src.data.simulator import ClickstreamSimulator
        sim = ClickstreamSimulator()
        logger.info("Starting live simulation at %.1f events/sec...", events_per_second)
        for event in sim.stream(events_per_second):
            self.send_event(event)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=10.0, help="Events per second")
    parser.add_argument(
        "--servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    )
    args = parser.parse_args()
    ClickstreamProducer(bootstrap_servers=args.servers).simulate_live(args.rate)
