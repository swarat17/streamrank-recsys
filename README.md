# StreamRank — Real-Time Product Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Kafka-Streaming-231F20?logo=apachekafka&logoColor=white)
![Apache Spark](https://img.shields.io/badge/PySpark-3.5-E25A1C?logo=apachespark&logoColor=white)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.13-005571?logo=elasticsearch&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Session_Store-DC382D?logo=redis&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Ranker-FF6600)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

A production-grade real-time recommendation system. Kafka ingests live clickstream events, PySpark Structured Streaming updates a Redis session store, and a two-stage collaborative filtering + XGBoost pipeline serves personalised recommendations at **75ms P50 / 102ms P95** via FastAPI. Monitored with Prometheus and Grafana. Trained on the Amazon Electronics dataset (500k interactions, 157k products).

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │        Clickstream Events        │
                        └────────────────┬────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    Kafka Producer    │
                              │  (confluent-kafka)   │
                              └──────────┬──────────┘
                                         │  topic: clickstream-events
                              ┌──────────▼──────────┐
                              │  PySpark Structured  │
                              │  Streaming Consumer  │
                              └──────────┬──────────┘
                                         │  foreachBatch
                              ┌──────────▼──────────┐
                              │   Redis Feature      │
                              │   Store (TTL 30m)    │
                              └──────────┬──────────┘
                                         │
┌────────────────┐            ┌──────────▼──────────────────────────┐
│  POST          │            │       RecommendationPipeline         │
│  /recommend    ├───────────►│                                      │
└────────────────┘            │  1. Redis   → session features       │
                              │  2. ES kNN  → 100 candidates (74ms)  │
                              │  3. Features → FeatureBuilder        │
                              │  4. XGBoost → re-rank (3ms)          │
                              │  5. Return top-N                     │
                              └──────────┬──────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    FastAPI Response  │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────▼──────────────────┐
                    │  Prometheus (/metrics)  →  Grafana     │
                    └───────────────────────────────────────┘
```

---

## How It Works

### 1. Data & Training (Offline)
The Amazon Electronics 2023 dataset (~43M reviews) is filtered to 500k high-quality interactions across 157k products. Two models are trained sequentially:

- **Collaborative Filter (ALS)** — learns a 128-dimensional taste vector for every user and item from implicit feedback. Achieves precision@10 = 0.219 on held-out data.
- **XGBoost Ranker** — takes the top-100 CF candidates and re-ranks them using session context (time of day, session length, whether the item was already viewed, price, rating). Trained at a 1:4 positive/negative ratio. Val AUC = 0.944.

Item embeddings are 192-dimensional: 128d CF vectors concatenated with 64d TF-IDF/SVD text vectors from product titles. These are bulk-indexed into Elasticsearch's HNSW approximate nearest-neighbour index.

### 2. Real-Time Streaming
A Kafka producer continuously publishes clickstream events. A PySpark Structured Streaming consumer reads these in micro-batches and updates per-user session state in Redis with a 30-minute TTL. Session state tracks: last 20 viewed items, event count, cart activity, last-seen timestamp.

### 3. Serving
On each `POST /recommend` request the pipeline runs in four stages:

| Stage | What happens | Latency |
|---|---|---|
| Session lookup | Redis `GET` for recent views + history | ~1ms |
| Candidate retrieval | Elasticsearch kNN — 100 nearest items to user's taste vector | 74ms P50 |
| Feature building | Assembles 11-feature vector per candidate | ~1ms |
| Re-ranking | XGBoost scores all 100 candidates, filters seen items, returns top-N | 3ms P50 |

Cold-start users (no CF embedding) fall back to session-based retrieval (average of recently viewed item vectors), then popularity-based fallback if no session exists.

---

## Performance

| Metric | Value |
|---|---|
| P50 latency (total) | **75ms** |
| P95 latency (total) | **102ms** |
| P99 latency (total) | 220ms (cold-start tail) |
| P50 retrieval (ES kNN) | 74ms |
| P50 ranking (XGBoost) | 3ms |
| CF precision@10 | 0.219 |
| CF recall@10 | 0.307 |
| XGBoost val AUC | 0.944 |
| Items indexed | 157,515 |
| Training interactions | ~500k |
| Cold-start rate | ~20% |

> P99 is elevated by cold-start popularity queries. Known users with CF embeddings are consistently at P50 75ms / P95 98ms.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dataset | Amazon Reviews 2023 (HuggingFace) — Electronics subset |
| Streaming | Apache Kafka · PySpark Structured Streaming |
| Session Store | Redis (TTL-based, LPUSH/LTRIM capped at 20 events) |
| Candidate Retrieval | Elasticsearch 8 — HNSW ANN on 192-dim dense vectors |
| Collaborative Filter | `implicit` ALS (Matrix Factorization, 128 factors) |
| Ranker | XGBoost binary classifier (`binary:logistic`, 300 trees) |
| Serving | FastAPI · Uvicorn · Pydantic v2 |
| Containerisation | Docker · docker-compose |
| MLOps | MLflow model tracking + artifact store |
| Monitoring | Prometheus · Grafana |

---

## Project Structure

```
streamrank-recsys/
├── src/
│   ├── data/
│   │   ├── loader.py          # Amazon dataset download + preprocessing
│   │   └── simulator.py       # Clickstream event generator
│   ├── kafka/
│   │   ├── producer.py        # Confluent-Kafka producer
│   │   └── consumer.py        # PySpark Structured Streaming consumer
│   ├── features/
│   │   ├── redis_store.py     # Redis session read/write (TTL-based)
│   │   └── feature_builder.py # Assembles 11-feature ranker input
│   ├── models/
│   │   ├── collaborative.py   # ALS matrix factorization
│   │   ├── embeddings.py      # 192-dim item embedding generation
│   │   └── ranker.py          # XGBoost pointwise ranker
│   ├── retrieval/
│   │   └── elastic_store.py   # Elasticsearch bulk index + kNN query
│   ├── serving/
│   │   ├── main.py            # FastAPI app (endpoints + startup)
│   │   ├── pipeline.py        # Retrieve → rank → return logic
│   │   └── schemas.py         # Pydantic request/response models
│   └── monitoring/
│       └── metrics.py         # Prometheus metrics definitions
├── infra/
│   ├── kafka/setup_topics.py
│   ├── elasticsearch/setup_index.py
│   └── grafana/dashboard.json
├── scripts/
│   ├── train.py               # Train CF + XGBoost, log to MLflow
│   ├── index_items.py         # Bulk-index embeddings to Elasticsearch
│   └── simulate_traffic.py    # Load generator for Grafana demo
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker-compose.yml
└── Dockerfile
```

---

## Local Setup

**Prerequisites:** Docker Desktop running, Python 3.10+, Java 8+

```bash
# 1. Clone and install dependencies
git clone https://github.com/swarat17/streamrank-recsys.git
cd streamrank-recsys
pip install -r requirements.txt

# 2. Start all infrastructure services
docker-compose up -d

# 3. Download and preprocess the dataset (~15GB, one-time)
python scripts/download_dataset.py

# 4. Train both models (CF + XGBoost) — logs to MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns-artifacts --port 5000 &
python scripts/train.py

# 5. Index item embeddings into Elasticsearch
python scripts/index_items.py

# 6. Get a recommendation
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "AEXAMPLE123", "n_recommendations": 10}'
```

**Dashboards:**
- Grafana: http://localhost:3000 (admin / admin)
- Prometheus: http://localhost:9090
- MLflow: http://localhost:5000

**Generate load for Grafana:**
```bash
python scripts/simulate_traffic.py --rps 10 --duration 60
```

---

## API Reference

### `POST /recommend`
```json
{
  "user_id": "AEXAMPLE123",
  "n_recommendations": 10,
  "context": { "device": "mobile" }
}
```
Returns a ranked list of products with per-stage latency breakdown.

### `GET /health`
```json
{ "status": "ok", "models_loaded": true, "timestamp": 1711234567.0 }
```

### `POST /feedback`
Accepts implicit feedback (click, purchase) to keep session features fresh.

### `GET /metrics`
Prometheus scrape endpoint — latency histograms, request counters, diversity gauge.

---

## Key Design Decisions

**Why two-stage retrieve + rank?**
Scoring all 157k items with XGBoost on every request would take seconds. Elasticsearch's HNSW index narrows the field to 100 plausible candidates in ~74ms, then XGBoost adds only 3ms to pick the best 10. This is the same architecture used by Pinterest, LinkedIn, and YouTube.

**Why implicit ALS instead of explicit ratings?**
Most interactions are views and clicks, not star ratings. ALS treats interaction count as a confidence signal rather than a preference magnitude — it's better suited to this kind of data.

**Why Redis TTL for sessions?**
Session context is only useful while the user is active. A 30-minute TTL means stale sessions auto-expire without any cleanup job. Redis LPUSH+LTRIM keeps the last 20 views capped without scanning.

**Why 192-dim embeddings?**
Concatenating CF (128d) + text (64d) vectors means the ANN search finds items that are similar both in terms of who buys them *and* what they are. A purely CF-based search fails for new items with few interactions (the cold-item problem).

---

## License

MIT
