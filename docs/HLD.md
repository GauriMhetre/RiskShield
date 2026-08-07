# High-Level Design (HLD)
## Real-Time Fraud Detector

---

## 1. System Overview

The system is a **transaction-scoring service**: a backend API receives transaction events, computes behavioral features, runs them through a trained classifier, persists the result, and surfaces flagged transactions on a live dashboard. It's designed to sit in the authorization path (synchronous call) for MVP, with a path to becoming a fully event-driven, streaming system in the stretch phase.

## 2. Architecture Diagram (Phase 1–2: MVP)

```
                ┌────────────────────┐
                │   Client / POS /    │
                │  Payment Gateway    │
                └─────────┬──────────┘
                          │ POST /score  (transaction JSON)
                          ▼
                ┌────────────────────┐
                │   FastAPI Service   │
                │  ┌──────────────┐  │
                │  │ Feature Eng.  │  │  ← velocity, amount deviation,
                │  │   Module      │  │     location/device mismatch
                │  └──────┬───────┘  │
                │         ▼           │
                │  ┌──────────────┐  │
                │  │  XGBoost      │  │  ← loaded once at startup
                │  │  Model (.pkl) │  │
                │  └──────┬───────┘  │
                │         ▼           │
                │   risk_score,       │
                │   flag (bool)       │
                └─────────┬──────────┘
                          │
             ┌────────────┴─────────────┐
             ▼                           ▼
   ┌───────────────────┐      ┌───────────────────┐
   │   PostgreSQL        │      │   API Response      │
   │ (scored_transactions)│      │  (to caller)        │
   └─────────┬───────────┘      └───────────────────┘
             │
             ▼
   ┌───────────────────┐
   │  React Dashboard    │
   │  (Recharts)         │
   │  polls / fetches     │
   │  flagged txns        │
   └───────────────────┘
```

## 3. Architecture Diagram (Phase 3: Stretch — Streaming)

```
Transaction Producer → Kafka Topic (transactions)
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Kafka Consumer /        │
                  │  Scoring Worker           │
                  │  (feature eng + model)    │
                  └───────────┬────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
     scored_transactions   flags topic      SHAP values
     (PostgreSQL)          (Kafka, for      (attached to
                            dashboard        flagged txn
                            consumer)        record)
                              │
                              ▼
                   Dashboard (live feed via
                   websocket/poll on flags)
                              │
                              ▼
              Analyst confirms/rejects flag
                              │
                              ▼
                 feedback_labels table
                              │
                              ▼
              Periodic retrain job (cron/Airflow)
                     → new model.pkl
                     → hot-swapped into scoring worker
```

## 4. Components

### 4.1 Feature Engineering Module
Responsible for turning a raw transaction + user history into a fixed feature vector. Must run identically at training and inference time (shared code path, not duplicated logic).
- **Velocity features:** requires quick lookup of a user's recent transaction count. For MVP this can be a SQL query against PostgreSQL with an index on `(user_id, timestamp)`; for the streaming stretch phase, a Redis-backed rolling counter is more appropriate given the latency budget.
- **Amount deviation:** requires the user's historical mean/std of transaction amount — precomputed and cached (e.g., updated nightly, or incrementally).
- **Location/device mismatch:** requires the user's last known device fingerprint / geolocation, stored in a `user_profile` table.

### 4.2 Model Serving Layer
- FastAPI app, model loaded into memory at process startup (not per-request).
- Stateless — horizontally scalable behind a load balancer if needed.
- Model artifact versioned (filename or model registry entry) so the retrain loop can hot-swap without downtime.

### 4.3 Persistence Layer (PostgreSQL)
- `scored_transactions`: every scoring event, immutable log.
- `user_profile`: rolling aggregates per user (mean/std amount, last device, last location) — updated asynchronously.
- `feedback_labels` (stretch): analyst decisions, used to build retraining datasets.

### 4.4 Dashboard
- React frontend, Recharts for visualizations (risk score distribution, flags-over-time).
- Talks to a read API (either the same FastAPI service or a thin `/flags` query endpoint) — not directly to the DB.

### 4.5 Streaming Layer (Stretch)
- Kafka topic `transactions` — producers publish raw transaction events.
- A consumer/worker group performs feature engineering + scoring, replacing the synchronous `/score` call for the streaming path (the REST endpoint can remain for direct/manual scoring or as a fallback).
- Scored/flagged results published to a `flags` topic or written directly to PostgreSQL for the dashboard to pick up.

### 4.6 Model Monitoring (Stretch)
- A scheduled job compares the distribution of live feature values against the training distribution (e.g., Population Stability Index) and logs/alerts on drift.

### 4.7 Explainability (Stretch)
- SHAP values computed at scoring time (or async, if latency-sensitive) for flagged transactions only, stored alongside the flag so the dashboard can render a feature-attribution breakdown.

### 4.8 Feedback & Retraining Loop (Stretch)
- Dashboard action lets an analyst mark a flag as confirmed fraud / false positive.
- These labels accumulate in `feedback_labels`.
- A periodic job (cron or Airflow-style) retrains the model on the expanded labeled set and produces a new versioned model artifact, which the serving layer picks up (rolling reload).

## 5. Technology Choices & Rationale

| Layer | Choice | Why |
|---|---|---|
| API framework | FastAPI | Async-friendly, low overhead, auto-generated docs, good fit for a sub-200ms latency target. |
| ML | XGBoost + scikit-learn | XGBoost handles tabular/imbalanced data well and is fast at inference; scikit-learn/imbalanced-learn provide SMOTE and baseline RF for comparison. |
| Database | PostgreSQL | Relational integrity for transaction records, good indexing for velocity-feature lookups, mature and free. |
| Streaming (stretch) | Kafka | Standard for durable, decoupled, high-throughput event streaming — natural fit for "true real-time" scoring. |
| Frontend | React + Recharts | Component-driven UI, Recharts gives quick, clean charting without heavy custom D3 work. |

## 6. Cross-Cutting Concerns

- **Latency budget:** feature lookup + inference must fit inside 200ms — this drives the decision to cache/precompute aggregates rather than compute them from scratch per request.
- **Class imbalance:** addressed at training time (SMOTE/class weighting), not by post-hoc threshold hacking alone.
- **No train/serve skew:** feature engineering logic must be a shared module imported by both the training script and the serving API — never reimplemented twice.
- **Extensibility:** MVP architecture (synchronous REST scoring) is designed so Phase 3 can add a Kafka consumer path *alongside* it without a rewrite.

