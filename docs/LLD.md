# Low-Level Design (LLD)
## Real-Time Fraud Detector

---

## 1. Database Schema (PostgreSQL)

### 1.1 `users` / `user_profile`
```sql
CREATE TABLE user_profile (
    user_id             UUID PRIMARY KEY,
    avg_txn_amount       NUMERIC(12,2) DEFAULT 0,
    std_txn_amount        NUMERIC(12,2) DEFAULT 0,
    txn_count             INTEGER DEFAULT 0,
    last_device_id         TEXT,
    last_country           TEXT,
    last_latitude           NUMERIC(9,6),
    last_longitude          NUMERIC(9,6),
    updated_at             TIMESTAMPTZ DEFAULT now()
);
```

### 1.2 `transactions` (raw incoming)
```sql
CREATE TABLE transactions (
    txn_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES user_profile(user_id),
    amount            NUMERIC(12,2) NOT NULL,
    currency          TEXT NOT NULL,
    merchant_id        TEXT,
    device_id          TEXT,
    ip_address          INET,
    country            TEXT,
    latitude            NUMERIC(9,6),
    longitude           NUMERIC(9,6),
    created_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_txn_user_time ON transactions (user_id, created_at DESC);
```

### 1.3 `scored_transactions`
```sql
CREATE TABLE scored_transactions (
    id                 BIGSERIAL PRIMARY KEY,
    txn_id              UUID NOT NULL REFERENCES transactions(txn_id),
    risk_score           NUMERIC(5,4) NOT NULL,       -- 0.0000–1.0000
    flagged              BOOLEAN NOT NULL,
    model_version         TEXT NOT NULL,
    feature_snapshot       JSONB NOT NULL,             -- exact features used
    shap_values            JSONB,                        -- stretch: per-feature attribution
    scored_at              TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_scored_flagged ON scored_transactions (flagged, scored_at DESC);
```

### 1.4 `feedback_labels` (stretch)
```sql
CREATE TABLE feedback_labels (
    id             BIGSERIAL PRIMARY KEY,
    txn_id          UUID NOT NULL REFERENCES transactions(txn_id),
    analyst_id       TEXT NOT NULL,
    decision         TEXT CHECK (decision IN ('confirmed_fraud','false_positive')),
    decided_at        TIMESTAMPTZ DEFAULT now()
);
```

---

## 2. Feature Engineering — Detailed Logic

Implemented as a single shared module (e.g., `features.py`) imported by both the training pipeline and the FastAPI service, to guarantee identical logic at train and serve time.

```python
def compute_features(txn: TransactionInput, profile: UserProfile, recent_txns: list[Transaction]) -> dict:
    now = txn.created_at

    # --- Velocity features ---
    txn_count_1h  = count_txns_in_window(recent_txns, now, hours=1)
    txn_count_24h = count_txns_in_window(recent_txns, now, hours=24)

    # --- Amount deviation ---
    if profile.std_txn_amount > 0:
        amount_zscore = (txn.amount - profile.avg_txn_amount) / profile.std_txn_amount
    else:
        amount_zscore = 0.0
    amount_ratio_to_avg = txn.amount / max(profile.avg_txn_amount, 1e-6)

    # --- Location / device mismatch ---
    device_mismatch = int(txn.device_id != profile.last_device_id)
    country_mismatch = int(txn.country != profile.last_country)
    geo_distance_km = haversine(
        txn.latitude, txn.longitude,
        profile.last_latitude, profile.last_longitude
    ) if profile.last_latitude is not None else 0.0

    return {
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "amount_zscore": amount_zscore,
        "amount_ratio_to_avg": amount_ratio_to_avg,
        "device_mismatch": device_mismatch,
        "country_mismatch": country_mismatch,
        "geo_distance_km": geo_distance_km,
        "amount": txn.amount,
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
    }
```

**Note on `recent_txns` lookup:** MVP fetches via an indexed SQL query (`WHERE user_id = ? AND created_at > now() - interval '24 hours'`). Stretch phase replaces this with a Redis sorted-set keyed by `user_id`, `ZADD`/`ZCOUNT` for O(log n) windowed counts, to keep the Kafka consumer's per-event latency low.

---

## 3. Model Training Pipeline

```python
# train.py — pseudocode structure

1. Load dataset (historical transactions + fraud labels)
2. Split: train/val/test (stratified by label, time-based split preferred
   over random split to avoid leakage from future to past)
3. Apply compute_features() to every row → feature matrix X, labels y
4. Handle class imbalance:
     a. Approach A: SMOTE (imblearn.over_sampling.SMOTE) on training set only
     b. Approach B: class_weight / scale_pos_weight in XGBoost
     → train both, compare on validation PR-AUC, pick winner
5. Train XGBoost classifier:
     model = XGBClassifier(
         scale_pos_weight=<fraud_ratio_inverse>,   # if using approach B
         max_depth=6, n_estimators=300,
         learning_rate=0.05, eval_metric='aucpr'
     )
     model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
6. Evaluate on held-out test set:
     - PR-AUC
     - F1 (fraud class)
     - Recall @ fixed precision (e.g., recall when precision >= 0.90)
     - Confusion matrix
7. Select threshold: choose the score cutoff that meets the target
   false-positive budget defined in the PRD (not a fixed 0.5 default)
8. Serialize model + threshold + feature schema version → model_vX.pkl
```

---

## 4. API Design (FastAPI)

### 4.1 `POST /score`
**Request:**
```json
{
  "txn_id": "uuid",
  "user_id": "uuid",
  "amount": 4599.00,
  "currency": "INR",
  "merchant_id": "merchant_123",
  "device_id": "device_abc",
  "ip_address": "203.0.113.5",
  "country": "IN",
  "latitude": 18.5204,
  "longitude": 73.8567,
  "created_at": "2026-08-07T10:15:00Z"
}
```

**Response:**
```json
{
  "txn_id": "uuid",
  "risk_score": 0.8734,
  "flagged": true,
  "model_version": "xgb_v3",
  "top_reasons": [
    {"feature": "amount_zscore", "value": 4.2},
    {"feature": "device_mismatch", "value": 1}
  ]
}
```
- `top_reasons` in MVP is a simple rule ("top 2 features farthest from the user's typical values"); becomes true SHAP output in the stretch phase.
- Target: p95 < 200ms. Internally: DB lookup for profile/recent txns (~30–60ms budget) + feature computation (~5ms) + model inference (~10–20ms) + serialization/network overhead.

### 4.2 `GET /flags`
Query params: `since`, `limit`, `min_score`. Returns paginated flagged transactions for the dashboard.

### 4.3 `POST /feedback` (stretch)
```json
{ "txn_id": "uuid", "analyst_id": "a123", "decision": "confirmed_fraud" }
```
Writes to `feedback_labels`.

---

## 5. Class / Module Structure (Backend)

```
app/
├── main.py                # FastAPI app, route registration
├── api/
│   ├── score.py           # POST /score
│   ├── flags.py           # GET /flags
│   └── feedback.py        # POST /feedback (stretch)
├── ml/
│   ├── features.py        # shared feature engineering (train + serve)
│   ├── model_loader.py     # loads model_vX.pkl at startup, exposes .predict()
│   └── explain.py          # SHAP wrapper (stretch)
├── db/
│   ├── models.py           # SQLAlchemy models mirroring schema above
│   └── repository.py       # query helpers (get_recent_txns, get_profile, etc.)
├── streaming/               # stretch
│   ├── producer.py
│   └── consumer.py          # Kafka consumer, calls ml/features.py + model_loader
└── config.py
```

Key design principle: `ml/features.py` has **zero dependency** on FastAPI or Kafka — it's pure functions operating on plain data structures, so `train.py`, `api/score.py`, and `streaming/consumer.py` can all call it identically.

---

## 6. Frontend Component Structure (React)

```
src/
├── App.jsx
├── components/
│   ├── FlaggedTransactionFeed.jsx   # live/polled list of flags
│   ├── RiskScoreDistribution.jsx    # Recharts histogram
│   ├── FlagsOverTimeChart.jsx       # Recharts line chart
│   ├── TransactionDetailPanel.jsx   # shows features + (stretch) SHAP bar chart
│   └── FeedbackButtons.jsx          # confirm/reject (stretch)
├── api/
│   └── client.js                     # fetch wrappers for /flags, /feedback
└── hooks/
    └── usePolling.js                  # simple interval-based refresh for MVP
```

---

## 7. Sequence: Scoring a Transaction (MVP path)

```
Client → POST /score → FastAPI
   → repository.get_user_profile(user_id)
   → repository.get_recent_txns(user_id, window=24h)
   → features.compute_features(txn, profile, recent_txns)
   → model_loader.predict(feature_vector) → risk_score
   → threshold check → flagged (bool)
   → repository.save_scored_transaction(...)
   → return response to client
```

## 8. Non-Functional Requirements Mapping

| Requirement | Design Decision |
|---|---|
| < 200ms p95 latency | Indexed velocity queries, in-memory model, precomputed profile aggregates |
| Severe class imbalance | SMOTE + class-weighting benchmarked; PR-AUC/recall-based eval, not accuracy |
| No train/serve skew | Single shared `features.py` module |
| Extensible to streaming | Kafka consumer reuses the same `features.py` + `model_loader.py` |
| Auditability | `feature_snapshot` JSONB stored per scored transaction for reproducibility |

