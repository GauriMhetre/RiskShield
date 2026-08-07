# Product Requirements Document (PRD)
## Real-Time Fraud Detector

**Tier:** S — Build First
**Category:** Machine Learning / Deep Learning
**Status:** Draft v1.0

---

## 1. Problem Statement

Financial platforms lose money and user trust when fraudulent transactions are caught only after settlement — by which point the funds have often already moved. What's needed is a system that scores a transaction's fraud risk **at the moment of the swipe/click**, fast enough to block or flag it before authorization completes.

## 2. Goals & Non-Goals

### Goals
- Score every incoming transaction for fraud risk in **under 200ms**.
- Correctly separate genuine transactions from fraudulent ones despite fraud being a small minority of all transactions (severe class imbalance).
- Give analysts a live, actionable view of flagged transactions with enough context to act on them.
- Build a system that's extendable toward true streaming and self-improving feedback loops.

### Non-Goals (for MVP)
- Not building a full case-management/ticketing system for fraud analysts — just a review dashboard.
- Not handling multi-currency/multi-region regulatory reporting.
- Not doing on-device/edge scoring — this is a server-side API service.
- Kafka streaming, drift monitoring, SHAP, and the retrain loop are explicitly **stretch**, not MVP.

## 3. Target Users

| User | Need |
|---|---|
| **Fraud analyst** | Wants a live dashboard of flagged transactions, ranked by risk, with enough detail to confirm/reject quickly. |
| **Platform/backend engineer (integrator)** | Wants a simple, fast, reliable REST endpoint to call synchronously during transaction authorization. |
| **Data scientist (you, iterating)** | Wants visibility into model performance, feature importance, and a clean retraining path. |

## 4. Core Features (MVP)

### 4.1 Feature Engineering
- **Velocity features:** transaction count per user in trailing 1hr / 24hr windows.
- **Amount deviation:** z-score or ratio of current transaction amount vs. that user's historical mean/std.
- **Location/device mismatch:** binary/categorical flags — new device fingerprint, geolocation distance from last known location, IP/country mismatch vs. billing country.
- Feature computation must be reproducible identically at both training time and inference time (no train/serve skew).

### 4.2 Model
- Binary classifier: **XGBoost** as primary, Random Forest as baseline comparison.
- Class imbalance handled via **SMOTE** (oversampling minority/fraud class) and/or **class weighting** (`scale_pos_weight` in XGBoost) — both implemented, benchmarked, and the better one shipped.
- Evaluation metrics centered on **Precision-Recall AUC, F1 on the fraud class, and recall at a fixed false-positive budget** — not raw accuracy, which is meaningless under imbalance.

### 4.3 Serving
- REST API (FastAPI) exposing `POST /score` — accepts a transaction payload, returns a risk score (0–1) and a flag (boolean, above threshold).
- **p95 latency target: < 200ms**, including feature lookup and model inference.
- Model loaded once at startup (or via a model registry), not reloaded per request.

### 4.4 Persistence
- Every scored transaction (input features, score, flag, timestamp) is written to PostgreSQL.
- Schema supports later querying by user, time range, risk band, and analyst decision.

### 4.5 Live Dashboard
- React + Recharts frontend showing:
  - A live/near-live feed of flagged transactions (risk score, amount, user, reason flags).
  - Aggregate view: flags over time, risk score distribution, top flag reasons.
- Polling or lightweight refresh acceptable for MVP; no hard requirement for websockets.

## 5. Stretch Features (9.5+ Upgrade)

| Feature | What it adds |
|---|---|
| **Kafka streaming** | Transactions flow through a Kafka topic instead of being scored via direct batch/API calls only — true event-driven real-time scoring, decoupled producer/consumer. |
| **Model monitoring / drift detection** | Track feature and prediction distribution drift over time (e.g., PSI or KS-test on incoming feature distributions vs. training distribution); surface drift alerts. |
| **Explainability (SHAP)** | Each flagged transaction is shown with its top contributing features (SHAP values), so an analyst can see *why* it was flagged. |
| **Feedback loop** | Analyst confirms/rejects a flag in the dashboard → decision stored → periodic retraining job incorporates confirmed labels to improve the model over time. |

## 6. Success Metrics

- **Model quality:** Precision-Recall AUC on held-out fraud data; target recall ≥ 0.85 on fraud class at a precision that keeps false-positive rate operationally tolerable (e.g., ≤ 2% of legitimate transactions flagged).
- **Latency:** p95 API response time < 200ms under realistic load.
- **Usability:** an analyst can look at the dashboard and understand, within seconds, which transactions need attention and why.
- **Stretch success:** demonstrable drift alert firing on injected distribution shift; SHAP values rendered per flagged transaction; at least one full retrain cycle triggered by feedback.

## 7. Scope & Build Sequence

| Phase | Deliverable |
|---|---|
| **Phase 1** | Static dataset, feature engineering pipeline, trained classifier, scoring API (`/score`) working end-to-end on historical data. |
| **Phase 2** | Live dashboard wired to the API + database; scoring pipeline handling a simulated live stream (even if via polling/batch injection). |
| **Phase 3** | Kafka-based true streaming, SHAP explainability panel, analyst feedback capture, periodic retraining job. |

## 8. Risks & Open Questions

- **Data availability:** no public dataset perfectly mirrors production behavioral signals (device fingerprinting, live geolocation). Likely need to use a public fraud dataset (e.g., IEEE-CIS or a synthetic transaction generator) and simulate velocity/location features.
- **Latency vs. feature complexity:** richer behavioral features (e.g., rolling windows) require fast feature stores (Redis) if velocity features are computed live — worth deciding early whether to precompute or compute on-the-fly.
- **Imbalance handling choice:** SMOTE vs. class weighting can materially change results; both must be benchmarked, not assumed.
- **False positive cost:** needs an explicit business threshold decision (precision/recall tradeoff) — this should be tunable, not hardcoded.

