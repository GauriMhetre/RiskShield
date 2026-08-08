"""
POST /score endpoint for real-time fraud risk scoring.

This module defines the FastAPI router for the /score endpoint, which takes a transaction
request, scores it using the trained XGBoost model, and returns a fraud risk assessment.

The route ties together:
  1. Request validation (ScoreRequest Pydantic model)
  2. User profile lookup (PostgreSQL via repository functions)
  3. Recent transaction history lookup (PostgreSQL via repository functions)
  4. Feature engineering (ml/features.py: compute_features)
  5. Model inference (backend/app/ml/model_loader.py: ModelLoader)
  6. Scoring decision persistence (save_scored_transaction, save_transaction)
  7. Response formatting (ScoreResponse Pydantic model)

Request timing is instrumented throughout to measure:
  - Feature engineering duration (milliseconds)
  - Model inference duration (milliseconds)
  - Total scoring duration (milliseconds)

These metrics are logged for every request to support latency monitoring and diagnosis.
"""

import logging
import time
from datetime import datetime
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.orm import Session

from backend.app.schemas import ScoreRequest, ScoreResponse, TopReason
from backend.app.db.session import get_db
from backend.app.db.repository import (
    get_user_profile,
    get_recent_transactions,
    save_transaction,
    save_scored_transaction,
    upsert_user_profile,
)
from ml.features import TransactionInput, UserProfile as UserProfileFeatures, compute_features


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["fraud-detection"])


@router.post("")
async def score_transaction(
    request_body: ScoreRequest, request: Request, db: Session = Depends(get_db)
) -> ScoreResponse:
    """
    Score a single transaction for fraud risk in real-time.

    This endpoint accepts a transaction in JSON, computes its features using the learned
    feature engineering pipeline, runs it through the trained XGBoost model, and returns
    a fraud probability and flagged decision.

    Database integration:
      - Queries user_profile table for the user's historical profile
      - Queries transactions table for recent transaction history
      - Saves the new transaction to transactions table
      - Saves the scoring decision to scored_transactions table

    Args:
        request_body: JSON request body (validated by Pydantic against ScoreRequest schema)
        request: FastAPI Request object (used to access app.state.model_loader)
        db: SQLAlchemy Session (injected by FastAPI dependency get_db)

    Returns:
        ScoreResponse with txn_id, risk_score, flagged, model_version, and top_reasons

    Raises:
        HTTPException 422: If feature computation or prediction fails (e.g., bad input data)
        HTTPException 500: If internal error (though this is caught more broadly)
    """
    try:
        # ====================================================================
        # Step 1: Look up the user's profile from PostgreSQL
        # ====================================================================
        user_id_uuid = UUID(request_body.user_id)
        db_profile = get_user_profile(db, user_id_uuid)

        # ====================================================================
        # Step 1b: If user doesn't exist, create a minimal profile in the database
        #         (This represents the "brand-new user" first transaction case)
        # ====================================================================
        if db_profile is None:
            db_profile = upsert_user_profile(
                db,
                user_id=user_id_uuid,
                profile_updates={
                    "avg_txn_amount": 0.0,
                    "std_txn_amount": 0.0,
                    "txn_count": 0,
                },
            )

        # ====================================================================
        # Step 2: Query recent transactions for velocity features
        #         Using request_body.created_at as the reference timestamp
        #         (no future leakage — only transactions BEFORE this one)
        # ====================================================================
        recent_txns_24h = get_recent_transactions(
            db,
            user_id=user_id_uuid,
            window_hours=24.0,
            before_timestamp=request_body.created_at,
        )
        recent_txn_timestamps = [txn.created_at for txn in recent_txns_24h]

        # ====================================================================
        # Step 3: Convert repository data (ORM model) to feature engineering format
        #         The repository returns DB models, but compute_features() expects
        #         the UserProfileFeatures dataclass. Map the fields:
        # ====================================================================
        profile_for_features = UserProfileFeatures(
            user_id=str(db_profile.user_id),
            avg_amount=float(db_profile.avg_txn_amount),
            std_amount=float(db_profile.std_txn_amount),
            known_device_ids=[db_profile.last_device_id] if db_profile.last_device_id else [],
            home_country=db_profile.last_country or "",
            home_latitude=float(db_profile.last_latitude) if db_profile.last_latitude else None,
            home_longitude=float(db_profile.last_longitude) if db_profile.last_longitude else None,
            recent_txn_timestamps=recent_txn_timestamps,
        )

        # ====================================================================
        # Step 4: Build a TransactionInput object from the incoming ScoreRequest
        # ====================================================================
        transaction = TransactionInput(
            transaction_id=request_body.txn_id,
            amount=request_body.amount,
            device_id=request_body.device_id,
            country=request_body.country,
            latitude=request_body.latitude,
            longitude=request_body.longitude,
            created_at=request_body.created_at,
        )

        # ====================================================================
        # TIMING POINT 1: Start feature engineering timer
        # ====================================================================
        t_feature_start = time.perf_counter()

        # ====================================================================
        # Step 5: Compute all 10 fraud-detection features
        # ====================================================================
        feature_dict = compute_features(transaction, profile_for_features)

        # ====================================================================
        # TIMING POINT 2: End feature engineering timer, start inference timer
        # ====================================================================
        t_feature_end = time.perf_counter()
        t_inference_start = time.perf_counter()

        # ====================================================================
        # Step 6: Get the ModelLoader instance from app startup
        # ====================================================================
        # This was instantiated once in main.py's lifespan, not fresh per request
        model_loader = request.app.state.model_loader

        # ====================================================================
        # Step 7: Predict fraud probability
        # ====================================================================
        risk_score = model_loader.predict_proba(feature_dict)

        # ====================================================================
        # TIMING POINT 3: End inference timer
        # ====================================================================
        t_inference_end = time.perf_counter()

        # ====================================================================
        # Step 8: Determine if flagged based on threshold
        # ====================================================================
        threshold = model_loader.get_threshold()
        flagged = risk_score >= threshold

        # ====================================================================
        # Step 9: Extract top contributing features (PLACEHOLDER)
        # ====================================================================
        # For now, use a simple heuristic: take the 2 features with the highest
        # absolute values among amount_zscore, device_mismatch, country_mismatch,
        # and geo_distance_km. This is NOT SHAP-based explainability (that's a
        # stretch goal for a much later phase when we add proper model interpretability).
        #
        # This is just good enough to show "why" the model made a decision, without
        # requiring complex ML interpretability infrastructure yet.
        #
        # In a future phase (after Phase 4), we'll replace this with:
        #   - SHAP values (model-agnostic feature importance)
        #   - Per-transaction explanation (which features moved the prediction up/down most)
        #   - Real historical impact (not just magnitude of raw features)
        candidate_features = {
            "amount_zscore": abs(feature_dict["amount_zscore"]),
            "device_mismatch": abs(feature_dict["device_mismatch"]),
            "country_mismatch": abs(feature_dict["country_mismatch"]),
            "geo_distance_km": abs(feature_dict["geo_distance_km"]),
        }

        # Sort by absolute value and take top 2
        sorted_features = sorted(
            candidate_features.items(), key=lambda x: x[1], reverse=True
        )
        top_reasons = [
            TopReason(feature=name, value=feature_dict[name])
            for name, _ in sorted_features[:2]
        ]

        # ====================================================================
        # Step 10: Save the transaction to the database
        # ====================================================================
        saved_txn = save_transaction(
            db,
            transaction_data={
                "user_id": user_id_uuid,
                "amount": request_body.amount,
                "currency": request_body.currency,
                "merchant_id": request_body.merchant_id if hasattr(request_body, "merchant_id") else None,
                "device_id": request_body.device_id,
                "ip_address": request_body.ip_address if hasattr(request_body, "ip_address") else None,
                "country": request_body.country,
                "latitude": request_body.latitude,
                "longitude": request_body.longitude,
                "created_at": request_body.created_at,
            },
        )

        # ====================================================================
        # Step 11: Save the scoring decision to the database
        # ====================================================================
        save_scored_transaction(
            db,
            scored_data={
                "txn_id": saved_txn.txn_id,
                "risk_score": float(risk_score),
                "flagged": flagged,
                "model_version": model_loader.get_model_version(),
                "feature_snapshot": feature_dict,
                "shap_values": None,  # Reserved for future use
            },
        )

        # ====================================================================
        # Compute timing durations in milliseconds
        # ====================================================================
        feature_engineering_ms = round((t_feature_end - t_feature_start) * 1000, 2)
        inference_ms = round((t_inference_end - t_inference_start) * 1000, 2)
        total_ms = round(feature_engineering_ms + inference_ms, 2)

        # ====================================================================
        # Log the successful scoring with all timing metrics
        # ====================================================================
        logger.info(
            f"txn_id={request_body.txn_id} user_id={request_body.user_id} "
            f"feature_engineering_ms={feature_engineering_ms} inference_ms={inference_ms} "
            f"total_ms={total_ms} flagged={flagged} risk_score={round(risk_score, 4)}"
        )

        # ====================================================================
        # Step 12: Build and return the response
        # ====================================================================
        return ScoreResponse(
            txn_id=request_body.txn_id,
            risk_score=risk_score,
            flagged=flagged,
            model_version=model_loader.get_model_version(),
            top_reasons=top_reasons,
        )

    except ValueError as e:
        # Feature computation or predict_proba() validation error
        # Log the error BEFORE raising the HTTPException so it appears in logs
        logger.error(f"txn_id={request_body.txn_id} ValueError: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Scoring failed: {str(e)}",
        )
    except Exception as e:
        # Any other error (shouldn't happen, but catch and log)
        # Log the error BEFORE raising the HTTPException so it appears in logs
        logger.error(
            f"txn_id={request_body.txn_id} {type(e).__name__}: {str(e)}"
        )
        raise HTTPException(
            status_code=422,
            detail=f"Scoring failed: unexpected error",
        )
