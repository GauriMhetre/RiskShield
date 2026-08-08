"""
POST /score endpoint for real-time fraud risk scoring.

This module defines the FastAPI router for the /score endpoint, which takes a transaction
request, scores it using the trained XGBoost model, and returns a fraud risk assessment.

The route ties together:
  1. Request validation (ScoreRequest Pydantic model)
  2. User profile lookup (mock store, later PostgreSQL)
  3. Feature engineering (ml/features.py: compute_features)
  4. Model inference (backend/app/ml/model_loader.py: ModelLoader)
  5. Response formatting (ScoreResponse Pydantic model)
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request

from backend.app.schemas import ScoreRequest, ScoreResponse, TopReason
from backend.app.mock_profile_store import get_user_profile_mock
from ml.features import TransactionInput, compute_features


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["fraud-detection"])


@router.post("")
async def score_transaction(request_body: ScoreRequest, request: Request) -> ScoreResponse:
    """
    Score a single transaction for fraud risk in real-time.

    This endpoint accepts a transaction in JSON, computes its features using the learned
    feature engineering pipeline, runs it through the trained XGBoost model, and returns
    a fraud probability and flagged decision.

    Args:
        request_body: JSON request body (validated by Pydantic against ScoreRequest schema)
        request: FastAPI Request object (used to access app.state.model_loader)

    Returns:
        ScoreResponse with txn_id, risk_score, flagged, model_version, and top_reasons

    Raises:
        HTTPException 422: If feature computation or prediction fails (e.g., bad input data)
        HTTPException 500: If internal error (though this is caught more broadly)
    """
    try:
        # ====================================================================
        # Step 1: Look up the user's profile (history, baseline behavior)
        # ====================================================================
        profile = get_user_profile_mock(request_body.user_id)

        # ====================================================================
        # Step 2: Build a TransactionInput object from the incoming ScoreRequest
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
        # Step 3: Compute all 10 fraud-detection features
        # ====================================================================
        feature_dict = compute_features(transaction, profile)

        # ====================================================================
        # Step 4: Get the ModelLoader instance from app startup
        # ====================================================================
        # This was instantiated once in main.py's lifespan, not fresh per request
        model_loader = request.app.state.model_loader

        # ====================================================================
        # Step 5: Predict fraud probability
        # ====================================================================
        risk_score = model_loader.predict_proba(feature_dict)

        # ====================================================================
        # Step 6: Determine if flagged based on threshold
        # ====================================================================
        threshold = model_loader.get_threshold()
        flagged = risk_score >= threshold

        # ====================================================================
        # Step 7: Extract top contributing features (PLACEHOLDER)
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
        # Step 8: Build and return the response
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
        logger.warning(f"Scoring error for txn {request_body.txn_id}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Scoring failed: {str(e)}",
        )
    except Exception as e:
        # Any other error (shouldn't happen, but catch and log)
        logger.error(
            f"Unexpected error scoring txn {request_body.txn_id}: {type(e).__name__}: {str(e)}"
        )
        raise HTTPException(
            status_code=422,
            detail=f"Scoring failed: unexpected error",
        )
