"""
Pydantic models for the RiskShield fraud detection API.

These schemas define the request/response contracts for the /score endpoint.
They are used for request validation, response serialization, and automatic
OpenAPI/Swagger documentation generation.
"""

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class ScoreRequest(BaseModel):
    """
    Request schema for the /score endpoint.

    Represents a single incoming transaction to be scored for fraud risk.
    
    This schema's fields intentionally mirror the LLD's documented /score request contract.
    If field names change here, the corresponding documentation in docs/LLD.md must be
    updated to maintain consistency between API contract and documentation.
    """

    txn_id: str = Field(..., max_length=100, description="Unique transaction identifier")
    user_id: str = Field(..., max_length=100, description="Unique user identifier")
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    currency: str = Field(..., min_length=3, max_length=3, description="Currency code (e.g., USD, INR)")
    merchant_id: str = Field(..., max_length=100, description="Merchant identifier")
    device_id: str = Field(..., max_length=255, description="Device fingerprint/identifier")
    ip_address: str = Field(..., max_length=45, description="IP address of transaction origin")
    country: str = Field(..., min_length=2, max_length=2, description="Country code where transaction occurred")
    latitude: float = Field(..., ge=-90, le=90, description="Geographic latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Geographic longitude")
    created_at: datetime = Field(..., description="Transaction timestamp")

    class Config:
        """Pydantic configuration for request validation."""
        schema_extra = {
            "example": {
                "txn_id": "txn_abc123",
                "user_id": "user_xyz789",
                "amount": 500.0,
                "currency": "USD",
                "merchant_id": "merch_001",
                "device_id": "device_fingerprint_abc",
                "ip_address": "192.168.1.1",
                "country": "US",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "created_at": "2026-08-07T14:30:00Z",
            }
        }


class TopReason(BaseModel):
    """
    A single feature contributing to the fraud risk score.

    Used inside ScoreResponse to explain model predictions.
    """

    feature: str = Field(..., description="Feature name (e.g., 'amount_zscore')")
    value: float = Field(..., description="Feature value (contribution to risk)")

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "feature": "amount_zscore",
                "value": 20.0,
            }
        }


class ScoreResponse(BaseModel):
    """
    Response schema for the /score endpoint.

    Contains the fraud risk prediction for a transaction, along with model metadata
    and top contributing features for explainability.
    """

    txn_id: str = Field(..., description="Echo of the input transaction ID")
    risk_score: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    flagged: bool = Field(..., description="Whether transaction is flagged as fraud")
    model_version: str = Field(..., description="Model version used for scoring")
    top_reasons: List[TopReason] = Field(..., description="Top contributing features")

    class Config:
        """Pydantic configuration for response serialization."""
        schema_extra = {
            "example": {
                "txn_id": "txn_abc123",
                "risk_score": 0.812,
                "flagged": True,
                "model_version": "model_v1",
                "top_reasons": [
                    {"feature": "amount_zscore", "value": 20.0},
                    {"feature": "device_mismatch", "value": 1},
                ],
            }
        }
