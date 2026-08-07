"""
FastAPI application for the RiskShield fraud detection system.

This module initializes the FastAPI app and defines the core routes.
For now, it includes only a minimal health-check endpoint.
Later phases will add /score, /flags, /feedback, and other ML-serving routes.
"""

from fastapi import FastAPI

app = FastAPI(title="RiskShield", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    """
    Health-check endpoint.
    
    Returns a simple JSON response indicating the service is alive.
    Real routes will later require proper error handling (try/except)
    and appropriate HTTP error responses (400, 500, etc.).
    """
    return {"status": "ok"}
