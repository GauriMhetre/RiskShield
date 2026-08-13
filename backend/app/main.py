"""
FastAPI application for the RiskShield fraud detection system.

This module initializes the FastAPI app and defines the core routes.
It includes:
  - GET /health: Health-check endpoint (Phase 0)
  - POST /score: Real-time fraud scoring endpoint (Phase 4, Task 2)

The app uses a lifespan context manager to load the XGBoost model once at startup
and store it on app.state so all requests can reuse it without reloading from disk.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.ml.model_loader import ModelLoader
from backend.app.api import score as score_routes
from backend.app.api import flags as flags_routes


# Configure basic logging for the entire application
# Format includes timestamp, level, module name, and message
# INFO level captures startup/shutdown and request-level details;
# WARNING and ERROR messages will also be visible
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# CORS configuration: allowed origins for frontend development and deployment
# Phase 6: localhost:5173 is Vite's dev server (React development)
# Phase 8: will add deployed frontend URL (e.g., https://www.example.com)
# For now, keep this specific and narrow — never use "*" in production without strong justification
ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite React dev server (Phase 6)
    "http://localhost:3000",  # Alternative dev port if needed
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager: startup and shutdown hooks.

    Code before 'yield' runs once when the app starts.
    Code after 'yield' runs once when the app shuts down.

    This is used to:
      1. Load the trained XGBoost model once at startup
      2. Store it on app.state so all routes can access it without reloading
      3. Ensure the app fails loudly if the model files are missing
    """
    # ========================================================================
    # STARTUP: Load the model once
    # ========================================================================
    logger.info("RiskShield startup: loading fraud detection model...")
    try:
        model_loader = ModelLoader(model_version="model_v1", save_dir="ml/models")
        app.state.model_loader = model_loader
        logger.info(
            f"✓ Model loaded successfully: {model_loader.get_model_version()} "
            f"(threshold={model_loader.get_threshold():.4f})"
        )
    except FileNotFoundError as e:
        # Fail loudly and immediately if model files are missing
        logger.error(f"✗ STARTUP FAILED: Cannot load model artifact")
        logger.error(f"  {str(e)}")
        logger.error(f"  Aborting startup. Please run: python ml/train.py")
        raise  # Re-raise to prevent app from starting

    yield

    # ========================================================================
    # SHUTDOWN: Cleanup (model doesn't need explicit cleanup)
    # ========================================================================
    logger.info("RiskShield shutdown: cleaning up...")
    # No cleanup needed for model_loader; just log for visibility


# Create the FastAPI app with the lifespan context manager
app = FastAPI(
    title="RiskShield",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow frontend development server to make requests
# CORS (Cross-Origin Resource Sharing) protects users by preventing unauthorized
# cross-origin requests. Browser blocks fetch() calls from origin A to origin B
# unless origin B explicitly allows it via CORS headers.
#
# Here we allow specific origins (ALLOWED_ORIGINS) to make requests with any method/header.
# This is safer than allow_origins=["*"] because:
# 1. Specific whitelist prevents accidental exposure to random origins
# 2. Credentials/auth tokens can safely be sent (allow_credentials=True)
# 3. Easy to add production domains in Phase 8 without changing logic
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the score route(s)
app.include_router(score_routes.router)
app.include_router(flags_routes.router)


@app.get("/health")
def health_check() -> dict:
    """
    Health-check endpoint.
    
    Returns a simple JSON response indicating the service is alive.
    Real routes will later require proper error handling (try/except)
    and appropriate HTTP error responses (400, 500, etc.).
    """
    return {"status": "ok"}
