"""
Database session management for RiskShield.

This module sets up SQLAlchemy's engine and session factory, reading database
configuration from environment variables. It provides a standard FastAPI
dependency-injection pattern for database access across routes.

Key concepts:
  - Engine: A connection pool and dialect configuration. Create once per app.
  - SessionLocal/sessionmaker: A factory that produces new sessions.
    Each session is a lightweight context manager for a transaction.
    Create one sessionmaker per app, then call it to get fresh sessions per request.
  - get_db(): A generator that yields a session, ensuring cleanup in a finally block.
    This is the standard FastAPI dependency pattern, ready for @app.get()/@app.post() routes.
"""

import os
import logging
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session


logger = logging.getLogger(__name__)

# Load .env file (if present) — non-committed local configuration
load_dotenv()

# Read DATABASE_URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set. "
        "Please create a .env file in the project root with: "
        "DATABASE_URL=postgresql://user:password@localhost/dbname"
    )

# Create the SQLAlchemy engine
# echo=False suppresses SQL statement logging (set to True for debugging)
# pool_pre_ping=True ensures the connection is still alive before using it (prevents stale connections)
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

# Create a sessionmaker bound to this engine
# expire_on_commit=False lets objects remain accessible after commit (not automatically refreshed)
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency: yields a database session for the lifetime of a request.

    Usage in a route:
        from backend.app.db.session import get_db
        from sqlalchemy.orm import Session

        @app.get("/endpoint")
        def my_route(db: Session = Depends(get_db)):
            user = db.query(User).first()
            return user

    The session is automatically closed after the request, whether successful or not.
    The finally block ensures cleanup even if an exception occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
