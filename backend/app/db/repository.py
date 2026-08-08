"""
Repository functions for RiskShield database queries.

Thin wrappers around SQLAlchemy queries. Business logic is kept minimal —
these are purely data-access functions that map database operations to Python.

All functions take a SQLAlchemy Session as their first parameter, enabling
dependency injection in FastAPI routes or flexible session management in scripts.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import select, desc, and_
from sqlalchemy.orm import Session

from backend.app.db.models import UserProfile, Transaction, ScoredTransaction


def get_user_profile(session: Session, user_id: UUID) -> Optional[UserProfile]:
    """
    Fetch a user's profile by primary key.

    Args:
        session: SQLAlchemy session
        user_id: UUID of the user to fetch

    Returns:
        UserProfile object if found, None if not found

    Why return None instead of raising an error?
    ============================================

    In a real fraud-scoring API, a user's first transaction ever is normal and expected.
    It's not an error condition — it's how every user begins (brand-new user, no history).

    This matches Phase 4's "brand new user" philosophy:
      - A new user has NO historical profile in the database yet
      - We should NOT fail the /score endpoint because a profile doesn't exist
      - Instead, we gracefully handle the missing profile downstream (return None,
        let the scoring logic handle it with sensible defaults)
      - The new user's transaction will be the *first* row in transactions table
      - A profile will be created for them later (via upsert or explicit insert)

    Returning None (vs raising) makes the error handling explicit and straightforward:
        profile = get_user_profile(session, user_id)
        if profile:
            # Use profile for feature computation
        else:
            # No profile yet (brand new user) — use defaults

    This is more Pythonic than catching exceptions, and it's what downstream code expects.
    """
    stmt = select(UserProfile).where(UserProfile.user_id == user_id)
    return session.scalar(stmt)


def get_recent_transactions(
    session: Session,
    user_id: UUID,
    window_hours: float,
    before_timestamp: datetime,
) -> list[Transaction]:
    """
    Fetch transactions for a user within a time window.

    Args:
        session: SQLAlchemy session
        user_id: UUID of the user
        window_hours: Width of the lookback window (e.g., 1.0 for 1 hour, 24.0 for 24 hours)
        before_timestamp: The "current" time reference — fetch transactions BEFORE this
                         (not including this timestamp or anything after)

    Returns:
        List of Transaction objects sorted by created_at DESC (most recent first)

    Why an explicit before_timestamp parameter instead of always using now()?
    ========================================================================

    This mirrors Phase 2's "no future leakage" principle from batch feature engineering:
      - When computing features for a specific transaction, we must use ONLY prior transactions
      - Using "now()" inside the query would include transactions that came in while
        the scoring was in progress, which is technically "future" relative to the original
        transaction being scored
      - Explicitly passing the transaction's timestamp ensures reproducibility and correctness

    Example:
        txn_timestamp = transaction.created_at  # e.g., 2026-08-08 14:30:00
        # Velocity: "transactions in the last 1 hour BEFORE this transaction"
        prior_txns = get_recent_transactions(
            session, user_id, window_hours=1.0, before_timestamp=txn_timestamp
        )
        # This ensures: created_at >= (txn_timestamp - 1 hour)
        #              AND created_at < txn_timestamp

    This is critical for:
      - Consistency: replaying the same scoring request produces identical features
      - Auditability: you can always re-derive why a transaction was scored a certain way
      - Testing: scores are reproducible, not nondeterministic based on wall-clock time
    """
    window_start = before_timestamp - timedelta(hours=window_hours)

    stmt = (
        select(Transaction)
        .where(
            and_(
                Transaction.user_id == user_id,
                Transaction.created_at >= window_start,
                Transaction.created_at < before_timestamp,
            )
        )
        .order_by(desc(Transaction.created_at))
    )

    return session.scalars(stmt).all()


def save_transaction(session: Session, transaction_data: dict) -> Transaction:
    """
    Insert a new Transaction row into the database.

    Args:
        session: SQLAlchemy session
        transaction_data: Dictionary with transaction fields (keys match Transaction columns)
                         Required: user_id, amount, currency
                         Optional: merchant_id, device_id, ip_address, country, latitude, longitude
                                  (txn_id and created_at are auto-generated/defaulted by DB)

    Returns:
        The newly created Transaction object (with auto-generated txn_id and created_at)

    Raises:
        sqlalchemy.exc.IntegrityError: if user_id doesn't exist (foreign key violation)
        Other SQLAlchemy exceptions for other constraint violations
    """
    transaction = Transaction(**transaction_data)
    session.add(transaction)
    session.commit()
    return transaction


def save_scored_transaction(session: Session, scored_data: dict) -> ScoredTransaction:
    """
    Insert a new ScoredTransaction row into the database.

    Args:
        session: SQLAlchemy session
        scored_data: Dictionary with scoring fields
                    Required: txn_id, risk_score, flagged, model_version, feature_snapshot
                    Optional: shap_values
                             (id and scored_at are auto-generated/defaulted by DB)

    Returns:
        The newly created ScoredTransaction object (with auto-generated id and scored_at)

    Raises:
        sqlalchemy.exc.IntegrityError: if txn_id doesn't exist (foreign key violation)
        Other SQLAlchemy exceptions for other constraint violations
    """
    scored_txn = ScoredTransaction(**scored_data)
    session.add(scored_txn)
    session.commit()
    return scored_txn


def upsert_user_profile(
    session: Session, user_id: UUID, profile_updates: dict
) -> UserProfile:
    """
    Update an existing UserProfile or create a new one if it doesn't exist.

    Args:
        session: SQLAlchemy session
        user_id: UUID of the user
        profile_updates: Dictionary of fields to set/update
                        (e.g., {'avg_txn_amount': 1000.0, 'txn_count': 5})

    Returns:
        The updated or newly created UserProfile object

    Behavior:
        - If a profile exists for user_id, update the given fields (others stay unchanged)
        - If no profile exists, create a new one with:
          * user_id: as provided
          * All fields from profile_updates: as provided
          * Any fields NOT in profile_updates: schema defaults (e.g., avg_txn_amount=0)

    Example:
        # Update an existing profile
        profile = upsert_user_profile(
            session,
            user_id=UUID("..."),
            profile_updates={'avg_txn_amount': 500.0, 'txn_count': 3}
        )

        # Create a new profile with minimal data
        profile = upsert_user_profile(
            session,
            user_id=UUID("..."),
            profile_updates={}  # Uses all schema defaults
        )
    """
    profile = get_user_profile(session, user_id)

    if profile:
        # Update existing profile
        for key, value in profile_updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
    else:
        # Create new profile with provided fields + schema defaults
        profile = UserProfile(user_id=user_id, **profile_updates)
        session.add(profile)

    session.commit()
    return profile
