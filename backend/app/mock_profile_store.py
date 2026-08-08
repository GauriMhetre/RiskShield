"""
TEMPORARY SCAFFOLDING: Mock user profile store for Phase 4, Task 2.

⚠️ WARNING: This module is TEMPORARY and will be REPLACED in a future database-integration phase.
This is NOT production code. It's a placeholder standing in for the real PostgreSQL-backed user
profile and transaction history lookup that will be built in a later phase.

For now, it provides a hardcoded in-memory dict of fake users so the /score endpoint can be
fully wired and tested end-to-end without blocking on database infrastructure.

DO NOT treat this as production logic. When Phase 5 (Database Integration) arrives,
this entire module will be ripped out and replaced with real database queries.

The sole purpose of this file is to allow Phase 4, Task 2 to complete without database
dependencies, making it clear what gets replaced later (just swap out get_user_profile_mock()
with get_user_profile_from_db() and you're done).
"""

from datetime import datetime, timedelta
from ml.features import UserProfile


# TEMPORARY: Hardcoded mock user profiles
# In production (Phase 5), this will be queried from PostgreSQL
MOCK_USERS = {
    "user_mk01": {
        "user_id": "user_mk01",
        "avg_amount": 1000.0,
        "std_amount": 200.0,
        "known_device_ids": ["device_laptop_01", "device_phone_01"],
        "home_country": "US",
        "home_latitude": 40.7128,
        "home_longitude": -74.0060,
        "recent_txn_timestamps": [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=6),
            datetime.now() - timedelta(hours=24),
        ],
    },
    "user_mk02": {
        "user_id": "user_mk02",
        "avg_amount": 500.0,
        "std_amount": 100.0,
        "known_device_ids": ["device_phone_02"],
        "home_country": "IN",
        "home_latitude": 28.7041,
        "home_longitude": 77.1025,
        "recent_txn_timestamps": [
            datetime.now() - timedelta(hours=12),
            datetime.now() - timedelta(hours=36),
        ],
    },
    "user_mk03": {
        "user_id": "user_mk03",
        "avg_amount": 250.0,
        "std_amount": 50.0,
        "known_device_ids": ["device_tablet_01", "device_laptop_02"],
        "home_country": "SG",
        "home_latitude": 1.3521,
        "home_longitude": 103.8198,
        "recent_txn_timestamps": [
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(hours=5),
            datetime.now() - timedelta(hours=48),
        ],
    },
}


def get_user_profile_mock(user_id: str) -> UserProfile:
    """
    Look up a user's profile from the mock store, or return a "brand-new user" default.

    Args:
        user_id: Unique user identifier

    Returns:
        UserProfile object with either the mock data or brand-new-user defaults

    Why return defaults instead of raising an error?
    ================================================

    In a real fraud-scoring API, a user's first transaction ever is a normal, expected case.
    It's not an error condition — it's how every user begins. A brand-new user has no history,
    so we represent them with:
      - avg_amount, std_amount = 0 (no historical baseline)
      - known_device_ids = [] (any device is new)
      - home_country, home_lat/lon = None/"" (no established location)
      - recent_txn_timestamps = [] (no prior transactions)

    The feature engineering in ml/features.py already handles these None/empty/zero values
    gracefully (tested in Phase 2), returning sensible defaults (z-score=0, mismatch=1, etc.)
    rather than crashing. So the route can score a brand-new user end-to-end without special cases.

    Raising an error would mean the /score endpoint would fail on first-time users, which is
    unacceptable for a production API. Returning graceful defaults is correct behavior.
    """
    if user_id in MOCK_USERS:
        profile_data = MOCK_USERS[user_id]
        return UserProfile(
            user_id=profile_data["user_id"],
            avg_amount=profile_data["avg_amount"],
            std_amount=profile_data["std_amount"],
            known_device_ids=profile_data["known_device_ids"],
            home_country=profile_data["home_country"],
            home_latitude=profile_data["home_latitude"],
            home_longitude=profile_data["home_longitude"],
            recent_txn_timestamps=profile_data["recent_txn_timestamps"],
        )
    else:
        # Brand-new user: no history
        return UserProfile(
            user_id=user_id,
            avg_amount=0.0,
            std_amount=0.0,
            known_device_ids=[],
            home_country="",
            home_latitude=None,
            home_longitude=None,
            recent_txn_timestamps=[],
        )
