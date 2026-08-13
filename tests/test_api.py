import pytest
import uuid

def test_health_check(client):
    """Test that the API health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["status"] == "ok"


def test_score_new_user(client, test_db_session):
    """
    Test scoring a completely new user who has no history in the database.
    
    This deliberately exercises the exact risk area from Phase 4 where missing/None 
    values for brand-new users could crash the pipeline if fallback values (like std=0, avg=0) 
    were not implemented correctly.
    """
    payload = {
        "txn_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),  # Deliberately using a brand-new ID
        "amount": 100.50,
        "currency": "USD",
        "merchant_id": "merch_123",
        "ip_address": "192.168.1.1",
        "device_id": "dev_test_001",
        "country": "US",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "created_at": "2026-08-13T10:00:00Z"
    }
    
    response = client.post("/score", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0, "Risk score must be a probability between 0 and 1"
    assert isinstance(data["flagged"], bool), "Flagged must be a boolean"
    assert "top_reasons" in data


def test_score_missing_field_returns_422(client):
    """Test that missing required fields trigger a 422 Validation Error."""
    payload = {
        "txn_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        # "amount": 100.0,  # Deliberately missing
        "currency": "USD",
        "merchant_id": "merch_123",
        "ip_address": "192.168.1.1",
        "device_id": "dev_123",
        "country": "US",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "created_at": "2026-08-13T10:00:00Z"
    }
    
    response = client.post("/score", json=payload)
    assert response.status_code == 422, f"Expected 422 for missing required field, got {response.status_code}"


def test_score_negative_amount_returns_422(client):
    """Test Pydantic validation: amount cannot be negative."""
    payload = {
        "txn_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "amount": -50.0,  # Invalid negative amount
        "currency": "USD",
        "merchant_id": "merch_123",
        "ip_address": "192.168.1.1",
        "device_id": "dev_123",
        "country": "US",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "created_at": "2026-08-13T10:00:00Z"
    }
    
    response = client.post("/score", json=payload)
    assert response.status_code == 422, f"Expected 422 for negative amount, got {response.status_code}"


def test_score_creates_flag_when_high_risk(client, test_db_session):
    """
    Test that an obviously fraudulent transaction is flagged and actually written to the database.
    """
    fraud_txn_id = str(uuid.uuid4())
    fraud_user_id = str(uuid.uuid4())
    payload = {
        "txn_id": fraud_txn_id,
        "user_id": fraud_user_id,
        "amount": 999999.0,  # Massive amount
        "currency": "USD",
        "merchant_id": "merch_123",
        "ip_address": "192.168.1.1",
        "device_id": "unknown_hacker_device",
        "country": "RU",  # Cross-border mismatch likely
        "latitude": 55.7558,
        "longitude": 37.6173,
        "created_at": "2026-08-13T10:00:00Z"
    }
    
    # Send the suspicious transaction
    score_response = client.post("/score", json=payload)
    assert score_response.status_code == 200, f"Expected 200, got {score_response.status_code}: {score_response.text}"
    
    score_data = score_response.json()
    assert score_data["flagged"] is True, "Expected extreme transaction to be flagged"
    
    # Verify the flag is actually stored in the database by querying the /flags endpoint
    flags_response = client.get("/flags")
    assert flags_response.status_code == 200
    
    flags_data = flags_response.json()
    # Check if our transaction made it into the flags feed by checking the unique user_id
    flagged_user_ids = [flag["user_id"] for flag in flags_data]
    assert fraud_user_id in flagged_user_ids, "Flagged transaction was not saved to the database feed"


def test_flags_returns_list(client):
    """Test that GET /flags returns a valid JSON list."""
    response = client.get("/flags")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    assert isinstance(response.json(), list), "Expected /flags to return a list"


def test_flags_respects_limit(client, test_db_session):
    """Test that the limit query parameter correctly bounds the response length."""
    # First, insert 3 flagged transactions
    for i in range(3):
        payload = {
            "txn_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "amount": 999999.0 + i,
            "currency": "USD",
            "merchant_id": "merch_123",
            "ip_address": "192.168.1.1",
            "device_id": "hacker_device",
            "country": "XX",
            "latitude": 0.0,
            "longitude": 0.0,
            "created_at": "2026-08-13T10:00:00Z"
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        assert resp.json()["flagged"] is True
        
    # Request a limit of 2
    response = client.get("/flags?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2, f"Expected exactly 2 flags returned, got {len(data)}"


def test_flags_limit_too_high_returns_422(client):
    """Test that passing an oversized limit parameter fails validation."""
    response = client.get("/flags?limit=500")
    assert response.status_code == 422, f"Expected 422 for limit > 100, got {response.status_code}"
