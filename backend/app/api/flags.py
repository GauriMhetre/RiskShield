from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.repository import get_flagged_transactions_with_details

router = APIRouter(prefix="/flags", tags=["fraud-detection"])

@router.get("")
def get_flags(
    since: Optional[datetime] = Query(default=None),
    min_score: float = Query(default=0.0),
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
) -> list[dict]:
    """
    Get flagged transactions with their details and feature snapshot.
    """
    if limit > 200:
        raise HTTPException(status_code=422, detail="Limit cannot exceed 200")

    results = get_flagged_transactions_with_details(
        session=db,
        since=since,
        min_score=min_score,
        limit=limit,
    )

    response = []
    for scored_txn, txn in results:
        response.append({
            "txn_id": str(scored_txn.txn_id),
            "risk_score": float(scored_txn.risk_score),
            "flagged": scored_txn.flagged,
            "model_version": scored_txn.model_version,
            "scored_at": scored_txn.scored_at.isoformat() if scored_txn.scored_at else None,
            "amount": float(txn.amount),
            "user_id": str(txn.user_id),
            "country": txn.country,
            "feature_snapshot": scored_txn.feature_snapshot
        })

    return response
