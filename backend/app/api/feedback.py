import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.repository import save_feedback_label
from backend.app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])

class FeedbackRequest(BaseModel):
    txn_id: str = Field(..., max_length=100)
    analyst_id: str = Field(..., max_length=100)
    decision: Literal["confirmed_fraud", "false_positive"]

@router.post("")
def record_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
) -> dict:
    """
    Record analyst feedback for a flagged transaction.
    """
    try:
        feedback = save_feedback_label(
            session=db,
            txn_id=request.txn_id,
            analyst_id=request.analyst_id,
            decision=request.decision
        )
        return {
            "status": "recorded",
            "txn_id": str(feedback.txn_id),
            "decision": feedback.decision
        }
    except IntegrityError as e:
        db.rollback()
        # If the txn_id does not exist, Postgres will throw a ForeignKey violation.
        # We catch it here and return a clear 404, because the requested resource
        # (the transaction to attach feedback to) was not found. Returning a 500
        # would incorrectly imply a server bug, and a 422 implies the syntax was wrong.
        logger.error(f"Failed to record feedback for missing txn_id {request.txn_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction {request.txn_id} not found."
        )
    except ValueError as e:
        # Pydantic's Literal type should catch invalid decisions before this, 
        # but just in case, we catch the ValueError from repository and return a 422.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
