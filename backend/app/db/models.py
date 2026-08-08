"""
SQLAlchemy ORM models for RiskShield.

Maps the PostgreSQL schema (db/schema.sql) to Python classes using SQLAlchemy's
declarative syntax. Field names, types, and constraints match the schema exactly.

Uses SQLAlchemy 2.x modern typed syntax:
  - Mapped[] type hints for columns
  - mapped_column() for explicit column configuration
  - UUID type from sqlalchemy.dialects.postgresql
  - JSON type for JSONB columns (Postgres binary JSON)
"""

from typing import Optional
from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Numeric,
    JSON,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, INET
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship


Base = declarative_base()


class UserProfile(Base):
    """
    User behavioral profile: aggregated statistics for feature computation.

    Columns match db/schema.sql user_profile table exactly.
    Used by feature engineering to compute velocity and amount deviation features.
    """

    __tablename__ = "user_profile"

    user_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True
    )
    avg_txn_amount: Mapped[float] = mapped_column(
        Numeric(12, 2), default=0
    )
    std_txn_amount: Mapped[float] = mapped_column(
        Numeric(12, 2), default=0
    )
    txn_count: Mapped[int] = mapped_column(Integer, default=0)
    last_device_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_country: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_latitude: Mapped[Optional[float]] = mapped_column(
        Numeric(9, 6), nullable=True
    )
    last_longitude: Mapped[Optional[float]] = mapped_column(
        Numeric(9, 6), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship: one user has many transactions
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<UserProfile user_id={self.user_id} "
            f"avg_txn_amount={self.avg_txn_amount} "
            f"txn_count={self.txn_count}>"
        )


class Transaction(Base):
    """
    Raw incoming transaction record.

    Columns match db/schema.sql transactions table exactly.
    Every transaction scored by /score endpoint is logged here.
    Foreign key ensures referential integrity with user_profile.
    """

    __tablename__ = "transactions"

    txn_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    user_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("user_profile.user_id"),
        nullable=False,
    )
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String, nullable=False)
    merchant_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    device_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(
        Numeric(9, 6), nullable=True
    )
    longitude: Mapped[Optional[float]] = mapped_column(
        Numeric(9, 6), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped[UserProfile] = relationship(back_populates="transactions")
    scored_transactions: Mapped[list["ScoredTransaction"]] = relationship(
        back_populates="transaction",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction txn_id={self.txn_id} "
            f"user_id={self.user_id} "
            f"amount={self.amount} "
            f"created_at={self.created_at}>"
        )


class ScoredTransaction(Base):
    """
    Immutable log of all scoring decisions.

    Columns match db/schema.sql scored_transactions table exactly.
    feature_snapshot stores the exact 10-element feature dict for auditability.
    shap_values reserved for future SHAP-based model interpretability.
    """

    __tablename__ = "scored_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    txn_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("transactions.txn_id"),
        nullable=False,
    )
    risk_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    feature_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)
    shap_values: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationship: many scored decisions per transaction
    transaction: Mapped[Transaction] = relationship(back_populates="scored_transactions")

    def __repr__(self) -> str:
        return (
            f"<ScoredTransaction id={self.id} "
            f"txn_id={self.txn_id} "
            f"risk_score={self.risk_score} "
            f"flagged={self.flagged}>"
        )
