import os
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.session import get_db

load_dotenv()

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
if not TEST_DATABASE_URL:
    raise RuntimeError(
        "TEST_DATABASE_URL environment variable is not set. "
        "Please add it to your .env file to run integration tests."
    )

# Overall isolation strategy:
# These tests run against a dedicated throwaway test database (riskshield_test).
# The real development database (riskshield) is NEVER touched, preventing data pollution 
# and ensuring deterministic, reliable tests. We use FastAPI's dependency_overrides
# to transparently swap the database session used by the API routes.

@pytest.fixture(scope="session")
def test_engine():
    """
    Session-scoped test engine.
    We only need to create the connection pool once for the entire test session.
    """
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_session(test_engine):
    """
    Function-scoped database session using a transaction-rollback pattern.
    
    Why this is faster than recreating schemas:
    Instead of running heavy DROP TABLE / CREATE TABLE DDL commands before every test,
    we open a single transaction, let the test run, and then unconditionally rollback.
    This guarantees every test starts from an identical clean slate while running almost instantly.
    """
    connection = test_engine.connect()
    # Begin a non-ORM transaction
    transaction = connection.begin()
    
    # Bind session to the connection, not the engine
    TestingSessionLocal = sessionmaker(bind=connection, expire_on_commit=False)
    session = TestingSessionLocal()
    
    # Create a savepoint (nested transaction). If the application code calls session.commit(),
    # it will just commit this savepoint rather than the true outer transaction.
    session.begin_nested()
    
    # Ensure that every time a savepoint is committed, a new one is started.
    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, transaction):
        if transaction.nested and not transaction._parent.nested:
            session.begin_nested()
            
    try:
        yield session
    finally:
        session.close()
        # Rollback the outer transaction, completely wiping all changes made by the test
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="function")
def client(test_db_session):
    """
    TestClient fixture overriding get_db to inject the test database session.
    """
    def override_get_db():
        yield test_db_session
        
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    # Clear overrides to prevent leakage to other test suites
    app.dependency_overrides.clear()
