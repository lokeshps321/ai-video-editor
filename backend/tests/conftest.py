from __future__ import annotations

import pytest

from app.database import init_db
from app.deps import get_current_user
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def initialize_test_database():
    # Most legacy route tests instantiate TestClient without a context manager,
    # so FastAPI's lifespan hook is not guaranteed to run for them.
    init_db()


@pytest.fixture(autouse=True)
def authenticated_test_user():
    """Run existing API tests as one deterministic Clerk user."""
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}
    yield
    app.dependency_overrides.pop(get_current_user, None)
