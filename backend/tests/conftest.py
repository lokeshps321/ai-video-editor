import pytest
from app.main import app
from app.deps import get_current_user

@pytest.fixture(autouse=True)
def mock_current_user():
    """Autouse fixture to mock Clerk authentication during unit tests."""
    app.dependency_overrides[get_current_user] = lambda: {"sub": "test_user_id"}
    yield
    app.dependency_overrides.pop(get_current_user, None)
