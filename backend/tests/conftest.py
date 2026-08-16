from __future__ import annotations

import os

import pytest

# IndicXlit (offline neural transliteration) always "looks" available once the
# package is installed, but loading it the first time downloads/initializes a
# large local model -- multi-second even when cached, a real download attempt
# (against no guaranteed network) when it isn't. Any test that reaches
# transliterate_words()/transliterate_text() without mocking would otherwise
# pay that cost once per test process and slow/flake the whole suite. Tests
# that want to exercise this path stub `_indicxlit_engine` directly instead.
os.environ.setdefault("TRANSLITERATE_USE_INDICXLIT", "false")

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
