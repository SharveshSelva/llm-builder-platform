import os
import pytest

# Set dummy env vars before any backend module is imported
os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("TAVILY_API_KEY", "dummy-key-for-tests")


@pytest.fixture
def anyio_backend():
    return "asyncio"
