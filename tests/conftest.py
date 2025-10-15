"""Pytest configuration and shared fixtures."""
import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires running server)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


@pytest.fixture(scope="session")
def test_customer_id():
    """Fixture for test customer ID."""
    return 14055


@pytest.fixture(scope="session")
def test_styles():
    """Fixture for valid investment styles."""
    return [
        "Bulletproof",
        "Conservative",
        "Moderate Low Risk",
        "Moderate High Risk",
        "High Risk",
        "Aggressive Growth",
        "Unwavering"
    ]
