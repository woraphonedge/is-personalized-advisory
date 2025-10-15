# Test Suite

## Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── unit/                          # Unit tests (fast, no external dependencies)
│   ├── __init__.py
│   └── test_rebalance_handler.py  # Tests for rebalance_handler module
└── integration/                   # Integration tests (requires running server)
    ├── __init__.py
    └── test_rebalance_api.py      # API endpoint tests
```

## Running Tests

### Install Dependencies

```bash
# Install dev dependencies including pytest
uv sync --dev
```

### Run All Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html --cov-report=term
```

### Run Specific Test Types

```bash
# Run only unit tests (fast)
uv run pytest -m unit

# Run only integration tests (requires server)
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"
```

### Run Specific Test Files

```bash
# Run unit tests only
uv run pytest tests/unit/

# Run integration tests only
uv run pytest tests/integration/

# Run specific test file
uv run pytest tests/unit/test_rebalance_handler.py

# Run specific test class
uv run pytest tests/unit/test_rebalance_handler.py::TestBuildDfStyle

# Run specific test
uv run pytest tests/unit/test_rebalance_handler.py::TestBuildDfStyle::test_valid_string_style
```

### Run with Verbose Output

```bash
# Show test names and results
uv run pytest -v

# Show print statements
uv run pytest -s

# Show detailed failure info
uv run pytest -vv
```

## Integration Tests

Integration tests require the FastAPI server to be running.

### Start Server

```bash
# Terminal 1: Start server
cd /Users/home/projects/is-personalized-advisory
uv run uvicorn app.main:app --reload --port 8100
```

### Run Integration Tests

```bash
# Terminal 2: Run integration tests
uv run pytest tests/integration/ -m integration
```

## Test Coverage

### Generate Coverage Report

```bash
# Generate HTML coverage report
uv run pytest --cov=app --cov-report=html

# Open report in browser
open htmlcov/index.html
```

### Coverage Goals

- **Unit Tests:** > 80% coverage for refactored modules
- **Integration Tests:** Cover all API endpoints
- **Error Cases:** Test all error scenarios from TEST_SCENARIOS.md

## Writing Tests

### Unit Test Example

```python
import pytest
from app.utils.rebalance_handler import _build_df_style

class TestMyFunction:
    def test_valid_input(self):
        """Test with valid input."""
        result = _build_df_style(14055, "High Risk")
        assert result is not None
    
    def test_invalid_input(self):
        """Test with invalid input."""
        with pytest.raises(ValueError):
            _build_df_style(0, "High Risk")
    
    @pytest.mark.parametrize("style,expected", [
        ("High Risk", "High Risk"),
        ("Conservative", "Conservative"),
    ])
    def test_multiple_styles(self, style, expected):
        """Test with multiple styles."""
        df = _build_df_style(14055, style)
        assert df.iloc[0]["portpop_style"] == expected
```

### Integration Test Example

```python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)

@pytest.mark.integration
class TestAPI:
    def test_endpoint(self, client):
        """Test API endpoint."""
        response = client.post("/api/v1/rebalance", json={...})
        assert response.status_code == 200
```

## Continuous Integration

### GitHub Actions (example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev
      - name: Run unit tests
        run: uv run pytest tests/unit/ -m unit
      - name: Run integration tests
        run: |
          uv run uvicorn app.main:app --port 8100 &
          sleep 5
          uv run pytest tests/integration/ -m integration
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests requiring server
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.parametrize` - Run test with multiple inputs

## Fixtures

Available fixtures (see `conftest.py`):

- `test_customer_id` - Test customer ID (14055)
- `test_styles` - List of valid investment styles
- `client` - FastAPI test client (integration tests)
- `valid_rebalance_request` - Valid rebalance request payload
- `valid_health_score_request` - Valid health score request payload

## Troubleshooting

### Tests Fail with Import Errors

```bash
# Ensure you're in the project root
cd /Users/home/projects/is-personalized-advisory

# Install in development mode
uv sync --dev
```

### Integration Tests Fail

```bash
# Check if server is running
curl http://localhost:8100/health

# Check if test data is loaded
curl http://localhost:8100/api/v1/portfolio/14055
```

### Slow Tests

```bash
# Skip slow tests
uv run pytest -m "not slow"

# Run only fast unit tests
uv run pytest tests/unit/ -m unit
```

## Next Steps

1. ✅ Run unit tests: `uv run pytest tests/unit/`
2. ⏳ Start server and run integration tests
3. ⏳ Add more test coverage for remaining functions
4. ⏳ Set up CI/CD pipeline
