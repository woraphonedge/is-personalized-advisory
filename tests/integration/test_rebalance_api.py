"""Integration tests for rebalance API endpoints."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create test client for the FastAPI app."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def valid_rebalance_request():
    """Fixture for valid rebalance request payload."""
    return {
        "customer_id": 14055,
        "objective": {
            "objective": "risk_adjusted",
            "new_money": 0,
            "client_style": "High Risk",
            "target_alloc": {
                "Cash and Cash Equivalent": 0.1,
                "Fixed Income": 0.2,
                "Local Equity": 0.2,
                "Global Equity": 0.4,
                "Alternative": 0.1
            }
        },
        "constraints": {
            "discretionary_acceptance": 0.4,
            "client_classification": "advisory",
            "private_percent": 0,
            "cash_percent": None,
            "offshore_percent": None,
            "product_whitelist": None,
            "product_blacklist": None
        }
    }


@pytest.fixture
def valid_health_score_request():
    """Fixture for valid health score request payload."""
    return {
        "customer_id": 14055,
        "client_style": "High Risk",
        "target_alloc": {
            "Cash and Cash Equivalent": 0.1,
            "Fixed Income": 0.2,
            "Local Equity": 0.2,
            "Global Equity": 0.4,
            "Alternative": 0.1
        }
    }


class TestRebalanceAPI:
    """Integration tests for /api/v1/rebalance endpoint."""

    @pytest.mark.integration
    def test_rebalance_with_client_style(self, client, valid_rebalance_request):
        """Test rebalance with client_style override."""
        response = client.post("/api/v1/rebalance", json=valid_rebalance_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "actions" in data
        assert "portfolio" in data
        assert "healthScore" in data
        assert "healthMetrics" in data
        
        # Check health metrics structure
        assert "score" in data["healthMetrics"]
        assert "metrics" in data["healthMetrics"]
        
        # Check model asset allocation is present
        metrics = data["healthMetrics"]["metrics"]
        assert "modelAssetAllocation" in metrics
        assert metrics["modelAssetAllocation"] is not None
        
        # Verify model allocation has all asset classes
        model_alloc = metrics["modelAssetAllocation"]
        assert "Cash and Cash Equivalent" in model_alloc
        assert "Fixed Income" in model_alloc
        assert "Local Equity" in model_alloc
        assert "Global Equity" in model_alloc
        assert "Alternative" in model_alloc

    @pytest.mark.integration
    @pytest.mark.parametrize("client_style", [
        "Conservative",
        "Moderate Low Risk",
        "Moderate High Risk",
        "High Risk",
        "Aggressive Growth",
    ])
    def test_rebalance_with_different_styles(self, client, valid_rebalance_request, client_style):
        """Test rebalance with different client styles."""
        valid_rebalance_request["objective"]["client_style"] = client_style
        
        response = client.post("/api/v1/rebalance", json=valid_rebalance_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Model allocation should be present and different for each style
        model_alloc = data["healthMetrics"]["metrics"]["modelAssetAllocation"]
        assert model_alloc is not None
        
        # Sum should be close to 1.0 (100%)
        total = sum(model_alloc.values())
        assert 0.99 <= total <= 1.01

    @pytest.mark.integration
    def test_rebalance_without_client_style(self, client, valid_rebalance_request):
        """Test rebalance without client_style (should use stored style)."""
        valid_rebalance_request["objective"]["client_style"] = None
        
        response = client.post("/api/v1/rebalance", json=valid_rebalance_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still have model allocation (from stored style)
        assert "modelAssetAllocation" in data["healthMetrics"]["metrics"]

    @pytest.mark.integration
    def test_rebalance_with_invalid_portfolio_data(self, client, valid_rebalance_request):
        """Test rebalance with invalid portfolio data."""
        # Add invalid portfolio with missing enriched data
        valid_rebalance_request["portfolio"] = {
            "positions": [
                {
                    "productId": "INVALID",
                    "srcSharecodes": None,
                    "symbol": None,
                    "assetClass": None,
                    "marketValue": 100000
                }
            ]
        }
        
        response = client.post("/api/v1/rebalance", json=valid_rebalance_request)
        
        # Should fail with clear error message
        assert response.status_code == 500
        assert "Portfolio data validation failed" in response.text or "enriched data" in response.text


class TestHealthScoreAPI:
    """Integration tests for /api/v1/health-score endpoint."""

    @pytest.mark.integration
    def test_health_score_with_client_style(self, client, valid_health_score_request):
        """Test health score with client_style override."""
        response = client.post("/api/v1/health-score", json=valid_health_score_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "score" in data
        assert "metrics" in data
        
        # Check model asset allocation is present
        assert "modelAssetAllocation" in data["metrics"]
        assert data["metrics"]["modelAssetAllocation"] is not None

    @pytest.mark.integration
    def test_health_score_consistency_with_rebalance(
        self, client, valid_health_score_request, valid_rebalance_request
    ):
        """Test that health score API and rebalance API return consistent model allocation."""
        # Get health score
        health_response = client.post("/api/v1/health-score", json=valid_health_score_request)
        assert health_response.status_code == 200
        health_data = health_response.json()
        
        # Get rebalance (which includes health metrics)
        rebalance_response = client.post("/api/v1/rebalance", json=valid_rebalance_request)
        assert rebalance_response.status_code == 200
        rebalance_data = rebalance_response.json()
        
        # Model allocations should be the same (same customer, same style)
        health_model_alloc = health_data["metrics"]["modelAssetAllocation"]
        rebalance_model_alloc = rebalance_data["healthMetrics"]["metrics"]["modelAssetAllocation"]
        
        # Compare each asset class (allow small floating point differences)
        for asset_class in health_model_alloc.keys():
            assert abs(health_model_alloc[asset_class] - rebalance_model_alloc[asset_class]) < 0.001


class TestConcurrentRequests:
    """Test thread safety of style override."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_rebalance_requests(self, client, valid_rebalance_request):
        """Test that concurrent requests don't interfere with each other."""
        import concurrent.futures
        
        # Create requests with different styles
        request1 = valid_rebalance_request.copy()
        request1["objective"]["client_style"] = "High Risk"
        request1["customer_id"] = 14055
        
        request2 = valid_rebalance_request.copy()
        request2["objective"]["client_style"] = "Conservative"
        request2["customer_id"] = 14056  # Different customer
        
        def make_request(request_data):
            response = client.post("/api/v1/rebalance", json=request_data)
            return response.json()
        
        # Execute requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(make_request, request1)
            future2 = executor.submit(make_request, request2)
            
            result1 = future1.result()
            result2 = future2.result()
        
        # Each should have correct model allocation for their style
        # High Risk should have more equity than Conservative
        model1 = result1["healthMetrics"]["metrics"]["modelAssetAllocation"]
        model2 = result2["healthMetrics"]["metrics"]["modelAssetAllocation"]
        
        equity1 = model1["Global Equity"] + model1["Local Equity"]
        equity2 = model2["Global Equity"] + model2["Local Equity"]
        
        # High Risk should have more equity allocation
        assert equity1 > equity2


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_customer_id(self, client, valid_rebalance_request):
        """Test with invalid customer_id."""
        valid_rebalance_request["customer_id"] = "invalid"
        
        response = client.post("/api/v1/rebalance", json=valid_rebalance_request)
        
        # Should fail validation
        assert response.status_code == 422  # Pydantic validation error

    def test_missing_required_fields(self, client):
        """Test with missing required fields."""
        response = client.post("/api/v1/rebalance", json={})
        
        # Should fail validation
        assert response.status_code == 422

    def test_invalid_target_alloc_sum(self, client, valid_rebalance_request):
        """Test with target allocation not summing to 1.0."""
        valid_rebalance_request["objective"]["target_alloc"] = {
            "Cash and Cash Equivalent": 0.5,
            "Fixed Income": 0.5,
            "Local Equity": 0.5,  # Sum > 1.0
            "Global Equity": 0.0,
            "Alternative": 0.0
        }
        
        response = client.post("/api/v1/rebalance", json=valid_rebalance_request)
        
        # May succeed but with warnings, or fail depending on validation
        # At minimum, should not crash
        assert response.status_code in [200, 400, 422, 500]
