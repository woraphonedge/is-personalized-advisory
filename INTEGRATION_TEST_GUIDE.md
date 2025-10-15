# Integration Test Guide

## Prerequisites

1. Start the advisory backend:
```bash
cd /Users/home/projects/is-personalized-advisory
uv run uvicorn app.main:app --reload --port 8100
```

2. Ensure you have test data loaded (customer_id=14055)

## Test 1: Valid Rebalance with Client Style

**Test the new client_style override feature:**

```bash
curl -X POST http://localhost:8100/api/v1/rebalance \
  -H "Content-Type: application/json" \
  -d '{
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
      "cash_percent": null,
      "offshore_percent": null,
      "product_whitelist": null,
      "product_blacklist": null
    }
  }'
```

**Expected:**
- ✅ Status 200
- ✅ Response includes `healthMetrics.metrics.modelAssetAllocation`
- ✅ `modelAssetAllocation` matches "High Risk" model
- ✅ Actions array with rebalance recommendations

## Test 2: Different Client Styles

**Test Conservative style:**

```bash
curl -X POST http://localhost:8100/api/v1/rebalance \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 14055,
    "objective": {
      "objective": "risk_adjusted",
      "new_money": 0,
      "client_style": "Conservative",
      "target_alloc": {
        "Cash and Cash Equivalent": 0.3,
        "Fixed Income": 0.4,
        "Local Equity": 0.1,
        "Global Equity": 0.1,
        "Alternative": 0.1
      }
    },
    "constraints": {
      "discretionary_acceptance": 0.4,
      "client_classification": "advisory"
    }
  }'
```

**Expected:**
- ✅ `modelAssetAllocation` shows Conservative allocation (more cash/FI, less equity)

## Test 3: Error Case - Invalid Portfolio Data

**Test missing enriched data (should fail with clear error):**

```bash
curl -X POST http://localhost:8100/api/v1/rebalance \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 14055,
    "portfolio": {
      "positions": [
        {
          "productId": "INVALID",
          "srcSharecodes": null,
          "symbol": null,
          "assetClass": null,
          "marketValue": 100000
        }
      ]
    },
    "objective": {
      "objective": "risk_adjusted",
      "new_money": 0,
      "client_style": "High Risk",
      "target_alloc": {
        "Cash and Cash Equivalent": 0.2,
        "Fixed Income": 0.2,
        "Local Equity": 0.2,
        "Global Equity": 0.2,
        "Alternative": 0.2
      }
    },
    "constraints": {
      "discretionary_acceptance": 0.4,
      "client_classification": "advisory"
    }
  }'
```

**Expected:**
- ❌ Status 500
- ❌ Error message: "Portfolio data validation failed: X positions have missing enriched data"

## Test 4: Health Score API Consistency

**Compare health score from both APIs:**

```bash
# Health Score API
curl -X POST http://localhost:8100/api/v1/health-score \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 14055,
    "client_style": "High Risk",
    "target_alloc": {
      "Cash and Cash Equivalent": 0.1,
      "Fixed Income": 0.2,
      "Local Equity": 0.2,
      "Global Equity": 0.4,
      "Alternative": 0.1
    }
  }'

# Rebalance API (check healthMetrics in response)
curl -X POST http://localhost:8100/api/v1/rebalance \
  -H "Content-Type: application/json" \
  -d '{
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
      "client_classification": "advisory"
    }
  }'
```

**Expected:**
- ✅ Both APIs return same `modelAssetAllocation`
- ✅ Health scores should be similar (may differ slightly due to rebalanced portfolio)

## Test 5: Verify Logs

**Check server logs for:**

```
# Should see style override logs
DEBUG [app.utils.health_service] Temporarily overrode client_style to High Risk for customer_id=14055
DEBUG [app.utils.health_service] Restored original df_style after operation

# Should NOT see any [TEMP-DEBUG] logs (we removed them!)
```

## Test 6: Concurrent Requests

**Test thread safety of style override:**

Run multiple requests simultaneously for different customers:

```bash
# Terminal 1
curl -X POST http://localhost:8100/api/v1/rebalance \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 14055, "objective": {"client_style": "High Risk", ...}}'

# Terminal 2 (at same time)
curl -X POST http://localhost:8100/api/v1/rebalance \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 14056, "objective": {"client_style": "Conservative", ...}}'
```

**Expected:**
- ✅ No interference between requests
- ✅ Each gets correct modelAssetAllocation for their style

## Success Criteria

✅ All valid requests return 200  
✅ `modelAssetAllocation` present and correct  
✅ Error cases return clear error messages  
✅ No `[TEMP-DEBUG]` logs in output  
✅ Health score consistency between APIs  
✅ Thread-safe concurrent requests  

## Rollback Plan

If tests fail:
```bash
git checkout app/utils/rebalance_handler.py
git checkout app/utils/health_service.py
git checkout app/models.py
git checkout app/main.py
```
