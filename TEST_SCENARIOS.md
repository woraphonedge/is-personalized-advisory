# Rebalance API Test Scenarios

## Error Cases to Test

### 1. Portfolio Data Validation Errors

#### Scenario 1.1: Missing Enriched Data
**Test:** Send portfolio with missing `symbol`, `asset_class_name`, or `product_type_desc`

```json
{
  "customer_id": 14055,
  "portfolio": {
    "positions": [
      {
        "productId": "PROD001",
        "srcSharecodes": "TH0001010R14",
        "symbol": null,  // Missing!
        "assetClass": null,  // Missing!
        "marketValue": 100000
      }
    ]
  }
}
```

**Expected Error:**
```
ValueError: Portfolio data validation failed: 1 positions have missing enriched data.
Missing data in columns: ['asset_class_name', 'product_type_desc', 'symbol'].
Affected products: ['PROD001']
```

#### Scenario 1.2: Product Mapping Failure
**Test:** Send portfolio with invalid `src_sharecodes` that don't exist in product_mapping

```json
{
  "customer_id": 14055,
  "portfolio": {
    "positions": [
      {
        "productId": "INVALID_PROD",
        "srcSharecodes": "INVALID_CODE",
        "desk": "TRADE",
        "currency": "THB",
        "marketValue": 100000
      }
    ]
  }
}
```

**Expected Error:**
```
ValueError: Product mapping failed: 1 positions could not be mapped to product_mapping.
This indicates missing or invalid product data.
Unmapped products: [{'product_id': 'INVALID_PROD', 'src_sharecodes': 'INVALID_CODE', ...}]
```

#### Scenario 1.3: Incomplete Mapping After Merge
**Test:** Portfolio positions that partially match product_mapping but still have NA values

**Expected Error:**
```
ValueError: Product enrichment incomplete: X positions still have missing required data after mapping.
Sample rows: [...]
```

### 2. Model Asset Allocation Errors

#### Scenario 2.1: Missing Style Mapping
**Test:** Send `client_style` that doesn't exist in STYLE_MAP

```json
{
  "objective": {
    "client_style": "Invalid Style",
    "new_money": 0,
    "target_alloc": {...}
  }
}
```

**Expected:** Should fallback to "High Risk" (default in `_build_df_style`)

#### Scenario 2.2: No Model Allocation Found
**Test:** Customer with portfolio but no matching model in `ppm.df_model`

**Expected Error:**
```
ValueError: No model allocation found for customer_id=14055.
This may indicate missing style mapping or model configuration.
```

#### Scenario 2.3: Missing Model Columns
**Test:** Model allocation DataFrame missing required columns

**Expected Error:**
```
ValueError: Model allocation missing required columns: ['aa_cash_model'].
Available columns: [...]
```

### 3. Rebalancer Errors

#### Scenario 3.1: Rebalancer Execution Failure
**Test:** Invalid constraints or portfolio that causes rebalancer to fail

**Expected:** Exception logged and re-raised with context

### 4. Type Validation Errors

#### Scenario 4.1: Invalid customer_id Type
**Test:** Send non-numeric customer_id

```json
{
  "customer_id": "not_a_number"
}
```

**Expected:** Pydantic validation error (handled by FastAPI)

#### Scenario 4.2: Invalid Portfolio Structure
**Test:** Send malformed portfolio object

**Expected:** Pydantic validation error or conversion failure

## Success Cases to Test

### 1. Happy Path - Enriched Portfolio
**Test:** Send fully enriched portfolio with all required fields

**Expected:** Successful rebalance with actions and health metrics

### 2. Happy Path - Non-Enriched Portfolio  
**Test:** Send portfolio without enriched fields, rely on product_mapping

**Expected:** Successful mapping and rebalance

### 3. Client Style Override
**Test:** Send different `client_style` values

```json
{
  "objective": {
    "client_style": "High Risk",
    ...
  }
}
```

**Expected:** 
- `modelAssetAllocation` matches "High Risk" model
- Rebalance optimized for "High Risk" profile

### 4. Multiple Investment Styles
**Test Cases:**
- Conservative
- Moderate Low Risk
- Moderate High Risk
- High Risk
- Aggressive Growth
- Unwavering

**Expected:** Each returns appropriate model allocation

## Edge Cases

### 1. Empty Portfolio
**Test:** Send portfolio with no positions

**Expected:** Handle gracefully (may return empty actions)

### 2. Very Large Portfolio
**Test:** Send portfolio with 100+ positions

**Expected:** Process successfully without timeout

### 3. Null/None Values
**Test:** Send request with optional fields as null

```json
{
  "objective": {
    "client_style": null,
    "new_money": 0
  }
}
```

**Expected:** Use defaults, no errors

### 4. Concurrent Requests
**Test:** Multiple simultaneous rebalance requests for different customers

**Expected:** No interference due to `override_client_style` context manager

## Performance Tests

### 1. Response Time
**Test:** Measure API response time for typical portfolio (10-20 positions)

**Expected:** < 2 seconds

### 2. Memory Usage
**Test:** Monitor memory during rebalance with large portfolios

**Expected:** No memory leaks, proper cleanup

## Integration Tests

### 1. Health Score Consistency
**Test:** Compare health score from rebalance API vs health-score API

**Expected:** Same score for same portfolio and style

### 2. Model Allocation Consistency
**Test:** Verify `modelAssetAllocation` matches expected model from `ppm.df_model`

**Expected:** Exact match with model percentages
