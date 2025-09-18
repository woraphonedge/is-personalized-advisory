#!/usr/bin/env python3
"""
Simple client to call the FastAPI endpoint /api/v2/rebalance with a mocked
RebalanceRequest payload. Run with:

  uv run python scripts/call_rebalance_v2.py

Optionally override the API base URL with the API_BASE_URL environment variable.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date
from typing import Any, Dict


def post_json(url: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any] | str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8")
            try:
                return status, json.loads(body)
            except json.JSONDecodeError:
                return status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except Exception as e:  # noqa: BLE001
        return 0, f"Request failed: {e}"


def build_mock_request() -> Dict[str, Any]:
    # Minimal realistic mock that satisfies the pydantic model in models.RebalanceRequest
    today = date.today().isoformat()

    # Mock positions (camelCase to match frontend aliases)
    portfolio = {
        "positions": [
            {
                "posDate": today,
                "productId": "CASH",
                "symbol": "CASH",
                "assetClass": "Cash and Cash Equivalent",
                "assetSubClass": "Cash THB",
                "unitBal": 0.0,
                "unitPriceThb": 1.0,
                "unitCostThb": 1.0,
                "marketValue": 200_000.0,
                "expectedReturn": 0.01,
                "expectedIncomeYield": 0.0,
                "volatility": 0.00,
                "isMonitored": True,
                "exposures": None,
            },
            {
                "posDate": today,
                "productId": "GLB_EQTY_FUND",
                "symbol": "GLB_EQTY_FUND",
                "assetClass": "Global Equity",
                "assetSubClass": "Equity",
                "unitBal": 100.0,
                "unitPriceThb": 800.0,
                "unitCostThb": 700.0,
                "marketValue": 80_000.0,
                "expectedReturn": 0.08,
                "expectedIncomeYield": 0.0,
                "volatility": 0.18,
                "isMonitored": True,
                "exposures": None,
            },
            {
                "posDate": today,
                "productId": "TH_BOND_FUND",
                "symbol": "TH_BOND_FUND",
                "assetClass": "Fixed Income",
                "assetSubClass": "Global Bond",
                "unitBal": 500.0,
                "unitPriceThb": 200.0,
                "unitCostThb": 200.0,
                "marketValue": 100_000.0,
                "expectedReturn": 0.03,
                "expectedIncomeYield": 0.02,
                "volatility": 0.05,
                "isMonitored": True,
                "exposures": None,
            },
        ]
    }

    # Target allocation across asset classes used by health score
    target_alloc = {
        "Global Equity": 0.40,
        "Local Equity": 0.10,
        "Fixed Income": 0.35,
        "Cash and Cash Equivalent": 0.10,
        "Alternative": 0.05,
    }

    # Constraints/settings for v2 rebalancer
    constraints = {
        "discretionary_percent": 0.5,
        "private_percent": 0.0,
        "cash_percent": None,
        "offshore_percent": None,
        "product_restriction": ["ILLQ_FUND"],
        "discretionary_acceptance": 0.4,
    }

    # Style payload expected by v2. Only AS_OF_DATE and CUSTOMER_ID are required by
    # create_portfolio_id mapping in main.rebalance_v2; we include some optional
    # descriptive fields for realism.
    style = [
        {
            "AS_OF_DATE": today,
            "CUSTOMER_ID": "cust-001",
            "PORT_INVESTMENT_STYLE": "Balanced",
            "NOTES": "Mock style row for testing",
        }
    ]

    payload: Dict[str, Any] = {
        "customer_id": "cust-001",
        "objective": {
            "objective": "risk_adjusted",
            "new_money": 0.0,
            "target_alloc": target_alloc,
        },
        "constraints": constraints,
        "portfolio": portfolio,
        "style": style,
    }
    return payload


def main() -> None:
    base = os.environ.get("API_BASE_URL", "http://127.0.0.1:8100")
    url = f"{base.rstrip('/')}/api/v2/rebalance"
    payload = build_mock_request()

    print(f"POST {url}")
    print("Payload:\n" + json.dumps(payload, indent=2))

    status, body = post_json(url, payload)
    print(f"\nStatus: {status}")
    if isinstance(body, (dict, list)):
        print("Response:\n" + json.dumps(body, indent=2))
    else:
        print("Response:\n" + str(body))


if __name__ == "__main__":
    main()
