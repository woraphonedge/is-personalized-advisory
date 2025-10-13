#!/usr/bin/env python3
"""
Test script to verify the fix for frontend payload handling.
This simulates the frontend sending a payload with clean symbols (without "(in lend)" suffixes).
"""

import json
import sys

import requests

BASE_URL = "http://localhost:8100"
CUSTOMER_ID = 14055


def test_frontend_payload():
    """Test rebalance with frontend-style payload (clean symbols)."""

    # This mimics the frontend payload structure you showed
    frontend_payload = {
        "customer_id": 14055,
        "objective": {
            "objective": "risk_adjusted",
            "new_money": 1000000,
            "target_alloc": {
                "Cash and Cash Equivalent": 0.05,
                "Fixed Income": 0.3,
                "Global Equity": 0.45,
                "Local Equity": 0.1,
                "Alternative": 0.1,
                "Asset Allocation": 0,
            },
            "client_style": "Moderate High Risk",
        },
        "constraints": {
            "do_not_sell": [],
            "client_classification": "UI",
            "discretionary_acceptance": 0.2,
            "private_percent": 0,
            "cash_percent": None,
            "offshore_percent": None,
            "product_whitelist": [],
            "product_blacklist": [],
        },
        "portfolio": {
            "positions": [
                {
                    "id": 1,
                    "productId": "C00020399",
                    "desk": "GIS",
                    "portType": "L",
                    "currency": "THB",
                    "productTypeDesc": "Cash",
                    "coveragePrdtype": "N/A",
                    "assetClass": "Cash and Cash Equivalent",
                    "assetSubClass": "Cash Offshore",
                    "symbol": "CASH",
                    "srcSharecodes": "GIS THB Collateral",  # Frontend now sends proper srcSharecodes
                    "productDisplayName": "",
                    "unitBal": 0,
                    "unitPriceThb": 1,
                    "unitCostThb": 1,
                    "marketValue": 211869.68,
                    "expectedReturn": 0.048,
                    "volatility": 0,
                    "flagTopPick": "Not Top-Pick",
                    "isMonitored": True,
                    "isRiskyAsset": False,
                    "isCoverage": True,
                    "esSellList": None,
                    "esCorePort": False,
                },
                {
                    "id": 2,
                    "productId": "S00088788",
                    "desk": "TRADE",
                    "portType": "L",
                    "currency": "THB",
                    "productTypeDesc": "Listed Securities",
                    "coveragePrdtype": "LOCAL_STOCK",
                    "assetClass": "Local Equity",
                    "assetSubClass": "Local Stock",
                    "symbol": "SCC",
                    "srcSharecodes": "SCC",
                    "productDisplayName": "",
                    "unitBal": 0,
                    "unitPriceThb": 1,
                    "unitCostThb": 1,
                    "marketValue": 6486000.0,
                    "expectedReturn": 0.042,
                    "volatility": 0,
                    "flagTopPick": "Not Top-Pick",
                    "isMonitored": True,
                    "isRiskyAsset": True,
                    "isCoverage": False,
                    "esSellList": "Sell",
                    "esCorePort": False,
                },
                {
                    "id": 3,
                    "productId": "S00085703",
                    "desk": "SIDEB",  # This should trigger the "(in lend)" suffix logic
                    "portType": "L",
                    "currency": "THB",
                    "productTypeDesc": "Listed Securities",
                    "coveragePrdtype": "LOCAL_STOCK",
                    "assetClass": "Local Equity",
                    "assetSubClass": "Local Stock",
                    "symbol": "SCGP",
                    "srcSharecodes": "SCGP (in lend)",  # Frontend now sends complete srcSharecodes
                    "productDisplayName": "",
                    "unitBal": 0,
                    "unitPriceThb": 1,
                    "unitCostThb": 1,
                    "marketValue": 117158.4,
                    "expectedReturn": 0.042,
                    "volatility": 0,
                    "flagTopPick": "Not Top-Pick",
                    "isMonitored": True,
                    "isRiskyAsset": True,
                    "isCoverage": False,
                    "esSellList": "Sell",
                    "esCorePort": False,
                },
                {
                    "id": 4,
                    "productId": "S00080017",
                    "desk": "SIDEB",  # This should trigger the "(in lend)" suffix logic
                    "portType": "L",
                    "currency": "THB",
                    "productTypeDesc": "Listed Securities",
                    "coveragePrdtype": "LOCAL_STOCK",
                    "assetClass": "Local Equity",
                    "assetSubClass": "Local Stock",
                    "symbol": "KBANK",
                    "srcSharecodes": "KBANK (in lend)",  # Frontend now sends complete srcSharecodes
                    "productDisplayName": "",
                    "unitBal": 0,
                    "unitPriceThb": 1,
                    "unitCostThb": 1,
                    "marketValue": 91471750.0,
                    "expectedReturn": 0.042,
                    "volatility": 0,
                    "flagTopPick": "Not Top-Pick",
                    "isMonitored": True,
                    "isRiskyAsset": True,
                    "isCoverage": False,
                    "esSellList": None,
                    "esCorePort": False,
                },
            ]
        },
        "style": {
            "INVESTMENT_STYLE_AUMX": "Moderate High Risk",
            "AS_OF_DATE": "2025-09-30",
            "CUSTOMER_ID": 14055,
        },
    }

    url = f"{BASE_URL}/api/v1/rebalance"

    print("🧪 Testing frontend payload handling...")
    print(f"URL: {url}")
    print(f"Payload positions: {len(frontend_payload['portfolio']['positions'])}")

    try:
        response = requests.post(url, json=frontend_payload)

        print("\nResponse status: {}".format(response.status_code))

        if response.status_code != 200:
            print("❌ Request failed")
            print("Response body: {}".format(response.text))
            return False

        result = response.json()
        print("✅ Request succeeded")
        print(f"Actions count: {len(result.get('actions', []))}")
        print(
            f"New portfolio positions: {len(result.get('portfolio', {}).get('positions', []))}"
        )
        print(f"Health score: {result.get('health_score', 'N/A')}")

        # Save results for inspection
        with open("frontend_test_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("✅ Results saved to frontend_test_result.json")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return False


def main():
    print("🚀 Testing frontend payload fix")
    print("=" * 50)

    success = test_frontend_payload()

    print("\n" + "=" * 50)
    if success:
        print("✅ Frontend payload test completed successfully")
        print("Check the FastAPI logs to see if NA columns issue is resolved")
    else:
        print("❌ Frontend payload test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
