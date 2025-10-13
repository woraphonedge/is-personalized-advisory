#!/usr/bin/env python3
"""
Test script to debug the rebalance API issue.

This script:
1. Fetches a portfolio from the GET /api/v1/portfolio/{customer_id} endpoint
2. Saves the portfolio to a JSON file
3. Calls the POST /api/v1/rebalance endpoint using the saved portfolio as payload
4. Compares the behavior with the notebook workflow
"""

import json
import sys
from typing import Any, Dict

import requests

# Configuration
BASE_URL = "http://localhost:8100"
CUSTOMER_ID = 14055
OUTPUT_FILE = "portfolio_data.json"


def fetch_portfolio(customer_id: int) -> Dict[str, Any]:
    """Fetch portfolio from the API endpoint."""
    url = f"{BASE_URL}/api/v1/portfolio/{customer_id}"

    print(f"Fetching portfolio for customer {customer_id}...")
    print(f"URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        portfolio_data = response.json()
        print(
            f"✅ Successfully fetched portfolio with {len(portfolio_data.get('positions', []))} positions"
        )

        return portfolio_data

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching portfolio: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        sys.exit(1)


def save_portfolio(portfolio_data: Dict[str, Any], filename: str) -> None:
    """Save portfolio data to JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(portfolio_data, f, indent=2, default=str)
        print(f"✅ Portfolio saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving portfolio: {e}")
        sys.exit(1)


def load_portfolio(filename: str) -> Dict[str, Any]:
    """Load portfolio data from JSON file."""
    try:
        with open(filename, "r") as f:
            portfolio_data = json.load(f)
        print(f"✅ Portfolio loaded from {filename}")
        return portfolio_data
    except Exception as e:
        print(f"❌ Error loading portfolio: {e}")
        sys.exit(1)


def call_rebalance_api(
    portfolio_data: Dict[str, Any], customer_id: int
) -> Dict[str, Any]:
    """Call the rebalance API with the portfolio data."""
    url = f"{BASE_URL}/api/v1/rebalance"

    # Prepare the rebalance request payload
    rebalance_request = {
        "customer_id": customer_id,
        "objective": {
            "objective": "risk_adjusted",
            "new_money": 1000000,
            "target_alloc": {
                "Global Equity": 0.4,
                "Local Equity": 0.2,
                "Fixed Income": 0.3,
                "Cash and Cash Equivalent": 0.1,
            },
        },
        "constraints": {
            "discretionary_acceptance": 0.2,
            "client_classification": "UI",
            "private_percent": 0.0,
            "cash_percent": None,
            "offshore_percent": None,
            "product_restriction": None,
            "product_whitelist": ["KKP", "PTTEP"],
            "product_blacklist": ["KKP GNP", "K-GSELECTU-A(A)"],
        },
        "portfolio": portfolio_data,
        "style": {
            "INVESTMENT_STYLE_AUMX": "Moderate High Risk",
            "AS_OF_DATE": "2025-09-30",
            "CUSTOMER_ID": customer_id,
        },
    }

    print("\nCalling rebalance API...")
    print(f"URL: {url}")
    print(f"Portfolio positions count: {len(portfolio_data.get('positions', []))}")

    # Show sample of portfolio positions for debugging
    positions = portfolio_data.get("positions", [])
    if positions:
        print("\nSample position data:")
        sample_pos = positions[0]
        for key, value in sample_pos.items():
            print(f"  {key}: {value}")

    try:
        response = requests.post(url, json=rebalance_request)

        print(f"\nResponse status: {response.status_code}")

        if response.status_code != 200:
            print("❌ Rebalance request failed")
            print(f"Response body: {response.text}")
            return {"error": response.text, "status_code": response.status_code}

        result = response.json()
        print("✅ Rebalance completed successfully")
        print(f"Actions count: {len(result.get('actions', []))}")
        print(
            f"New portfolio positions: {len(result.get('portfolio', {}).get('positions', []))}"
        )
        print(f"Health score: {result.get('health_score', 'N/A')}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling rebalance API: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return {"error": str(e)}


def main():
    """Main execution function."""
    print("🚀 Starting rebalance API test script")
    print("=" * 50)

    # Step 1: Fetch portfolio
    portfolio_data = fetch_portfolio(CUSTOMER_ID)

    # Step 2: Save portfolio
    save_portfolio(portfolio_data, OUTPUT_FILE)

    # Step 3: Load portfolio (to simulate real usage)
    loaded_portfolio = load_portfolio(OUTPUT_FILE)

    # Step 4: Call rebalance API
    rebalance_result = call_rebalance_api(loaded_portfolio, CUSTOMER_ID)

    # Step 5: Save results
    results_file = f"rebalance_result_{CUSTOMER_ID}.json"
    try:
        with open(results_file, "w") as f:
            json.dump(rebalance_result, f, indent=2, default=str)
        print(f"✅ Rebalance results saved to {results_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

    print("\n" + "=" * 50)
    print("🏁 Test script completed")

    if "error" in rebalance_result:
        print("❌ Test failed - check logs above")
        sys.exit(1)
    else:
        print("✅ Test completed successfully")


if __name__ == "__main__":
    main()
