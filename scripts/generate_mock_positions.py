#!/usr/bin/env python
"""Generate mock positions JSON for a given customer_id from parquet.

Usage (from project root):

    uv run python scripts/generate_mock_positions.py --customer-id 39597

This reads app/data/portfolios_client_out_enriched_2025-10-31.parquet,
filters by the provided customer_id, maps columns into the Position
model's frontend aliases (productId, assetClass, etc.), and writes a
JSON file under app/data/mock_positions_<customer_id>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _opt_str(value: object) -> str | None:
    """Convert pandas NA/None to None, otherwise cast to str."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mock positions JSON for a specific customer_id",
    )
    parser.add_argument(
        "--customer-id",
        type=int,
        default=39597,
        help="customer_id to export positions for (default: 39597)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    parquet_path = (
        project_root
        / "app"
        / "data"
        / "portfolios_client_out_enriched_2025-10-31.parquet"
    )
    out_path = project_root / "app" / "data" / f"mock_positions_{args.customer_id}.json"

    df = pd.read_parquet(parquet_path)
    df = df[df["customer_id"] == args.customer_id]

    positions: list[dict] = []

    for _, row in df.iterrows():
        value = float(row["value"])
        unit_bal = 1.0
        unit_price = value
        unit_cost = value

        payload = {
            "productId": _opt_str(row["product_id"]),
            "desk": _opt_str(row.get("desk")),
            "portType": _opt_str(row.get("port_type")),
            "symbol": _opt_str(row["symbol"]),
            "srcSharecodes": _opt_str(row.get("src_sharecodes")),
            "assetClass": _opt_str(row["asset_class_name"]),
            "assetSubClass": _opt_str(row.get("pp_asset_sub_class")),
            "unitBal": unit_bal,
            "unitPriceThb": unit_price,
            "unitCostThb": unit_cost,
            "marketValue": value,
            "currency": _opt_str(row["currency"]),
            "expectedReturn": float(row["expected_return"]),
            "expectedIncomeYield": None,
            "volatility": 0.0,
            "productTypeDesc": _opt_str(row["product_type_desc"]),
            "coveragePrdtype": _opt_str(row.get("coverage_prdtype")),
            "isMonitored": True,
            "isRiskyAsset": bool(row["is_risky_asset"]),
            "isCoverage": bool(row["is_coverage"]),
            "productDisplayName": _opt_str(row.get("product_display_name")),
            "esCorePort": bool(row["es_core_port"]),
            "esSellList": _opt_str(row.get("es_sell_list")),
            "flagTopPick": _opt_str(row["flag_top_pick"]),
            "flagTaxSaving": _opt_str(row.get("flag_tax_saving")),
            "exposures": None,
            "posDate": _opt_str(row.get("as_of_date")),
        }

        positions.append(payload)

    out_path.write_text(json.dumps(positions, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(positions)} positions for customer_id={args.customer_id} to {out_path}"
    )


if __name__ == "__main__":
    main()
