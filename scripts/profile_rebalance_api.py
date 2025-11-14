#!/usr/bin/env python3
"""
Comprehensive profiling script for the rebalance API to identify CPU bottlenecks.

Usage:
    python scripts/profile_rebalance_api.py [--output-dir ./profiling_results]

Requirements:
    pip install snakeviz memory_profiler psutil

This script will:
1. Create realistic test data matching production scenarios
2. Profile the rebalance API with cProfile
3. Generate memory usage reports
4. Create visualizable profiling output for snakeviz
"""

import cProfile
import json
import logging
import os
import pstats
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
from memory_profiler import profile

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import ConstraintsV2, Objective, Portfolio, Position, RebalanceRequest
from app.utils.rebalance_handler import perform_rebalance

# Configure logging to reduce noise during profiling
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_realistic_test_request(customer_id: int | None = None) -> RebalanceRequest:
    """Create a realistic rebalance request.

    Prefer loading a real portfolio from parquet. If the parquet file is missing
    or the schema does not conform to the `Position` model, fall back to the
    built-in mock portfolio used previously.
    """

    def _opt_str(value: object) -> str | None:
        """Convert pandas NA/None to None, otherwise cast to str.

        This avoids Pydantic validation errors when parquet contains <NA> in
        optional string fields.
        """

        try:
            import pandas as _pd  # local import to avoid circular issues

            if value is None or _pd.isna(value):
                return None
        except Exception:
            if value is None:
                return None
        return str(value)

    def build_mock_portfolio() -> Portfolio:
        """Fallback portfolio using the existing hard-coded positions."""

        positions = [
            Position(
                productId="PROD001",
                desk="Equity",
                portType="Discretionary",
                currency="THB",
                symbol="BBL",
                srcSharecodes="BBL",
                productDisplayName="Bangkok Bank Public Company Limited",
                productTypeDesc="Equity",
                assetClass="Local Equity",
                marketValue=5000000.0,
                expectedReturn=0.08,
                isRiskyAsset=True,
                isCoverage=False,
                isMonitored=True,
                unitBal=1000.0,
                unitPriceThb=5000.0,
                unitCostThb=4800.0,
                volatility=0.15,
                esCorePort=False,
                flagTopPick="",
            ),
            Position(
                productId="PROD002",
                desk="Equity",
                portType="Discretionary",
                currency="THB",
                symbol="SCB",
                srcSharecodes="SCB",
                productDisplayName="Siam Commercial Bank",
                productTypeDesc="Equity",
                assetClass="Local Equity",
                marketValue=3000000.0,
                expectedReturn=0.07,
                isRiskyAsset=True,
                isCoverage=False,
                isMonitored=True,
                unitBal=600.0,
                unitPriceThb=5000.0,
                unitCostThb=4900.0,
                volatility=0.14,
                esCorePort=False,
                flagTopPick="",
            ),
            Position(
                productId="PROD003",
                desk="Fixed Income",
                portType="Discretionary",
                currency="THB",
                symbol="LBHAGA",
                srcSharecodes="LBHAGA",
                productDisplayName="LH Financial Group Bond",
                productTypeDesc="Bond",
                assetClass="Fixed Income",
                marketValue=8000000.0,
                expectedReturn=0.04,
                isRiskyAsset=False,
                isCoverage=False,
                isMonitored=True,
                unitBal=8000.0,
                unitPriceThb=1000.0,
                unitCostThb=990.0,
                volatility=0.05,
                esCorePort=False,
                flagTopPick="",
            ),
            Position(
                productId="PROD004",
                desk="Mutual Fund",
                portType="Discretionary",
                currency="THB",
                symbol="TMBEQUAL",
                srcSharecodes="TMBEQUAL",
                productDisplayName="TMB Equal Fund",
                productTypeDesc="Mutual Fund",
                assetClass="Global Equity",
                marketValue=4000000.0,
                expectedReturn=0.10,
                isRiskyAsset=True,
                isCoverage=False,
                isMonitored=True,
                unitBal=4000.0,
                unitPriceThb=1000.0,
                unitCostThb=950.0,
                volatility=0.12,
                esCorePort=False,
                flagTopPick="",
            ),
            Position(
                productId="PROD005",
                desk="Cash",
                portType="Discretionary",
                currency="THB",
                symbol="CASH",
                srcSharecodes="CASH",
                productDisplayName="Cash Account",
                productTypeDesc="Cash",
                assetClass="Cash and Cash Equivalent",
                marketValue=1000000.0,
                expectedReturn=0.01,
                isRiskyAsset=False,
                isCoverage=False,
                isMonitored=True,
                unitBal=1000000.0,
                unitPriceThb=1.0,
                unitCostThb=1.0,
                volatility=0.01,
                esCorePort=False,
                flagTopPick="",
            ),
            # Add more positions to simulate realistic portfolio
            Position(
                productId="PROD006",
                desk="Equity",
                portType="Discretionary",
                currency="THB",
                symbol="AOT",
                srcSharecodes="AOT",
                productDisplayName="Airports of Thailand",
                productTypeDesc="Equity",
                assetClass="Local Equity",
                marketValue=2500000.0,
                expectedReturn=0.06,
                isRiskyAsset=True,
                isCoverage=False,
                isMonitored=True,
                unitBal=500.0,
                unitPriceThb=5000.0,
                unitCostThb=4800.0,
                volatility=0.13,
                esCorePort=False,
                flagTopPick="",
            ),
            Position(
                productId="PROD007",
                desk="Alternative",
                portType="Discretionary",
                currency="USD",
                symbol="GOLD",
                srcSharecodes="GOLD",
                productDisplayName="Gold Fund",
                productTypeDesc="Alternative",
                assetClass="Alternative",
                marketValue=2000000.0,
                expectedReturn=0.12,
                isRiskyAsset=True,
                isCoverage=False,
                isMonitored=True,
                unitBal=2000.0,
                unitPriceThb=1000.0,
                unitCostThb=950.0,
                volatility=0.18,
                esCorePort=False,
                flagTopPick="",
            ),
            Position(
                productId="PROD008",
                desk="Private Market",
                portType="Discretionary",
                currency="THB",
                symbol="PRIVATE1",
                srcSharecodes="PRIVATE1",
                productDisplayName="Private Equity Fund",
                productTypeDesc="Private Market",
                assetClass="Alternative",
                marketValue=5000000.0,
                expectedReturn=0.15,
                isRiskyAsset=True,
                isCoverage=False,
                isMonitored=False,
                unitBal=5000.0,
                unitPriceThb=1000.0,
                unitCostThb=900.0,
                volatility=0.20,
                esCorePort=False,
                flagTopPick="",
            ),
        ]

        return Portfolio(positions=positions)

    # Try to load a real portfolio from parquet
    portfolio: Portfolio
    parquet_path = (
        Path(__file__).parent.parent
        / "app"
        / "data"
        / "portfolios_client_out_enriched_2025-10-31.parquet"
    )

    if parquet_path.exists():
        try:
            df_es = pd.read_parquet(parquet_path)

            # Validate schema: ensure required backend columns exist to build Position
            # Based on actual parquet columns
            required_backend_cols = [
                "product_id",
                "desk",
                "port_type",
                "currency",
                "product_display_name",
                "product_type_desc",
                "asset_class_name",
                "symbol",
                "src_sharecodes",
                "pp_asset_sub_class",
                "is_risky_asset",
                "coverage_prdtype",
                "is_coverage",
                "expected_return",
                "es_core_port",
                "es_sell_list",
                "flag_top_pick",
                "flag_tax_saving",
                "value",
            ]

            missing = [c for c in required_backend_cols if c not in df_es.columns]
            if missing:
                logger.warning(
                    "[profile] parquet schema missing required backend columns for Position: %s",
                    missing,
                )

            # If customer_id is present, optionally target a specific client
            if "customer_id" in df_es.columns:
                target_cust = (
                    customer_id
                    if customer_id is not None
                    else df_es["customer_id"].iloc[0]
                )
                df_es = df_es[df_es["customer_id"] == target_cust]

            positions_from_df = []
            for _, row in df_es.iterrows():
                # Map backend columns to Position's aliased field names.
                # Synthesise unit/price/cost from value for profiling purposes.
                try:
                    market_value = float(row["value"])
                except Exception:
                    logger.warning(
                        "[profile] row has invalid value=%r; skipping", row.get("value")
                    )
                    continue

                unit_bal = 1.0
                unit_price = market_value
                unit_cost = market_value

                def _opt_str(val):
                    if pd.isna(val):
                        return None
                    return str(val)

                try:
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
                        "marketValue": market_value,
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
                except KeyError as e:
                    logger.warning(
                        "[profile] missing backend column %s when building Position; skipping row",
                        e,
                    )
                    continue

                try:
                    positions_from_df.append(Position(**payload))
                except Exception as e:
                    logger.warning(
                        "[profile] skipping row due to Position validation error: %s",
                        e,
                    )

            if not positions_from_df:
                logger.warning(
                    "[profile] no valid positions could be constructed from parquet; falling back to mock portfolio"
                )
                portfolio = build_mock_portfolio()
            else:
                portfolio = Portfolio(positions=positions_from_df)
        except Exception as e:
            logger.warning(
                "[profile] failed to load parquet portfolio from %s: %s; falling back to mock portfolio",
                parquet_path,
                e,
            )
            portfolio = build_mock_portfolio()
    else:
        logger.warning(
            "[profile] parquet file not found at %s; falling back to mock portfolio",
            parquet_path,
        )
        portfolio = build_mock_portfolio()

    # Target allocation for rebalancing
    target_alloc = {
        "Cash and Cash Equivalent": 0.05,
        "Fixed Income": 0.40,
        "Local Equity": 0.25,
        "Global Equity": 0.20,
        "Alternative": 0.10,
    }

    effective_customer_id = customer_id if customer_id is not None else 14055

    return RebalanceRequest(
        customer_id=effective_customer_id,
        portfolio=portfolio,
        objective=Objective(
            objective="risk_adjusted",
            client_style="Moderate High Risk",
            target_alloc=target_alloc,
            new_money=1000000.0,  # 1M THB new money
        ),
        constraints=ConstraintsV2(
            discretionary_acceptance=0.3,
            private_percent=0.1,
            cash_percent=0.05,
            offshore_percent=0.15,
            product_whitelist=[],
            product_blacklist=["RISKY_PRODUCT"],
        ),
        style=[{"client_style": "Moderate High Risk"}],
    )


class MockAppState:
    """Mock FastAPI app.state for testing."""

    def __init__(self):
        from app.data_store import (
            get_sales_customer_mapping,
            hs,
            load_sales_customer_mapping,
            ports,
            ppm,
            prepare_portfolio_data,
            rb,
        )

        # Initialize data like the real app
        try:
            load_sales_customer_mapping()
            prepare_portfolio_data()
        except Exception as e:
            print(f"Warning: Data initialization partially failed: {e}")

        # Set up state like the real app
        self.ports = ports
        self.ppm = ppm
        self.hs = hs
        self.rb = rb
        self.sales_customer_mapping = get_sales_customer_mapping()


def profile_rebalance_function(customer_id: int | None = None):
    """Profile the rebalance function with detailed timing."""

    print("🚀 Starting rebalance API profiling...")
    print("=" * 60)

    # Setup
    state = MockAppState()
    request = create_realistic_test_request(customer_id=customer_id)

    # Memory tracking
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    print(f"📊 Memory before: {memory_before:.1f} MB")
    print(f"📈 Portfolio positions: {len(request.portfolio.positions)}")
    print(
        f"💰 Portfolio value: {sum(p.market_value() for p in request.portfolio.positions):,.0f} THB"
    )
    print()

    # CPU profiling with cProfile
    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.perf_counter()

    try:
        # Execute rebalance
        result = perform_rebalance(state, request)

        end_time = time.perf_counter()
        profiler.disable()

        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB

        print(f"✅ Rebalance completed successfully!")
        print(f"⏱️  Total execution time: {(end_time - start_time):.3f} seconds")
        print(f"📊 Memory after: {memory_after:.1f} MB")
        print(f"📈 Memory delta: {memory_after - memory_before:+.1f} MB")
        print(f"🔄 Rebalance actions: {len(result.actions)}")
        print(f"🏥 Health score: {result.health_score:.2f}")
        print()

        return profiler, result, (end_time - start_time)

    except Exception as e:
        profiler.disable()
        print(f"❌ Rebalance failed: {e}")
        raise


def save_profiling_results(
    profiler: cProfile.Profile, execution_time: float, output_dir: str
):
    """Save profiling results in multiple formats."""

    os.makedirs(output_dir, exist_ok=True)

    # Save raw profile data for snakeviz
    profile_file = os.path.join(output_dir, "rebalance_profile.prof")
    profiler.dump_stats(profile_file)
    print(f"💾 Profile saved: {profile_file}")
    print(f"   View with: snakeviz {profile_file}")
    print()

    # Save text report
    text_file = os.path.join(output_dir, "rebalance_profile.txt")
    with open(text_file, "w") as f:
        # Create stats object and sort by cumulative time
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumulative")
        print("=" * 80, file=f)
        print("REBALANCE API PROFILING REPORT", file=f)
        print("=" * 80, file=f)
        print(f"Total execution time: {execution_time:.3f} seconds", file=f)
        print(file=f)

        print("Top 20 functions by cumulative time:", file=f)
        print("-" * 50, file=f)
        stats.print_stats(20)

        print(file=f)
        print("Top 20 functions by total time:", file=f)
        print("-" * 50, file=f)
        stats.sort_stats("tottime")
        stats.print_stats(20)

        print(file=f)
        print("Function callers (who called the expensive functions):", file=f)
        print("-" * 50, file=f)
        stats.sort_stats("cumulative")
        stats.print_callers(10)

    print(f"📄 Text report saved: {text_file}")

    # Save summary JSON
    summary_file = os.path.join(output_dir, "rebalance_summary.json")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")

    # Extract top functions
    top_functions = []
    for func_info in stats.stats.items():
        func, (cc, nc, tt, ct, callers) = func_info
        if len(top_functions) < 10:  # Top 10 only
            top_functions.append(
                {
                    "function": f"{func[2]}:{func[1]}({func[0]})",
                    "cumulative_time": ct,
                    "total_time": tt,
                    "call_count": nc,
                    "per_call_time": ct / nc if nc > 0 else 0,
                }
            )

    summary = {
        "execution_time_seconds": execution_time,
        "profile_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_functions": top_functions,
        "total_functions_profiled": len(stats.stats),
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📋 Summary JSON saved: {summary_file}")
    print()


def profile_memory_usage():
    """Profile memory usage patterns."""

    print("🧠 Running memory profiling...")

    @profile
    def memory_intensive_rebalance():
        state = MockAppState()
        request = create_realistic_test_request()
        return perform_rebalance(state, request)

    # This will output memory line-by-line analysis
    result = memory_intensive_rebalance()
    return result


def main():
    """Main profiling execution."""

    import argparse

    parser = argparse.ArgumentParser(description="Profile rebalance API performance")
    parser.add_argument(
        "--output-dir",
        default="./profiling_results",
        help="Directory to save profiling results",
    )
    parser.add_argument(
        "--memory", action="store_true", help="Run detailed memory profiling"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations to run"
    )
    parser.add_argument(
        "--customer-id",
        type=int,
        default=None,
        help="Optional customer_id to target when loading portfolio from parquet",
    )

    args = parser.parse_args()

    print("🔍 REBALANCE API PERFORMANCE PROFILER")
    print("=" * 60)

    execution_times = []

    # Run multiple iterations for consistent results
    for i in range(args.iterations):
        print(f"\n--- Iteration {i+1}/{args.iterations} ---")

        profiler, result, exec_time = profile_rebalance_function(
            customer_id=args.customer_id
        )
        execution_times.append(exec_time)

        # Save profiling results for first iteration
        if i == 0:
            save_profiling_results(profiler, exec_time, args.output_dir)

    # Summary statistics
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)

    print("\n📈 PERFORMANCE SUMMARY")
    print("=" * 40)
    print(f"Iterations run: {args.iterations}")
    print(f"Average time: {avg_time:.3f}s")
    print(f"Min time: {min_time:.3f}s")
    print(f"Max time: {max_time:.3f}s")
    print(
        f"Std deviation: {(sum((t - avg_time)**2 for t in execution_times) / len(execution_times))**0.5:.3f}s"
    )

    # Memory profiling if requested
    if args.memory:
        print("\n" + "=" * 60)
        profile_memory_usage()

    print(f"\n✨ Profiling complete! Check {args.output_dir}/ for detailed results.")
    print(
        "🔍 To visualize the profile: snakeviz profiling_results/rebalance_profile.prof"
    )


if __name__ == "__main__":
    main()
