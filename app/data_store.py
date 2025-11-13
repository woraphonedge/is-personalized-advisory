"""
Centralized in-memory data store for the backend. Uses a FastAPI lifespan
initializer to populate data at startup.

This module wires together the overhauled utils using the repository pattern:
- `DataLoader` controls whether to pull from DB or local parquet cache via env.
- `PortfoliosRepository` loads client outputs/styles and product reference tables.
- `PortpropMatricesRepository` loads PortProp and advisory model reference tables.
- `Portfolios`, `PortpropMatrices`, `HealthScore`, `Rebalancer` provide core logic.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd

from app.models import Position
from app.utils.data_loader import DataLoader
from app.utils.health_score import HealthScore
from app.utils.portfolios import Portfolios
from app.utils.portfolios_repo import PortfoliosRepository
from app.utils.portprop_matrices import PortpropMatrices
from app.utils.portprop_matrices_repo import PortpropMatricesRepository
from app.utils.rebalancer import Rebalancer
from app.utils.rebalancer_repo import RebalancerRepository

# Module-level caches populated during app lifespan
_candidate_data: Optional[List[Position]] = None
df_out_loaded = None
df_style_loaded = None
port_ids_loaded = None
port_id_mapping_loaded = None
sales_customer_mapping_loaded = None
acct_customer_mapping_loaded = None

discretionary_acceptance = 0.4
as_of_date = os.getenv("AS_OF_DATE", "2025-09-30")


loader = DataLoader()

# Repositories
ports_repo = PortfoliosRepository(loader)
ppm_repo = PortpropMatricesRepository(loader)

# Reference dictionaries
ports_ref_table = {
    "product_mapping": ports_repo.load_product_mapping(as_of_date=as_of_date),
    "product_underlying": ports_repo.load_product_underlying(),
}

ppm_ref_dict = {
    "portprop_factsheet": ppm_repo.load_portprop_factsheet(),
    "portprop_fallback": ppm_repo.load_portprop_fallback(),
    "portprop_benchmark": ppm_repo.load_portprop_benchmark(),
    "portprop_ge_mapping": ppm_repo.load_portprop_ge_mapping(),
    "portprop_ret_eow": ppm_repo.load_portprop_ret_eow(),
    "advisory_health_score": ppm_repo.load_advisory_health_score(),
}

# Core singletons
ports = Portfolios()
ports.set_ref_tables(ports_ref_table)
ppm = PortpropMatrices(ppm_ref_dict)
hs = HealthScore()
rb = Rebalancer(discretionary_acceptance=discretionary_acceptance)
rb_repo = RebalancerRepository(loader)

# Preload rebalancer reference tables
try:
    _rb_refs = {
        "es_sell_list": rb_repo.load_es_sell_list(),
        "product_recommendation_rank_raw": rb_repo.load_product_recommendation_rank_raw(),
        "mandate_allocation": rb_repo.load_mandate_candidates(),
    }
    rb.set_ref_tables(_rb_refs)
except Exception as e:
    # Keep startup resilient; phases will skip with [TEMP-DEBUG] logs if missing
    print(f"[startup] Rebalancer refs not fully loaded: {e}")


def load_data() -> None:
    """Load in-memory dataset and candidate universe."""
    global _candidate_data
    # Define candidate universe for purchase
    _candidate_data = [
        Position(
            product_id="CASH",
            src_symbol="CASH",
            asset_class="Cash and Cash Equivalent",
            asset_sub_class="Cash THB",
            unit_bal=0,
            unit_price_thb=1,
            unit_cost_thb=1,
            expected_return=0.0,
            expected_income_yield=0.0,
            volatility=0.01,
            is_monitored=True,
        ),
        Position(
            product_id="GLB_EQTY_FUND",
            src_symbol="GLBL2",
            asset_class="Global Equity",
            asset_sub_class="Global Equity",
            unit_bal=0,
            unit_price_thb=100,
            unit_cost_thb=100,
            expected_return=0.11,
            expected_income_yield=0.02,
            volatility=0.20,
            is_monitored=True,
        ),
        Position(
            product_id="GLB_BOND_FUND",
            src_symbol="BOND3",
            asset_class="Fixed Income",
            asset_sub_class="Global Bond",
            unit_bal=0,
            unit_price_thb=50,
            unit_cost_thb=50,
            expected_return=0.04,
            expected_income_yield=0.04,
            volatility=0.03,
            is_monitored=True,
        ),
        Position(
            product_id="TH_EQTY_FUND",
            src_symbol="LOCAL2",
            asset_class="Local Equity",
            asset_sub_class="Thai Equity",
            unit_bal=0,
            unit_price_thb=60,
            unit_cost_thb=60,
            expected_return=0.07,
            expected_income_yield=0.015,
            volatility=0.22,
            is_monitored=True,
        ),
        Position(
            product_id="ALT1",
            src_symbol="ALT1",
            asset_class="Alternative",
            asset_sub_class="Prop/Infra Fund",
            unit_bal=0,
            unit_price_thb=80,
            unit_cost_thb=80,
            expected_return=0.10,
            expected_income_yield=0.0,
            volatility=0.35,
            is_monitored=True,
        ),
        Position(
            product_id="AA2",
            src_symbol="AA2",
            asset_class="Allocation",
            asset_sub_class="Moderate Allocation",
            unit_bal=0,
            unit_price_thb=105,
            unit_cost_thb=105,
            expected_return=0.075,
            expected_income_yield=0.03,
            volatility=0.12,
            is_monitored=True,
            exposures={
                "Cash and Cash Equivalent": 0.05,
                "Fixed Income": 0.30,
                "Global Equity": 0.45,
                "Local Equity": 0.10,
                "Alternative": 0.10,
            },
        ),
    ]


def get_candidate_data() -> List[Position]:
    if _candidate_data is None:
        raise RuntimeError("Data store not initialized: candidate_data is None")
    return _candidate_data


def prepare_portfolio_data() -> None:
    """Load client portfolio outputs/styles and set `ports`.

    Uses new repository loaders consistent with the sample notebook usage.
    """
    global df_out_loaded, df_style_loaded, port_ids_loaded, port_id_mapping_loaded, acct_customer_mapping_loaded

    # Load all client positions and styles as of configured date
    df_out_raw = ports_repo.load_client_out_product_enriched(
        as_of_date=as_of_date,
        value_column="AUMX_THB",
    )
    df_style_raw = ports_repo.load_client_style(
        as_of_date=as_of_date,
        style_column="INVESTMENT_STYLE_AUMX",
    )

    # Create portfolio ids by ['as_of_date','customer_id'] and set portfolio
    df_out_loaded, df_style_loaded, port_ids_loaded, port_id_mapping_loaded = (
        ports.create_portfolio_id(
            df_out_raw.copy(),
            df_style_raw.copy(),
            column_mapping=["as_of_date", "customer_id"],
        )
    )
    ports.set_portfolio(
        df_out_loaded, df_style_loaded, port_ids_loaded, port_id_mapping_loaded
    )

    # Load account-customer mapping for account-number based search
    try:
        acct_customer_mapping_loaded = ports_repo.load_acct_customer_mapping()
        # Attach to global portfolios instance for easy access via app.state.ports
        ports.acct_cust_mapping = acct_customer_mapping_loaded
    except Exception as e:
        # Non-fatal at startup; account-number search will simply not return results
        print(f"[startup] Error loading acct_customer_mapping: {e}")


def load_sales_customer_mapping() -> None:
    """Load sales-customer mapping from CSV file for access control.

    Reads sales_customer_mapping.csv and stores it in memory for fast access.
    The CSV should contain SALES_ID and CUSTOMER_ID columns.
    """
    global sales_customer_mapping_loaded

    csv_path = os.path.join(
        os.path.dirname(__file__), "data", "sales_customer_mapping.csv"
    )

    try:
        if os.path.exists(csv_path):
            sales_customer_mapping_loaded = pd.read_csv(csv_path)
            print(
                f"[startup] Loaded sales-customer mapping: {len(sales_customer_mapping_loaded)} mappings"
            )
        else:
            print(
                f"[startup] Warning: sales_customer_mapping.csv not found at {csv_path}"
            )
            sales_customer_mapping_loaded = pd.DataFrame(
                columns=["SALES_ID", "CUSTOMER_ID"]
            )
    except Exception as e:
        print(f"[startup] Error loading sales_customer_mapping.csv: {e}")
        sales_customer_mapping_loaded = pd.DataFrame(
            columns=["SALES_ID", "CUSTOMER_ID"]
        )


def get_sales_customer_mapping() -> pd.DataFrame:
    """Get the sales-customer mapping DataFrame.

    Returns:
        DataFrame containing SALES_ID and CUSTOMER_ID mappings.
        Returns empty DataFrame if not loaded.
    """
    if sales_customer_mapping_loaded is None:
        raise RuntimeError(
            "Sales customer mapping not initialized - call load_sales_customer_mapping() first"
        )
    return sales_customer_mapping_loaded


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context that initializes in-memory data."""
    load_data()
    # Load sales-customer mapping for access control
    load_sales_customer_mapping()
    # Best-effort to prepare portfolio data; keep app starting even if DB not reachable
    try:
        # Expose shared singletons and preloaded data via app.state
        app.state.loader = loader
        app.state.ports_repo = ports_repo
        app.state.ppm_repo = ppm_repo
        app.state.ports = ports
        app.state.ppm = ppm
        app.state.hs = hs
        app.state.rb = rb
        # Expose commonly used types/constants to avoid per-request imports
        try:
            from app.models import Portfolio as _Portfolio
            from app.models import Position as _Position

            app.state.Portfolio = _Portfolio
            app.state.Position = _Position
        except Exception:
            # If models import fails at startup, routes can still import lazily
            pass
        app.state.AS_OF_DATE = as_of_date
        # Candidate purchase universe
        app.state.candidate_data = get_candidate_data()
        # Expose sales-customer mapping for access control
        app.state.sales_customer_mapping = get_sales_customer_mapping()
        # Optionally prepare portfolio data if backend sources are reachable
        prepare_portfolio_data()
    except Exception as e:
        # Defer failures to request time; not fatal for server startup
        print(f"[startup] Skipped prepare_portfolio_data due to error: {e}")
    try:
        yield
    finally:
        # No special cleanup needed for now
        pass
