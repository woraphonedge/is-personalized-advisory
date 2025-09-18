"""
Centralized in-memory data store for the backend. Uses a FastAPI lifespan
initializer to populate data at startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from app.models import Position
from app.utils.health_score import HealthScore
from app.utils.portfolios import Portfolios
from app.utils.portprop_matrices import PortpropMatrices
from app.utils.rebalancer import Rebalancer

# Module-level caches populated during app lifespan
_candidate_data: Optional[List[Position]] = None
df_out_loaded = None
df_style_loaded = None
port_ids_loaded = None
port_id_mapping_loaded = None

discretionary_acceptance = 0.4
as_of_date = "2025-07-31"
customer_id = 15689


ports = Portfolios()
ppm = PortpropMatrices()
hs = HealthScore()
rb = Rebalancer(discretionary_acceptance=discretionary_acceptance)
ports.load_product_mapping(as_of_date)


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
    """Load portfolio outputs and styles into the shared `ports` object.

    Populates module-level DataFrames `df_out_loaded`, `df_style_loaded`, and id mappings
    to be reused by API handlers requiring the Portfolios structure.
    """
    global df_out_loaded, df_style_loaded, port_ids_loaded, port_id_mapping_loaded

    # Pull client positions and style as of configured date
    where_clause = f"WHERE CUSTOMER_ID = {customer_id}"
    styles_and = f"AND CUSTOMER_ID = {customer_id}"
    df_out_raw = ports.get_client_out_from_query(
        as_of_date, as_of_date, where_query=where_clause
    )
    df_style_raw = ports.get_client_style_from_query(
        as_of_date, as_of_date, and_query=styles_and
    )

    # Map product info and create portfolio ids
    df_out_loaded = ports.map_client_out_prod_info(df_out_raw)
    df_style_loaded = df_style_raw
    df_out_loaded, df_style_loaded, port_ids_loaded, port_id_mapping_loaded = (
        ports.create_portfolio_id(
            df_out_loaded,
            df_style_loaded,
            column_mapping=["AS_OF_DATE", "CUSTOMER_ID"],
        )
    )
    ports.set_portfolio(
        df_out_loaded, df_style_loaded, port_ids_loaded, port_id_mapping_loaded
    )


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context that initializes in-memory data."""
    load_data()
    # Best-effort to prepare portfolio data; keep app starting even if DB not reachable
    try:
        # Expose shared singletons and preloaded data via app.state
        app.state.ports = ports
        app.state.ppm = ppm
        app.state.hs = hs
        app.state.rb = rb
        # Candidate purchase universe
        app.state.candidate_data = get_candidate_data()
        # Optionally prepare portfolio data if backend sources are reachable
        # prepare_portfolio_data()
    except Exception as e:
        # Defer failures to request time; not fatal for server startup
        print(f"[startup] Skipped prepare_portfolio_data due to error: {e}")
    try:
        yield
    finally:
        # No special cleanup needed for now
        pass
