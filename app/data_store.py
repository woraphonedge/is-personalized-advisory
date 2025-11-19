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
from typing import Optional

import pandas as pd

from app.utils.data_loader import DataLoader
from app.utils.health_score import HealthScore
from app.utils.portfolios import Portfolios
from app.utils.portfolios_repo import PortfoliosRepository
from app.utils.portprop_matrices import PortpropMatrices
from app.utils.portprop_matrices_repo import PortpropMatricesRepository
from app.utils.rebalancer import Rebalancer
from app.utils.rebalancer_repo import RebalancerRepository

# Module-level caches populated during app lifespan
df_out_loaded: Optional[pd.DataFrame] = None
df_style_loaded: Optional[pd.DataFrame] = None
port_ids_loaded: Optional[pd.DataFrame] = None
port_id_mapping_loaded: Optional[pd.DataFrame] = None
acct_customer_mapping_loaded: Optional[pd.DataFrame] = None
sales_customer_mapping_loaded: Optional[dict[str, str]] = None

discretionary_acceptance = 0.4
as_of_date = os.getenv("AS_OF_DATE", "2025-09-30")
prod_comp_keys = ['product_id', 'src_sharecodes', 'desk', 'port_type', 'currency']

# Repositories
loader = DataLoader()
ports_repo = PortfoliosRepository(loader)
ppm_repo = PortpropMatricesRepository(loader)

# Reference dictionaries
ports_ref_table = {
    "product_mapping": ports_repo.load_product_mapping(as_of_date=as_of_date),
    "product_underlying": ports_repo.load_product_underlying(),
}
ports_ref_table['product_mapping'].reset_index(inplace=True)
ports_ref_table['product_mapping'].rename(columns={'index': 'sec_id'}, inplace=True)

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

    # map sec_id to mandate
    _rb_refs['mandate_allocation'] = _rb_refs['mandate_allocation'].merge(
    ports_ref_table['product_mapping'][prod_comp_keys + ['sec_id']],
    on=prod_comp_keys,
    how='left',
    validate='one_to_one'
    )

    # map sec_id to prod recom rank
    mf_prod_mapping_rank = ports_ref_table["product_mapping"][
        ports_ref_table["product_mapping"]["product_type_desc"] == "Mutual Fund"
    ]
    product_recommendation_rank_raw = _rb_refs["product_recommendation_rank_raw"][
        ["src_sharecodes", "desk", "currency"] + ["is_ui", "rank_product"]
    ]
    _rb_refs["product_recommendation_rank_raw"] = product_recommendation_rank_raw.merge(
        mf_prod_mapping_rank,
        on=["src_sharecodes", "desk", "currency"],
        how="left",
        validate="one_to_one",
    )
    rb.set_ref_tables(_rb_refs)
except Exception as e:
    # Keep startup resilient; phases will skip with [TEMP-DEBUG] logs if missing
    print(f"[startup] Rebalancer refs not fully loaded: {e}")


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
    df_out_raw = df_out_raw.merge(
        ports_ref_table['product_mapping'][prod_comp_keys + ['sec_id']],
        on=prod_comp_keys,
        how='left',
        validate='many_to_one'
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
        # Expose sales-customer mapping for access control
        app.state.sales_customer_mapping = get_sales_customer_mapping()
        # Optionally prepare portfolio data if backend sources are reachable
        prepare_portfolio_data()
        for name, df in [
            ("ports_ref_table['product_mapping']", ports.product_mapping),
            ("client_out_enriched", ports.df_out),
            ("rb_ref_dict['product_recommendation_rank_raw']", rb.prod_reco_rank_raw),
            ("rb_ref_dict['mandate_allocation']", rb.discretionary_allo_weight),
            ]:
            print(f"{name} has 'sec_id'? {'sec_id' in df.columns}")
    except Exception as e:
        # Defer failures to request time; not fatal for server startup
        print(f"[startup] Skipped prepare_portfolio_data due to error: {e}")
    try:
        yield
    finally:
        # No special cleanup needed for now
        pass
