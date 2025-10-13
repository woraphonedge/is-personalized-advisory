from __future__ import annotations

import logging
from typing import List

import pandas as pd

from app.models import Position
from app.utils.portfolios_service import PortfolioService

logger = logging.getLogger(__name__)


def get_portfolio_for_customer(ports, customer_id: int):
    """
    Build and return a Portfolio model for the given customer_id using the
    preloaded `ports` store. This function encapsulates slicing, cleaning, and
    mapping from DataFrame rows to the public Position model with camelCase
    aliases.

    Parameters
    - ports: app.state.ports (Portfolios)
    - customer_id: numeric customer id

    Returns
    - Portfolio model instance (from app.models.Portfolio)
    """
    # Lazy import to avoid circular import at module level
    from app.models import Portfolio  # type: ignore

    # Use preloaded portfolios and slice by customer via service
    service = PortfolioService(ports)
    client_port = service.get_client_portfolio(customer_id)

    df = client_port.df_out
    if df is None or df.empty:
        logger.info(
            "No portfolio rows returned | customer_id=%s (preloaded)",
            customer_id,
        )
        raise ValueError("No portfolio found for customer")

    # Normalize null-like values for flags
    try:
        # replace <NA> with None where appropriate
        if "es_sell_list" in df.columns:
            df["es_sell_list"] = df["es_sell_list"].replace({pd.NA: None})
        if "flag_tax_saving" in df.columns:
            df["flag_tax_saving"] = df["flag_tax_saving"].replace({pd.NA: None})
    except Exception:
        # best-effort
        pass

    positions: List[Position] = []
    logger.debug(
        "Sliced preloaded portfolio | shape=%s columns=%s sample=%s",
        tuple(df.shape),
        list(df.columns),
        df.head(3).to_dict(orient="records") if len(df) > 0 else [],
    )

    for _, r in df.iterrows():
        # Prefer explicit value columns produced by ETL
        mv = r.get("value", 0.0)
        # Format pos date if present
        pos_date = None
        try:
            as_of = r.get("as_of_date")
            if as_of is not None:
                if hasattr(as_of, "strftime"):
                    pos_date = as_of.strftime("%Y-%m-%d")
                else:
                    pos_date = str(as_of)
        except Exception:
            pos_date = None

        positions.append(
            Position(
                productId=r.get("product_id"),
                desk=r.get("desk"),
                portType=r.get("port_type"),
                currency=r.get("currency"),
                symbol=r.get("symbol") or r.get("src_sharecodes"),
                srcSharecodes=r.get("src_sharecodes"),
                productTypeDesc=r.get("product_type_desc"),
                assetClass=r.get("asset_class_name"),
                assetSubClass=r.get("pp_asset_sub_class"),
                unitBal=0.0,
                unitPriceThb=1.0,
                unitCostThb=1.0,
                marketValue=float(mv or 0.0),
                expectedReturn=float(r.get("expected_return") or 0.0),
                expectedIncomeYield=None,
                volatility=0.0,
                isMonitored=True,
                coveragePrdtype=r.get("coverage_prdtype"),
                isCoverage=r.get("is_coverage"),
                esCorePort=r.get("es_core_port"),
                esSellList=r.get("es_sell_list"),
                flagTopPick=r.get("flag_top_pick"),
                flagTaxSaving=r.get("flag_tax_saving"),
                exposures=None,
                posDate=pos_date,
            )
        )

    portfolio_model = Portfolio(positions=positions)
    logger.debug("Built Portfolio model with %d positions", len(positions))
    return portfolio_model
