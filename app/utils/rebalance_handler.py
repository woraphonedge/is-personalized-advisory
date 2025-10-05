from __future__ import annotations

import copy
import logging
from typing import Any, List

import pandas as pd

from app.models import (
    ActionLog,
    Portfolio,
    Position,
    RebalanceRequest,
    RebalanceResponse,
)
from app.utils.portfolios_service import PortfolioService
from app.utils.rebalancer_mock import compute_portfolio_health
from app.utils.utils import convert_portfolio_to_df

logger = logging.getLogger(__name__)

def _build_df_style(customer_id: int, style: Any) -> pd.DataFrame:
    """Normalize style payload into a single-row DataFrame.

    Accepts style as a string, dict, or list of dicts/strings and produces columns:
    - customer_id (int)
    - port_investment_style (str)
    - portpop_style (str)
    """
    style_map = {
        'Bulletproof': 'Conservative',
        'Conservative': 'Conservative',
        'Moderate Low Risk': 'Medium to Moderate Low Risk',
        'Moderate High Risk': 'Medium to Moderate High Risk',
        'High Risk': 'High Risk',
        'Aggressive Growth': 'Aggressive',
        'Unwavering': 'Aggressive',
    }

    # Derive a string label from supported inputs
    label = None
    try:
        if isinstance(style, str):
            label = style
        elif isinstance(style, dict):
            label = (
                style.get('INVESTMENT_STYLE_AUMX')
                or style.get('INVESTMENT_STYLE')
                or style.get('style')
                or style.get('client_style')
            )
        elif isinstance(style, list) and len(style) > 0:
            first = style[0]
            if isinstance(first, str):
                label = first
            elif isinstance(first, dict):
                label = (
                    first.get('INVESTMENT_STYLE_AUMX')
                    or first.get('INVESTMENT_STYLE')
                    or first.get('style')
                    or first.get('client_style')
                )
    except Exception:
        # Best-effort parsing only
        pass

    if not label:
        label = 'High Risk'

    data = {
        "customer_id": customer_id,
        "port_investment_style": label,
        "portpop_style": style_map.get(label, "High Risk"),
    }
    # Build a single-row DataFrame to avoid pandas scalar dict error
    df = pd.DataFrame([data])
    return df


def perform_rebalance(state: Any, request: RebalanceRequest) -> RebalanceResponse:
    """Core rebalancing logic extracted from FastAPI route.

    Parameters
    - state: FastAPI app.state providing ports, ppm, hs
    - request: validated RebalanceRequest
    """
    # Convert incoming portfolio to df and tag with customer id
    try:
        df_out = convert_portfolio_to_df(request.portfolio)
        logger.debug("converted portfolio df columns=%s rows=%d", list(df_out.columns), len(df_out))
    except Exception as e:
        logger.exception("Failed to convert portfolio to DataFrame: %s", e)
        raise
    df_out["customer_id"] = request.customer_id
    df_out["port_id_mapping"] = request.customer_id

    # Build style DataFrame
    df_style_loaded = _build_df_style(request.customer_id, request.style)

    # Create/assign portfolio ids
    # deepcopy ports to avoid modifying the original state
    ports = copy.deepcopy(state.ports)
    df_out, df_style, port_ids, mapping = ports.create_portfolio_id(
        df_out, df_style_loaded, column_mapping=["customer_id"]
    )
    logger.debug("created portfolio ids columns=%s rows=%d", list(df_out.columns), len(df_out))

    ports.set_portfolio(df_out, df_style, port_ids, mapping)
    port_service = PortfolioService(ports)
    port = port_service.get_client_portfolio(request.customer_id)
    # Instantiate and run rebalancer v2
    from app.utils.rebalancer import Rebalancer  # local import to avoid circular issues

    c = request.constraints
    # Try to infer optional fields from style
    as_of_date = None
    customer_id_val = None
    try:
        if isinstance(request.style, list) and len(request.style) > 0:
            tmp = pd.DataFrame(request.style)
            as_of_date = str(tmp.get("AS_OF_DATE").iloc[0]) if "AS_OF_DATE" in tmp else None
            customer_id_val = pd.to_numeric(tmp.get("CUSTOMER_ID").iloc[0], errors="coerce") if "CUSTOMER_ID" in tmp else None
        elif isinstance(request.style, dict):
            as_of_date = str(request.style.get("AS_OF_DATE")) if request.style.get("AS_OF_DATE") is not None else None
            customer_id_val = pd.to_numeric(request.style.get("CUSTOMER_ID"), errors="coerce") if request.style.get("CUSTOMER_ID") is not None else None
    except Exception:
        pass

    logger.debug(f"""
                Rebalancing for customer_id={request.customer_id} as_of_date={as_of_date} customer_id_val={customer_id_val}
                new_money={request.objective.new_money}
                discretionary_acceptance={c.discretionary_acceptance}
                private_percent={c.private_percent}
                cash_percent={c.cash_percent}
                offshore_percent={c.offshore_percent}
                product_restriction={c.product_restriction}
                """)
    rb_local = Rebalancer(
        as_of_date=as_of_date or None,
        customer_id=(int(customer_id_val) if customer_id_val is not None and not pd.isna(customer_id_val) else None),
        new_money=request.objective.new_money,
        discretionary_acceptance=c.discretionary_acceptance,
        private_percent=c.private_percent,
        cash_percent=c.cash_percent,
        offshore_percent=c.offshore_percent,
        product_restriction=c.product_restriction,
    )
    try:
        logger.debug("ports.df_out head=\n%s", ports.df_out.head())
    except Exception:
        logger.debug("ports.df_out not available for preview")
    logger.debug("Rebalancing...")
    try:
        actions_df = rb_local.rebalance(port, state.ppm, state.hs)
    except Exception as e:
        logger.exception("Rebalance failed: %s", e)
        raise
    logger.debug("Rebalance completed")
    action_logs: List[ActionLog] = []
    if actions_df is not None and not actions_df.empty:
        for _, r in actions_df.iterrows():
            action_logs.append(
                ActionLog(
                    action=r.get("flag"),
                    step="",
                    trade_type=r.get("action"),
                    symbol=r.get("src_sharecodes"),
                    amount_thb=r.get("amount"),
                    unit=None,
                    price=None,
                    asset_class="Unknown",
                    notes=None,
                )
            )

    # Build proposed portfolio model
    positions: List[Position] = []
    proposed_portfolio_df = port.df_out
    if proposed_portfolio_df is not None and not proposed_portfolio_df.empty:
        for _, r in proposed_portfolio_df.iterrows():
            positions.append(
                Position(
                    productId=str(r.get("product_id", "UNKNOWN")),
                    desk=str(r.get("desk", "UNKNOWN")),
                    portType=str(r.get("port_type", "UNKNOWN")),
                    currency=str(r.get("currency", "UNKNOWN")),
                    symbol=str(r.get("src_sharecodes", "UNKNOWN")),
                    assetClass=r.get("asset_class_name"),
                    assetSubClass=r.get("asset_sub_class"),
                    unitBal=0.0,
                    unitPriceThb=1.0,
                    unitCostThb=1.0,
                    marketValue=float(r.get("value", 0.0) or 0.0),
                    expectedReturn=float(r.get("expected_return", 0.0) or 0.0),
                    expectedIncomeYield=0.0,
                    volatility=0.0,
                    isMonitored=True,
                    exposures=None,
                )
            )
    proposed_portfolio_model = Portfolio(positions=positions)

    # Compute health metrics on proposed portfolio
    _metrics = compute_portfolio_health(
        proposed_portfolio_model,
        request.objective.target_alloc,
    )
    health = _metrics.score

    return RebalanceResponse(
        actions=action_logs,
        portfolio=proposed_portfolio_model,
        health_score=health,
    )
