from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional

from app.models import ActionLog, Portfolio, Position, RebalanceRequest, RebalanceResponse
from app.utils.rebalancer_mock import compute_portfolio_health
from app.utils.utils import convert_portfolio_to_df


def _build_df_style(request: RebalanceRequest) -> pd.DataFrame:
    """Normalize style payload into DataFrame with required columns.

    Ensures columns:
    - CUSTOMER_ID (int)
    - port_investment_style (str)
    """
    style = request.style
    if style is not None:
        if isinstance(style, list):
            df = pd.DataFrame(style)
        elif isinstance(style, dict):
            df = pd.DataFrame([style])
        else:
            df = pd.DataFrame()
        if "CUSTOMER_ID" not in df.columns:
            df["CUSTOMER_ID"] = request.customer_id
        if "port_investment_style" not in df.columns:
            df["port_investment_style"] = request.objective.client_style or "High Risk"
        return df
    # fallback to objective.client_style
    return pd.DataFrame([
        {
            "CUSTOMER_ID": request.customer_id,
            "port_investment_style": request.objective.client_style or "High Risk",
        }
    ])


def perform_rebalance(state: Any, request: RebalanceRequest) -> RebalanceResponse:
    """Core rebalancing logic extracted from FastAPI route.

    Parameters
    - state: FastAPI app.state providing ports, ppm, hs
    - request: validated RebalanceRequest
    """
    # Convert incoming portfolio to df and tag with customer id
    df_out = convert_portfolio_to_df(request.portfolio)
    df_out["CUSTOMER_ID"] = request.customer_id

    # Build style DataFrame
    df_style_loaded = _build_df_style(request)

    # Create/assign portfolio ids
    df_out, df_style, port_ids, mapping = state.ports.create_portfolio_id(
        df_out, df_style_loaded, column_mapping=["CUSTOMER_ID"]
    )
    state.ports.set_portfolio(df_out, df_style, port_ids, mapping)

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

    rb_local = Rebalancer(
        as_of_date=as_of_date or None,
        customer_id=(int(customer_id_val) if customer_id_val is not None and not pd.isna(customer_id_val) else None),
        new_money=request.objective.new_money,
        discretionary_percent=c.discretionary_percent,
        private_percent=c.private_percent,
        cash_percent=c.cash_percent,
        offshore_percent=c.offshore_percent,
        product_restriction=c.product_restriction,
        discretionary_acceptance=c.discretionary_acceptance,
    )

    actions_df = rb_local.rebalance(state.ports, state.ppm, state.hs)

    # Convert actions DataFrame to List[ActionLog]
    action_logs: List[ActionLog] = []
    if actions_df is not None and not actions_df.empty:
        for _, r in actions_df.iterrows():
            action_logs.append(
                ActionLog(
                    action=(str(r.get("ACTION", "")).lower() if pd.notna(r.get("ACTION")) else ""),
                    step=str(r.get("FLAG", "")) if pd.notna(r.get("FLAG")) else "",
                    trade_type=(str(r.get("ACTION")) if pd.notna(r.get("ACTION")) else None),
                    symbol=(str(r.get("SRC_SHARECODES")) if pd.notna(r.get("SRC_SHARECODES")) else None),
                    amount_thb=(float(r.get("AMOUNT")) if pd.notna(r.get("AMOUNT")) else None),
                    unit=None,
                    price=None,
                    asset_class=None,
                    notes=(
                        f"currency={r.get('CURRENCY')}; expected_weight={r.get('EXPECTED_WEIGHT')}"
                        if pd.notna(r.get("CURRENCY")) or pd.notna(r.get("EXPECTED_WEIGHT"))
                        else None
                    ),
                )
            )

    # Build proposed portfolio model
    positions: List[Position] = []
    proposed_portfolio_df = state.ports.df_out
    if proposed_portfolio_df is not None and not proposed_portfolio_df.empty:
        for _, r in proposed_portfolio_df.iterrows():
            positions.append(
                Position(
                    symbol=str(r.get("SYMBOL", r.get("SRC_SHARECODES", "UNKNOWN"))),
                    assetClass=str(r.get("ASSET_CLASS_NAME", "Cash and Cash Equivalent")),
                    unitBal=0.0,
                    unitPriceThb=1.0,
                    unitCostThb=1.0,
                    marketValue=float(r.get("VALUE", 0.0) or 0.0),
                    expectedReturn=float(r.get("EXPECTED_RETURN", 0.0) or 0.0),
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
