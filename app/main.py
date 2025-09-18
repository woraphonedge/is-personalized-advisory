"""
FastAPI application exposing endpoints to retrieve a customer’s portfolio and
perform a heuristic rebalancing.  Data is stored in an in‑memory pandas
DataFrame for demonstration purposes.

This API supports two primary endpoints:

* `GET /portfolio/{customer_id}` returns the current portfolio for a customer.
* `POST /rebalance` accepts a `RebalanceRequest` with configuration and
  returns a rebalanced portfolio along with a log of actions and the new
  health score.

The rebalancing logic is implemented in the `rebalancer` module.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

from .data_store import (
    lifespan,
)
from .models import (
    ActionLog,
    HealthMetrics,
    HealthMetricsRequest,
    RebalanceRequest,
    RebalanceRequestMock,
    RebalanceResponse,
)
from .utils.rebalancer_mock import compute_portfolio_health, propose_rebalance
from .utils.utils import convert_portfolio_to_df

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = FastAPI(title="Investment Rebalancing API", lifespan=lifespan)


@app.post("/api/v1/rebalance_mock", response_model=RebalanceResponse)
def rebalance_mock(request: RebalanceRequestMock) -> RebalanceResponse:
    """Rebalance a customer’s portfolio based on the provided request."""
    # Use provided portfolio or build from customer_id
    try:
        current = request.portfolio
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # Perform rebalance
    new_portfolio, actions = propose_rebalance(request, current, app.state.candidate_data)
    # Compute health metrics and extract score
    _metrics = compute_portfolio_health(
        new_portfolio,
        request.objective.target_alloc,
    )
    health = _metrics.score
    # Build response
    return RebalanceResponse(
        actions=[ActionLog(**a) for a in actions],
        portfolio=new_portfolio,
        health_score=health,
    )


@app.post("/api/v1/rebalance", response_model=RebalanceResponse)
def rebalance(request: RebalanceRequest) -> RebalanceResponse:
    """Rebalance a customer’s portfolio based on the provided request."""
    # Use provided portfolio or build from customer_id
    try:
        current = request.portfolio
        style = request.style
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    df_out = convert_portfolio_to_df(current)
    df_out["CUSTOMER_ID"] = request.customer_id
    logger.debug(f"df_out columns: {df_out.columns}")
    df_style_loaded = pd.DataFrame(style)
    df_out, df_style, port_ids, mapping = app.state.ports.create_portfolio_id(
        df_out, df_style_loaded, column_mapping=["AS_OF_DATE", "CUSTOMER_ID"]
    )

    app.state.ports.set_portfolio(df_out, df_style, port_ids, mapping)

    # Run v2 rebalancer -> pandas DataFrame of actions
    # Build a request-scoped Rebalancer configured from v2 constraints
    from app.utils.rebalancer import Rebalancer  # local import to avoid circular issues

    c = request.constraints
    # Try to infer as_of_date and customer_id from style if provided
    as_of_date = None
    customer_id_val = None
    try:
        if isinstance(style, list) and len(style) > 0:
            as_of_date = str(pd.DataFrame(style).get("AS_OF_DATE").iloc[0])
            customer_id_val = pd.to_numeric(pd.DataFrame(style).get("CUSTOMER_ID").iloc[0], errors="coerce")
        elif isinstance(style, dict):
            as_of_date = str(style.get("AS_OF_DATE")) if style.get("AS_OF_DATE") is not None else None
            customer_id_val = pd.to_numeric(style.get("CUSTOMER_ID"), errors="coerce") if style.get("CUSTOMER_ID") is not None else None
    except Exception:
        pass

    rb_local = Rebalancer(
        as_of_date=as_of_date or None,
        customer_id=int(customer_id_val) if customer_id_val is not None and not pd.isna(customer_id_val) else None,
        new_money=request.objective.new_money,
        discretionary_percent=c.discretionary_percent,
        private_percent=c.private_percent,
        cash_percent=c.cash_percent,
        offshore_percent=c.offshore_percent,
        product_restriction=c.product_restriction,
        discretionary_acceptance=c.discretionary_acceptance,
    )

    actions_df = rb_local.rebalance(app.state.ports, app.state.ppm, app.state.hs)
    proposed_portfolio_df = app.state.ports.df_out

    # Convert actions DataFrame to List[ActionLog]
    action_logs: list[ActionLog] = []
    if actions_df is not None and not actions_df.empty:
        for _, r in actions_df.iterrows():
            # Map DataFrame columns to ActionLog fields
            action_logs.append(
                ActionLog(
                    action=(
                        str(r.get("ACTION", "")).lower()
                        if pd.notna(r.get("ACTION"))
                        else ""
                    ),
                    step=str(r.get("FLAG", "")) if pd.notna(r.get("FLAG")) else "",
                    trade_type=(
                        str(r.get("ACTION")) if pd.notna(r.get("ACTION")) else None
                    ),
                    symbol=(
                        str(r.get("SRC_SHARECODES"))
                        if pd.notna(r.get("SRC_SHARECODES"))
                        else None
                    ),
                    amount_thb=(
                        float(r.get("AMOUNT")) if pd.notna(r.get("AMOUNT")) else None
                    ),
                    unit=None,
                    price=None,
                    asset_class=None,
                    notes=(
                        f"currency={r.get('CURRENCY')}; expected_weight={r.get('EXPECTED_WEIGHT')}"
                        if pd.notna(r.get("CURRENCY"))
                        or pd.notna(r.get("EXPECTED_WEIGHT"))
                        else None
                    ),
                )
            )

    # Convert proposed_portfolio_df to Portfolio model
    from models import Portfolio, Position  # local import to avoid circular issues

    positions: list[Position] = []
    if proposed_portfolio_df is not None and not proposed_portfolio_df.empty:
        for _, r in proposed_portfolio_df.iterrows():
            positions.append(
                Position(
                    symbol=str(r.get("SYMBOL", r.get("SRC_SHARECODES", "UNKNOWN"))),
                    assetClass=str(
                        r.get("ASSET_CLASS_NAME", "Cash and Cash Equivalent")
                    ),
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


@app.post("/api/v1/health-score", response_model=HealthMetrics)
def get_health_score(request: HealthMetricsRequest) -> HealthMetrics:
    """Calculate and return only the health score for a customer's current portfolio."""
    # Use provided portfolio or build from customer_id
    try:
        current = request.portfolio
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # Compute health metrics for current portfolio
    metrics = compute_portfolio_health(
        current,
        request.target_alloc,
    )
    # Return full metrics as typed model
    return metrics


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)


def run_server():
    """CLI entry point for running the server."""
    import uvicorn
    PORT = os.getenv("PORT", "8100")
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(PORT), reload=True)
