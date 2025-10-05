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
from fastapi import FastAPI, HTTPException, Request

from app.utils.portfolio_fetcher import get_portfolio_for_customer

load_dotenv()

# Configure logging early so import-time logs from app modules respect LOG_LEVEL
level_name = os.getenv("LOG_LEVEL", "DEBUG").upper()
level = getattr(logging, level_name, logging.DEBUG)
logging.basicConfig(level=level)
logger = logging.getLogger(__name__)

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

app = FastAPI(title="Investment Rebalancing API", lifespan=lifespan)


@app.get("/health")
def health_check():
    """Lightweight health check endpoint."""
    return {"status": "ok"}


@app.get("/api/v1/portfolio/{customer_id}")
def get_portfolio(customer_id: str, request: Request):
    """Return the current portfolio for the specified customer.

    This pulls client positions as of the configured date using
    `app.state.ports_repo.load_client_out_product_enriched` and maps them
    to the public `Portfolio` model with camelCase field aliases.
    """
    # Trace request context for debugging
    try:
        client_host = (
            getattr(request.client, "host", "unknown")
            if request and request.client
            else "unknown"
        )
        logger.info(
            "GET %s from %s | raw customer_id=%r",
            request.url.path if request else "/api/v1/portfolio",
            client_host,
            customer_id,
        )
    except Exception:
        # Best-effort; do not block request on logging issues
        pass

    try:
        # Ensure numeric id if underlying storage expects it
        cust_id_int = int(customer_id)
        logger.debug("Parsed customer_id to int successfully: %s", cust_id_int)
    except ValueError as e:
        logger.warning(
            "Invalid customer_id, must be numeric: %r | error=%s", customer_id, e
        )
        raise HTTPException(
            status_code=400, detail="customer_id must be numeric"
        ) from e

    try:
        # Delegate to utility to build Portfolio model
        portfolio_model = get_portfolio_for_customer(app.state.ports, cust_id_int)
        return portfolio_model
    except HTTPException:
        # re-raise expected errors
        raise
    except Exception as e:
        logger.exception("Failed to build portfolio for customer_id=%s", customer_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/rebalance_mock", response_model=RebalanceResponse)
def rebalance_mock(request: RebalanceRequestMock) -> RebalanceResponse:
    """Rebalance a customer’s portfolio based on the provided request."""
    # Use provided portfolio or build from customer_id
    try:
        current = request.portfolio
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # Perform rebalance
    new_portfolio, actions = propose_rebalance(
        request, current, app.state.candidate_data
    )
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
    try:
        from app.utils.rebalance_handler import perform_rebalance
        return perform_rebalance(app.state, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
