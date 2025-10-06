from __future__ import annotations

from fastapi import HTTPException

from app.models import HealthDetailMetrics, HealthMetrics
from app.utils.portfolios_service import PortfolioService


def get_health_metrics_for_customer(ports, ppm, hs, customer_id: int) -> HealthMetrics:
    """Compute notebook-style health metrics for a given customer.

    Uses shared singletons (ports, ppm, hs) prepared in app.data_store to slice
    the client's portfolio and compute health metrics aligned to the notebook
    output. Returns a HealthMetrics object with top-level score and detailed
    components in metrics.
    """
    try:
        svc = PortfolioService(ports)
        client_port = svc.get_client_portfolio(customer_id=customer_id)

        # Compute health score and components
        df_health, _comp = client_port.get_portfolio_health_score(ppm, hs, cal_comp=True)
        if df_health is None or len(df_health) == 0:
            raise HTTPException(status_code=404, detail="No health metrics available for this customer")

        row = df_health.iloc[0]

        # Current asset allocation (lookthrough)
        try:
            alloc_df = client_port.get_portfolio_asset_allocation_lookthrough(ppm)
            arow = alloc_df.iloc[0] if alloc_df is not None and len(alloc_df) > 0 else None
            asset_allocation = {
                "Cash and Cash Equivalent": float(arow["aa_cash"]) if arow is not None else 0.0,
                "Fixed Income": float(arow["aa_fi"]) if arow is not None else 0.0,
                "Local Equity": float(arow["aa_le"]) if arow is not None else 0.0,
                "Global Equity": float(arow["aa_ge"]) if arow is not None else 0.0,
                "Alternative": float(arow["aa_alt"]) if arow is not None else 0.0,
                "Allocation": 0.0,
            }
        except Exception:
            asset_allocation = {}

        detail = HealthDetailMetrics(
            port_id=int(row.get("port_id", 0) or 0),
            expected_return=float(row.get("expected_return", 0.0) or 0.0),
            expected_return_model=float(row.get("expected_return_model", 0.0) or 0.0),
            score_ret=int(row.get("score_ret", 0) or 0),
            volatility=float(row.get("volatility", 0.0) or 0.0),
            volatility_model=float(row.get("volatility_model", 0.0) or 0.0),
            score_vol=int(row.get("score_vol", 0) or 0),
            score_portfolio_risk=int(row.get("score_portfolio_risk", 0) or 0),
            acd=float(row.get("acd", 0.0) or 0.0),
            score_acd=int(row.get("score_acd", 0) or 0),
            ged=float(row.get("ged", 0.0) or 0.0),
            score_ged=int(row.get("score_ged", 0) or 0),
            score_diversification=int(row.get("score_diversification", 0) or 0),
            score_bulk_risk=int(row.get("score_bulk_risk", 0) or 0),
            score_issuer_risk=int(row.get("score_issuer_risk", 0) or 0),
            score_non_cover_global_stock=int(row.get("score_non_cover_global_stock", 0) or 0),
            score_non_cover_local_stock=int(row.get("score_non_cover_local_stock", 0) or 0),
            score_non_cover_mutual_fund=int(row.get("score_non_cover_mutual_fund", 0) or 0),
            score_not_monitored_product=float(row.get("score_not_monitored_product", 0.0) or 0.0),
            asset_allocation=asset_allocation,
        )

        return HealthMetrics(
            score=float(row.get("health_score", 0.0) or 0.0),
            metrics=detail,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
