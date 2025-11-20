from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

from fastapi import HTTPException

from app.models import HealthDetailMetrics, HealthMetrics
from app.utils.portfolios_service import PortfolioService

# Style mapping from user-friendly names to PortProp model names
STYLE_MAP = {
    'Bulletproof': 'Conservative',
    'Conservative': 'Conservative',
    'Moderate Low Risk': 'Medium to Moderate Low Risk',
    'Moderate High Risk': 'Medium to Moderate High Risk',
    'High Risk': 'High Risk',
    'Aggressive Growth': 'Aggressive',
    'Unwavering': 'Aggressive'
}


@contextmanager
def override_client_style(ports, customer_id: int, client_style: Optional[str] = None):
    """Context manager to temporarily override client investment style in df_style.

    This allows dynamic health score and rebalance calculations with different
    investment styles without modifying the stored data permanently.

    Args:
        ports: Portfolios instance with df_style DataFrame
        customer_id: Customer ID to override style for
        client_style: New investment style to apply (e.g., 'High Risk', 'Conservative')

    Yields:
        None

    Example:
        with override_client_style(app.state.ports, 14055, "High Risk"):
            # df_style is temporarily modified
            health_metrics = get_health_metrics_for_customer(...)
        # df_style is automatically restored here
    """
    import logging
    logger = logging.getLogger(__name__)

    original_df_style = None

    try:
        if client_style:
            # Save original df_style for restoration
            original_df_style = ports.df_style.copy()

            # Override port_investment_style for the customer's portfolio
            mask = ports.df_style["port_id"].isin(
                ports.port_id_mapping[
                    ports.port_id_mapping["customer_id"] == customer_id
                ]["port_id"]
            )
            ports.df_style.loc[mask, "port_investment_style"] = client_style

            # Remap to portpop_styles using the style_map
            ports.df_style.loc[mask, "portpop_styles"] = (
                ports.df_style.loc[mask, "port_investment_style"].map(STYLE_MAP)
            )
            logger.debug(
                "Temporarily overrode client_style to %s for customer_id=%s",
                client_style, customer_id
            )

        yield

    finally:
        # Restore original df_style to ensure modification is only for this context
        if original_df_style is not None:
            ports.df_style = original_df_style
            logger.debug("Restored original df_style after operation")


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
            }
        except Exception:
            asset_allocation = {}

        # Model asset allocation (advisory model based on client investment style)
        try:
            model_alloc_df = client_port.get_model_asset_allocation_lookthrough(ppm)

            if model_alloc_df is None or len(model_alloc_df) == 0:
                raise ValueError(
                    f"No model allocation found for customer_id={customer_id}. "
                    "This may indicate missing style mapping or model configuration."
                )

            mrow = model_alloc_df.iloc[0]

            # Validate required columns exist
            required_cols = ["aa_cash_model", "aa_fi_model", "aa_le_model", "aa_ge_model", "aa_alt_model"]
            missing_cols = [col for col in required_cols if col not in mrow.index]
            if missing_cols:
                raise ValueError(
                    f"Model allocation missing required columns: {missing_cols}. "
                    f"Available columns: {list(mrow.index)}"
                )

            model_asset_allocation = {
                "Cash and Cash Equivalent": float(mrow["aa_cash_model"]),
                "Fixed Income": float(mrow["aa_fi_model"]),
                "Local Equity": float(mrow["aa_le_model"]),
                "Global Equity": float(mrow["aa_ge_model"]),
                "Alternative": float(mrow["aa_alt_model"]),
            }
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                "Failed to retrieve model asset allocation for customer_id=%s: %s",
                customer_id, str(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve advisory model allocation: {str(e)}"
            ) from e

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
            model_asset_allocation=model_asset_allocation,
        )

        return HealthMetrics(
            score=float(row.get("health_score", 0.0) or 0.0),
            metrics=detail,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
