from __future__ import annotations

import copy
import logging
from typing import Any, List

import pandas as pd

from app.models import (
    ActionLog,
    HealthMetrics,
    Portfolio,
    Position,
    RebalanceRequest,
    RebalanceResponse,
)
from app.utils.health_service import STYLE_MAP, override_client_style
from app.utils.portfolios_service import PortfolioService
from app.utils.rebalancer_mock import compute_portfolio_health
from app.utils.utils import convert_portfolio_to_df

logger = logging.getLogger(__name__)


def _build_df_style(
    customer_id: int, style: str | dict[str, Any] | list[dict[str, Any] | str] | None
) -> pd.DataFrame:
    """Normalize style payload into a single-row DataFrame.

    Args:
        customer_id: Customer ID to associate with the style
        style: Investment style in various formats:
            - str: Direct style name (e.g., "High Risk")
            - dict: Style object with keys like "client_style", "INVESTMENT_STYLE"
            - list: List of style dicts or strings (uses first element)
            - None: Uses default "High Risk"

    Returns:
        Single-row DataFrame with columns:
            - customer_id (int)
            - port_investment_style (str): Original style label
            - portpop_style (str): Mapped style for PortProp system

    Raises:
        ValueError: If customer_id is invalid
    """
    # Validate customer_id
    if not isinstance(customer_id, int) or customer_id <= 0:
        raise ValueError(
            f"Invalid customer_id: {customer_id}. Must be a positive integer."
        )

    # Extract style label from various input formats
    label: str | None = None

    if style is None:
        label = None
    elif isinstance(style, str):
        label = style.strip() if style.strip() else None
    elif isinstance(style, dict):
        # Try multiple possible keys for style
        label = (
            style.get("client_style")
            or style.get("style")
            or style.get("INVESTMENT_STYLE")
            or style.get("INVESTMENT_STYLE_AUMX")
        )
    elif isinstance(style, list) and len(style) > 0:
        first = style[0]
        if isinstance(first, str):
            label = first.strip() if first.strip() else None
        elif isinstance(first, dict):
            label = (
                first.get("client_style")
                or first.get("style")
                or first.get("INVESTMENT_STYLE")
                or first.get("INVESTMENT_STYLE_AUMX")
            )

    # Default to "High Risk" if no valid label found
    if not label:
        label = "High Risk"
        logger.debug(
            "No valid style provided for customer_id=%s, defaulting to 'High Risk'",
            customer_id,
        )

    # Map to PortProp style name
    portpop_style = STYLE_MAP.get(label, "High Risk")
    if label not in STYLE_MAP:
        logger.warning(
            "Unknown investment style '%s' for customer_id=%s, mapping to 'High Risk'",
            label,
            customer_id,
        )

    data = {
        "customer_id": customer_id,
        "port_investment_style": label,
        "portpop_style": portpop_style,
    }

    return pd.DataFrame([data])


def _enrich_portfolio_data(
    df_portfolio: pd.DataFrame, product_mapping: pd.DataFrame | None
) -> pd.DataFrame:
    """Enrich portfolio data with product information from product_mapping.

    Args:
        df_portfolio: Portfolio DataFrame with positions
        product_mapping: Product mapping DataFrame with enrichment data

    Returns:
        Enriched DataFrame with symbol, asset_class_name, product_type_desc, etc.

    Raises:
        ValueError: If portfolio data validation fails or mapping is incomplete
    """
    required_cols = [
        "product_id",
        "src_sharecodes",
        "desk",
        "port_type",
        "currency",
        "sec_id",
    ]
    enriched_cols = ["asset_class_name", "product_type_desc", "symbol", "sec_id"]

    # Check if data is already enriched (from portfolio_fetcher)
    is_already_enriched = all(
        col in df_portfolio.columns and not df_portfolio[col].isna().all()
        for col in enriched_cols
    )

    if is_already_enriched:
        # Ensure all required columns exist
        for c in required_cols:
            if c not in df_portfolio.columns:
                df_portfolio[c] = pd.NA

        # Validate enriched data has no missing values
        enriched_na_mask = df_portfolio[enriched_cols].isna().any(axis=1)
        if enriched_na_mask.any():
            na_count = enriched_na_mask.sum()
            na_rows = df_portfolio.loc[
                enriched_na_mask, ["product_id", "src_sharecodes"]
            ]
            raise ValueError(
                f"Portfolio data validation failed: {na_count} positions have missing enriched data. "
                f"Missing data in columns: {enriched_cols}. "
                f"Affected products: {na_rows['product_id'].tolist()}"
            )
        return df_portfolio

    # Need to enrich from product_mapping
    for c in required_cols:
        if c not in df_portfolio.columns:
            df_portfolio[c] = pd.NA

    mask_missing = df_portfolio[required_cols].isna().any(axis=1)

    if product_mapping is None:
        raise ValueError("product_mapping not loaded in server state")

    if not mask_missing.any():
        # All required columns already filled
        return df_portfolio

    # Prepare product mapping subset
    cols_map = list(
        set(
            required_cols
            + [
                "symbol",
                "product_display_name",
                "product_type_desc",
                "asset_class_name",
            ]
        )
    )
    pm_sub = product_mapping[cols_map].copy()

    # Try exact match with sec_id
    use_keys = ["sec_id"]
    df_portfolio["sec_id"] = pd.to_numeric(df_portfolio["sec_id"], errors="coerce").astype("Int64")
    pm_sub["sec_id"] = pd.to_numeric(pm_sub["sec_id"], errors="coerce").astype("Int64")

    merged = df_portfolio.merge(
        pm_sub, on=use_keys, how="left", suffixes=("", "_pm"), indicator=True
    )

    # Validate all positions were successfully mapped
    still_missing = merged["_merge"] == "left_only"
    if still_missing.any():
        missing_count = still_missing.sum()
        missing_rows = df_portfolio.loc[
            still_missing,
            ["product_id", "src_sharecodes", "desk", "currency"],
        ]
        raise ValueError(
            f"Product mapping failed: {missing_count} positions could not be mapped to product_mapping. "
            f"This indicates missing or invalid product data. "
            f"Unmapped products: {missing_rows.to_dict('records')}"
        )

    merged.drop(columns="_merge", inplace=True)

    # Apply mapped values to main columns
    for c in required_cols:
        pm_col = f"{c}_pm"
        if pm_col in merged.columns:
            if c in merged.columns:
                merged[c] = merged[c].fillna(merged[pm_col])
            else:
                merged[c] = merged[pm_col]
            merged.drop(columns=pm_col, inplace=True)

    for c in [
        "symbol",
        "product_display_name",
        "product_type_desc",
        "asset_class_name",
    ]:
        pm_col = f"{c}_pm"
        if pm_col in merged.columns:
            if c in merged.columns:
                merged[c] = merged[c].fillna(merged[pm_col])
            else:
                merged[c] = merged[pm_col]
            merged.drop(columns=pm_col, inplace=True)

    df_out = merged[
        df_portfolio.columns.union(
            [
                "symbol",
                "product_display_name",
                "product_type_desc",
                "asset_class_name",
            ],
            sort=False,
        )
    ]

    # Validate mapping was successful - all required columns should be filled
    mask_missing_after = df_out[required_cols].isna().any(axis=1)
    if mask_missing_after.any():
        missing_count = mask_missing_after.sum()
        sample = df_out.loc[
            mask_missing_after, required_cols + ["src_sharecodes", "symbol"]
        ].head(5)
        raise ValueError(
            f"Product enrichment incomplete: {missing_count} positions still have missing required data after mapping. "
            f"Sample rows: {sample.to_dict('records')}"
        )

    return df_out


def _create_rebalancer(state: Any, request: RebalanceRequest) -> Any:
    """Create and configure a Rebalancer instance from request parameters.

    Args:
        state: FastAPI app.state with shared rebalancer references
        request: RebalanceRequest with constraints and objectives

    Returns:
        Configured Rebalancer instance ready to execute rebalancing

    Raises:
        ValueError: If constraints are invalid
    """
    from app.utils.rebalancer import Rebalancer

    c = request.constraints

    # Try to infer optional fields from legacy style field
    customer_id_val = None
    try:
        if isinstance(request.style, list) and len(request.style) > 0:
            tmp = pd.DataFrame(request.style)
            customer_id_val = (
                pd.to_numeric(tmp.get("CUSTOMER_ID").iloc[0], errors="coerce")
                if "CUSTOMER_ID" in tmp
                else None
            )
        elif isinstance(request.style, dict):
            customer_id_val = (
                pd.to_numeric(request.style.get("CUSTOMER_ID"), errors="coerce")
                if request.style.get("CUSTOMER_ID") is not None
                else None
            )
    except Exception:
        pass

    # Map legacy product_restriction to blacklist if new fields not provided
    product_whitelist = getattr(c, "product_whitelist", None)
    product_blacklist = getattr(c, "product_blacklist", None)
    if (not product_blacklist) and c.product_restriction:
        product_blacklist = c.product_restriction

    # Create rebalancer with validated parameters
    rb_local = Rebalancer(
        customer_id=(
            int(customer_id_val)
            if customer_id_val is not None and not pd.isna(customer_id_val)
            else None
        ),
        client_investment_style=request.objective.client_style,
        new_money=request.objective.new_money,
        discretionary_acceptance=c.discretionary_acceptance,
        client_classification=c.client_classification,
        private_percent=c.private_percent,
        cash_percent=c.cash_percent,
        offshore_percent=c.offshore_percent,
        product_whitelist=product_whitelist,
        product_blacklist=product_blacklist,
    )

    # Copy reference tables from shared state rebalancer if available
    try:
        rb_state = getattr(state, "rb", None)
        if rb_state is not None:
            refs = {
                "es_sell_list": getattr(rb_state, "es_sell_list", None),
                "product_recommendation_rank_raw": getattr(
                    rb_state, "prod_reco_rank_raw", None
                ),
                "mandate_allocation": getattr(
                    rb_state, "discretionary_allo_weight", None
                ),
            }
            rb_local.set_ref_tables(refs)
    except Exception:
        logger.debug(
            "Failed to copy rebalancer refs from state; proceeding with whatever is loaded"
        )

    return rb_local


def _compute_health_metrics(
    new_ports: Any,
    actions_df: pd.DataFrame | None,
    state: Any,
    request: RebalanceRequest,
) -> tuple[Portfolio, List[ActionLog], HealthMetrics]:
    """Compute health metrics for the proposed portfolio.

    Args:
        new_ports: Rebalanced portfolio (Portfolios instance)
        actions_df: DataFrame with rebalance actions
        state: FastAPI app.state with ppm, hs
        request: RebalanceRequest with target allocation

    Returns:
        Tuple of (proposed_portfolio_model, action_logs, health_metrics)

    Raises:
        Exception: If health metrics calculation fails completely
    """

    # Helper functions for handling NA values safely
    def safe_str(val, default="UNKNOWN"):
        if pd.isna(val) or val is None:
            return default
        return str(val)

    def safe_float(val, default=0.0):
        if pd.isna(val) or val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Build action logs from rebalance actions
    action_logs: List[ActionLog] = []
    if actions_df is not None and not actions_df.empty:
        for _, r in actions_df.iterrows():
            action_logs.append(
                ActionLog(
                    action=safe_str(r.get("action"), "trade"),  # buy, sell, funding
                    flag=safe_str(
                        r.get("flag"), None
                    ),  # new_money, not_monitored_product, etc.
                    flag_msg=safe_str(
                        r.get("flag_msg"), None
                    ),  # Human-readable description
                    symbol=safe_str(r.get("src_sharecodes"), "UNKNOWN"),
                    product_display_name=safe_str(
                        r.get("product_display_name"), "UNKNOWN"
                    ),
                    amount=safe_float(r.get("amount")),  # Transaction amount
                    asset_class_name=safe_str(
                        r.get("asset_class_name"), None
                    ),  # Asset class
                    # Legacy fields for backward compatibility
                    step="",
                    trade_type=safe_str(r.get("action"), "trade"),
                    amount_thb=safe_float(r.get("amount")),
                    unit=None,
                    price=None,
                    asset_class=safe_str(r.get("asset_class_name"), "Unknown"),
                    notes=safe_str(r.get("flag_msg"), None),
                )
            )

    # Build proposed portfolio model from rebalanced portfolio DataFrame
    positions: List[Position] = []
    proposed_portfolio_df = getattr(new_ports, "df_out", None)
    if proposed_portfolio_df is not None and not proposed_portfolio_df.empty:
        for _, r in proposed_portfolio_df.iterrows():
            _sym = r.get("symbol")
            try:
                _sym_is_na = pd.isna(_sym)
            except Exception:
                _sym_is_na = False
            symbol_val = (
                r.get("src_sharecodes")
                if (_sym_is_na or (_sym is None) or (_sym == ""))
                else _sym
            )
            if symbol_val is None or (not isinstance(symbol_val, str)):
                symbol_val = str(symbol_val) if symbol_val is not None else "UNKNOWN"

            positions.append(
                Position(
                    secId=r.get("sec_id"),
                    productId=safe_str(r.get("product_id")),
                    desk=safe_str(r.get("desk")),
                    portType=safe_str(r.get("port_type")),
                    currency=safe_str(r.get("currency")),
                    symbol=symbol_val,
                    srcSharecodes=safe_str(r.get("src_sharecodes"), ""),
                    productDisplayName=safe_str(r.get("product_display_name")),
                    productTypeDesc=safe_str(r.get("product_type_desc")),
                    coveragePrdtype=safe_str(r.get("coverage_prdtype")),
                    isRiskyAsset=bool(r.get("is_risky_asset", False)),
                    isCoverage=bool(r.get("is_coverage", False)),
                    esCorePort=bool(r.get("es_core_port", False)),
                    esSellList=safe_str(r.get("es_sell_list"), None),
                    flagTopPick=safe_str(r.get("flag_top_pick")),
                    flagTaxSaving=safe_str(r.get("flag_tax_saving"), None),
                    assetClass=safe_str(r.get("asset_class_name")),
                    assetSubClass=safe_str(r.get("asset_sub_class"), None),
                    unitBal=0.0,
                    unitPriceThb=1.0,
                    unitCostThb=1.0,
                    marketValue=safe_float(r.get("value")),
                    expectedReturn=safe_float(r.get("expected_return")),
                    expectedIncomeYield=0.0,
                    volatility=0.0,
                    isMonitored=True,
                    exposures=None,
                )
            )
    proposed_portfolio_model = Portfolio(positions=positions)

    # Compute health metrics on proposed portfolio
    try:
        df_health, _comp = new_ports.get_portfolio_health_score(
            state.ppm, state.hs, cal_comp=True
        )

        if df_health is None or len(df_health) == 0:
            logger.warning(
                "No health metrics returned from proposed portfolio, using mock"
            )
            _metrics = compute_portfolio_health(
                proposed_portfolio_model,
                request.objective.target_alloc,
            )
            health = _metrics.score
            health_metrics = HealthMetrics(score=health, metrics=_metrics.metrics)
        else:
            row = df_health.iloc[0]

            # Current asset allocation (lookthrough)
            try:
                alloc_df = new_ports.get_portfolio_asset_allocation_lookthrough(
                    state.ppm
                )
                arow = (
                    alloc_df.iloc[0]
                    if alloc_df is not None and len(alloc_df) > 0
                    else None
                )
                asset_allocation = {
                    "Cash and Cash Equivalent": (
                        float(arow["aa_cash"]) if arow is not None else 0.0
                    ),
                    "Fixed Income": float(arow["aa_fi"]) if arow is not None else 0.0,
                    "Local Equity": float(arow["aa_le"]) if arow is not None else 0.0,
                    "Global Equity": float(arow["aa_ge"]) if arow is not None else 0.0,
                    "Alternative": float(arow["aa_alt"]) if arow is not None else 0.0,
                }
            except Exception:
                asset_allocation = {}

            # Model asset allocation (advisory model based on client investment style)
            try:
                model_alloc_df = new_ports.get_model_asset_allocation_lookthrough(
                    state.ppm
                )
                mrow = (
                    model_alloc_df.iloc[0]
                    if model_alloc_df is not None and len(model_alloc_df) > 0
                    else None
                )
                model_asset_allocation = {
                    "Cash and Cash Equivalent": (
                        float(mrow["aa_cash_model"]) if mrow is not None else 0.0
                    ),
                    "Fixed Income": (
                        float(mrow["aa_fi_model"]) if mrow is not None else 0.0
                    ),
                    "Local Equity": (
                        float(mrow["aa_le_model"]) if mrow is not None else 0.0
                    ),
                    "Global Equity": (
                        float(mrow["aa_ge_model"]) if mrow is not None else 0.0
                    ),
                    "Alternative": (
                        float(mrow["aa_alt_model"]) if mrow is not None else 0.0
                    ),
                }
            except Exception as e:
                logger.error(
                    "Failed to retrieve model asset allocation for proposed portfolio: %s",
                    e,
                )
                model_asset_allocation = {}

            from app.models import HealthDetailMetrics

            detail = HealthDetailMetrics(
                port_id=int(row.get("port_id", 0) or 0),
                expected_return=float(row.get("expected_return", 0.0) or 0.0),
                expected_return_model=float(
                    row.get("expected_return_model", 0.0) or 0.0
                ),
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
                score_non_cover_global_stock=int(
                    row.get("score_non_cover_global_stock", 0) or 0
                ),
                score_non_cover_local_stock=int(
                    row.get("score_non_cover_local_stock", 0) or 0
                ),
                score_non_cover_mutual_fund=int(
                    row.get("score_non_cover_mutual_fund", 0) or 0
                ),
                score_not_monitored_product=float(
                    row.get("score_not_monitored_product", 0.0) or 0.0
                ),
                asset_allocation=asset_allocation,
                model_asset_allocation=model_asset_allocation,
            )

            health = float(row.get("health_score", 0.0) or 0.0)
            health_metrics = HealthMetrics(score=health, metrics=detail)

    except Exception as e:
        logger.exception(
            "Failed to compute real health score, falling back to mock: %s", e
        )
        _metrics = compute_portfolio_health(
            proposed_portfolio_model,
            request.objective.target_alloc,
        )
        health = _metrics.score
        health_metrics = HealthMetrics(score=health, metrics=_metrics.metrics)

    return proposed_portfolio_model, action_logs, health_metrics


def perform_rebalance(state: Any, request: RebalanceRequest) -> RebalanceResponse:
    """Core rebalancing logic extracted from FastAPI route.

    Parameters
    - state: FastAPI app.state providing ports, ppm, hs
    - request: validated RebalanceRequest
    """
    # Extract client_style from request for style override during rebalance and health calculation
    client_style = request.objective.client_style if request.objective else None

    # Wrap entire rebalance logic with style override context
    with override_client_style(state.ports, request.customer_id, client_style):
        return _perform_rebalance_inner(state, request)


def _perform_rebalance_inner(
    state: Any, request: RebalanceRequest
) -> RebalanceResponse:
    """Inner rebalance logic that runs within the style override context."""
    # Convert incoming portfolio to df and tag with customer id
    try:
        df_out = convert_portfolio_to_df(request.portfolio)
        logger.debug(
            "converted portfolio df columns=%s rows=%d",
            list(df_out.columns),
            len(df_out),
        )
    except Exception as e:
        logger.exception("Failed to convert portfolio to DataFrame: %s", e)
        raise
    df_out["customer_id"] = request.customer_id
    df_out["port_id_mapping"] = request.customer_id

    # Build style DataFrame
    # Prefer client_style from objective (new field), fallback to legacy style field
    style_value = request.objective.client_style if request.objective else request.style
    logger.debug(
        f"Style from request.style={request.style}, request.objective.client_style={request.objective.client_style if request.objective else None}"
    )
    df_style_loaded = _build_df_style(request.customer_id, style_value)

    # Create a shallow copy of ports for this rebalance operation
    # Deep copy is unnecessary as we only need to modify the portfolio data, not reference tables
    ports = copy.copy(state.ports)
    try:
        pm = getattr(ports, "product_mapping", None)
        df_out = _enrich_portfolio_data(df_out, pm)
    except Exception:
        logger.exception("Enrichment with product_mapping failed")
        raise

    df_out, df_style, port_ids, mapping = ports.create_portfolio_id(
        df_out, df_style_loaded, column_mapping=["customer_id"]
    )
    logger.debug(
        "created portfolio ids columns=%s rows=%d", list(df_out.columns), len(df_out)
    )

    ports.set_portfolio(df_out, df_style, port_ids, mapping)
    port_service = PortfolioService(ports)
    port = port_service.get_client_portfolio(request.customer_id)

    # Create and configure rebalancer
    rb_local = _create_rebalancer(state, request)

    # Execute rebalancing
    try:
        new_ports, actions_df = rb_local.rebalance(port, state.ppm, state.hs)
    except Exception as e:
        logger.exception("Rebalance failed: %s", e)
        raise

    # Compute health metrics and build response
    proposed_portfolio_model, action_logs, health_metrics = _compute_health_metrics(
        new_ports, actions_df, state, request
    )

    return RebalanceResponse(
        actions=action_logs,
        portfolio=proposed_portfolio_model,
        health_score=health_metrics.score,
        health_metrics=health_metrics,
    )
