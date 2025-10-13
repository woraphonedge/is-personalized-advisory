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
        "Bulletproof": "Conservative",
        "Conservative": "Conservative",
        "Moderate Low Risk": "Medium to Moderate Low Risk",
        "Moderate High Risk": "Medium to Moderate High Risk",
        "High Risk": "High Risk",
        "Aggressive Growth": "Aggressive",
        "Unwavering": "Aggressive",
    }

    # Derive a string label from supported inputs
    label = None
    try:
        if isinstance(style, str):
            label = style
        elif isinstance(style, dict):
            label = (
                style.get("INVESTMENT_STYLE_AUMX")
                or style.get("INVESTMENT_STYLE")
                or style.get("style")
                or style.get("client_style")
            )
        elif isinstance(style, list) and len(style) > 0:
            first = style[0]
            if isinstance(first, str):
                label = first
            elif isinstance(first, dict):
                label = (
                    first.get("INVESTMENT_STYLE_AUMX")
                    or first.get("INVESTMENT_STYLE")
                    or first.get("style")
                    or first.get("client_style")
                )
    except Exception:
        # Best-effort parsing only
        pass

    if not label:
        label = "High Risk"

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
    df_style_loaded = _build_df_style(request.customer_id, request.style)

    # Create/assign portfolio ids
    # deepcopy ports to avoid modifying the original state
    ports = copy.deepcopy(state.ports)
    try:
        pm = getattr(ports, "product_mapping", None)
        required_cols = [
            "product_id",
            "src_sharecodes",
            "desk",
            "port_type",
            "currency",
        ]

        # Check if data is already enriched (from portfolio_fetcher)
        enriched_cols = ["asset_class_name", "product_type_desc", "symbol"]
        is_already_enriched = all(
            col in df_out.columns and not df_out[col].isna().all()
            for col in enriched_cols
        )

        logger.debug(f"[TEMP-DEBUG] Portfolio enrichment check:")
        logger.debug(
            f"[TEMP-DEBUG] - Required cols present: {[c for c in required_cols if c in df_out.columns]}"
        )
        logger.debug(
            f"[TEMP-DEBUG] - Enriched cols present: {[c for c in enriched_cols if c in df_out.columns]}"
        )
        logger.debug(f"[TEMP-DEBUG] - Is already enriched: {is_already_enriched}")

        if is_already_enriched:
            logger.debug(
                "[TEMP-DEBUG] Portfolio is already enriched, skipping product mapping"
            )
            # Ensure all required columns exist even if already enriched
            for c in required_cols:
                if c not in df_out.columns:
                    df_out[c] = pd.NA

            # Since frontend now sends complete data, we expect no NA values in enriched columns
            enriched_na_mask = df_out[enriched_cols].isna().any(axis=1)
            if enriched_na_mask.any():
                logger.warning(
                    f"[UNEXPECTED] Found {enriched_na_mask.sum()} rows with NA in enriched columns despite being marked as enriched"
                )
                # Log the problematic rows for debugging
                na_rows = df_out.loc[
                    enriched_na_mask,
                    ["product_id", "src_sharecodes", "desk"] + enriched_cols,
                ]
                logger.warning(f"Rows with NA values:\n{na_rows}")
                is_already_enriched = False  # Force mapping as fallback

        if not is_already_enriched:
            # Original product mapping logic for non-enriched data
            for c in required_cols:
                if c not in df_out.columns:
                    df_out[c] = pd.NA
            mask_missing = df_out[required_cols].isna().any(axis=1)
            if pm is None:
                raise ValueError("product_mapping not loaded in server state")
            if mask_missing.any():
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
                pm_sub = pm[cols_map].copy()
                missing_rows = df_out.loc[mask_missing]

                # First try exact match with src_sharecodes + product_id + currency
                use_keys = ["src_sharecodes", "product_id", "currency"]
                logger.debug(f"[TEMP-DEBUG] Before product mapping merge:")
                logger.debug(
                    f"[TEMP-DEBUG] - df_out sample keys: {df_out[use_keys].head(3) if all(k in df_out.columns for k in use_keys) else 'MISSING KEYS'}"
                )
                logger.debug(
                    f"[TEMP-DEBUG] - pm_sub sample keys: {pm_sub[use_keys].head(3) if all(k in pm_sub.columns for k in use_keys) else 'MISSING KEYS'}"
                )

                merged = df_out.merge(
                    pm_sub, on=use_keys, how="left", suffixes=("", "_pm")
                )
                logger.debug(
                    f"[TEMP-DEBUG] After first merge - rows with missing symbol_pm: {merged['symbol_pm'].isna().sum()}/{len(merged)}"
                )

                # Check if any positions still need mapping after initial merge
                still_missing = (
                    merged[
                        [
                            f"{c}_pm"
                            for c in required_cols
                            if f"{c}_pm" in merged.columns
                        ]
                    ]
                    .isna()
                    .all(axis=1)
                )
                if still_missing.any():
                    logger.warning(
                        f"[UNEXPECTED] {still_missing.sum()} positions could not be mapped despite proper srcSharecodes from frontend"
                    )
                    # Log the problematic rows for debugging
                    missing_rows = df_out.loc[
                        still_missing,
                        [
                            "product_id",
                            "src_sharecodes",
                            "desk",
                            "port_type",
                            "currency",
                        ],
                    ]
                    logger.warning(f"Unmapped positions:\n{missing_rows}")
                    # Since frontend should send complete data, this indicates a data quality issue

                # Apply mapped values to main columns
                logger.debug(
                    f"[TEMP-DEBUG] Before applying mapped values - symbol NA count: {merged['symbol'].isna().sum() if 'symbol' in merged.columns else 'NO SYMBOL COL'}"
                )

                for c in required_cols:
                    src_col = c if c in merged.columns else f"{c}_pm"
                    if src_col in merged.columns:
                        before_na = (
                            merged[c].isna().sum() if c in merged.columns else "NEW_COL"
                        )
                        merged[c] = (
                            merged[c].fillna(merged[src_col])
                            if c in merged.columns
                            else merged[src_col]
                        )
                        after_na = merged[c].isna().sum()
                        if before_na != after_na:
                            logger.debug(
                                f"[TEMP-DEBUG] Column {c}: NA count changed from {before_na} to {after_na}"
                            )

                for c in [
                    "symbol",
                    "product_display_name",
                    "product_type_desc",
                    "asset_class_name",
                ]:
                    src_col = c if c in merged.columns else f"{c}_pm"
                    if src_col in merged.columns:
                        before_na = (
                            merged[c].isna().sum() if c in merged.columns else "NEW_COL"
                        )
                        merged[c] = (
                            merged[c].fillna(merged[src_col])
                            if c in merged.columns
                            else merged[src_col]
                        )
                        after_na = merged[c].isna().sum()
                        if str(before_na) != str(after_na):
                            logger.debug(
                                f"[TEMP-DEBUG] Column {c}: NA count changed from {before_na} to {after_na}"
                            )

                logger.debug(f"[TEMP-DEBUG] After applying mapped values:")
                logger.debug(
                    f"[TEMP-DEBUG] - symbol NA count: {merged['symbol'].isna().sum()}"
                )
                logger.debug(
                    f"[TEMP-DEBUG] - asset_class_name NA count: {merged['asset_class_name'].isna().sum()}"
                )
                logger.debug(
                    f"[TEMP-DEBUG] - symbol sample: {merged['symbol'].dropna().head(3).tolist()}"
                )

                df_out = merged[
                    df_out.columns.union(
                        [
                            "symbol",
                            "product_display_name",
                            "product_type_desc",
                            "asset_class_name",
                        ],
                        sort=False,
                    )
                ]

                logger.debug(f"[TEMP-DEBUG] Final df_out after enrichment:")
                logger.debug(f"[TEMP-DEBUG] - Total rows: {len(df_out)}")
                logger.debug(
                    f"[TEMP-DEBUG] - symbol NA count: {df_out['symbol'].isna().sum()}"
                )
                logger.debug(
                    f"[TEMP-DEBUG] - asset_class_name NA count: {df_out['asset_class_name'].isna().sum()}"
                )
                logger.debug(
                    f"[TEMP-DEBUG] - pp_asset_sub_class NA count: {df_out['pp_asset_sub_class'].isna().sum() if 'pp_asset_sub_class' in df_out.columns else 'NO COLUMN'}"
                )

                mask_missing_after = df_out[required_cols].isna().any(axis=1)
                if mask_missing_after.any():
                    sample = df_out.loc[
                        mask_missing_after, required_cols + ["src_sharecodes", "symbol"]
                    ].head(5)
                logger.debug("unmapped rows after enrichment=\n%s", sample)
                # Don't raise error, just log and continue with what we have
                logger.warning(
                    "Some portfolio positions could not be fully enriched with product_mapping"
                )
            else:
                logger.debug(
                    f"[TEMP-DEBUG] SUCCESS: All rows have required columns filled after enrichment"
                )
        logger.debug(
            "df_out enriched shape=%s cols=%s", df_out.shape, list(df_out.columns)
        )
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
    logger.debug("head of port = \n%s", port.df_out.head())
    # Instantiate and run rebalancer v2
    from app.utils.rebalancer import Rebalancer  # local import to avoid circular issues

    c = request.constraints
    # Try to infer optional fields from style
    as_of_date = None
    customer_id_val = None
    try:
        if isinstance(request.style, list) and len(request.style) > 0:
            tmp = pd.DataFrame(request.style)
            as_of_date = (
                str(tmp.get("AS_OF_DATE").iloc[0]) if "AS_OF_DATE" in tmp else None
            )
            customer_id_val = (
                pd.to_numeric(tmp.get("CUSTOMER_ID").iloc[0], errors="coerce")
                if "CUSTOMER_ID" in tmp
                else None
            )
        elif isinstance(request.style, dict):
            as_of_date = (
                str(request.style.get("AS_OF_DATE"))
                if request.style.get("AS_OF_DATE") is not None
                else None
            )
            customer_id_val = (
                pd.to_numeric(request.style.get("CUSTOMER_ID"), errors="coerce")
                if request.style.get("CUSTOMER_ID") is not None
                else None
            )
    except Exception:
        pass

    logger.debug(
        f"""
                Rebalancing for customer_id={request.customer_id} as_of_date={as_of_date} customer_id_val={customer_id_val}
                new_money={request.objective.new_money}
                discretionary_acceptance={c.discretionary_acceptance}
                client_classification={c.client_classification}
                private_percent={c.private_percent}
                cash_percent={c.cash_percent}
                offshore_percent={c.offshore_percent}
                product_restriction={c.product_restriction}
                product_whitelist={getattr(c, 'product_whitelist', None)}
                product_blacklist={getattr(c, 'product_blacklist', None)}
                """
    )

    # Map legacy product_restriction to blacklist if new fields not provided
    product_whitelist = getattr(c, "product_whitelist", None)
    product_blacklist = getattr(c, "product_blacklist", None)
    if (not product_blacklist) and c.product_restriction:
        product_blacklist = c.product_restriction

    rb_local = Rebalancer(
        customer_id=(
            int(customer_id_val)
            if customer_id_val is not None and not pd.isna(customer_id_val)
            else None
        ),
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
    try:
        logger.debug("ports.df_out head=\n%s", ports.df_out.head())
    except Exception:
        logger.debug("ports.df_out not available for preview")
    # [TEMP-DEBUG] Compare portfolio keys vs product_mapping for non-TRADE desks
    try:
        df_keys = ports.df_out[
            [
                "src_sharecodes",
                "product_id",
                "desk",
                "port_type",
                "currency",
                "asset_class_name",
            ]
        ].drop_duplicates()
        non_trade = df_keys[(df_keys["desk"].astype(str) != "TRADE")]
        if not non_trade.empty:
            logger.debug("non-TRADE portfolio keys (sample)=\n%s", non_trade.head(10))
            syms = non_trade["src_sharecodes"].dropna().astype(str).unique().tolist()
            pm_cols = [
                "src_sharecodes",
                "product_id",
                "desk",
                "port_type",
                "currency",
                "symbol",
                "product_type_desc",
                "asset_class_name",
            ]
            pm_slice = state.ports.product_mapping[
                state.ports.product_mapping["src_sharecodes"].astype(str).isin(syms)
            ][pm_cols].drop_duplicates()
            logger.debug(
                "product_mapping candidates for non-TRADE symbols (sample)=\n%s",
                pm_slice.head(20),
            )
            logger.debug("join keys used by rebalancer=%s", state.ports.prod_comp_keys)
    except Exception:
        logger.debug("TEMP-DEBUG non-TRADE key comparison failed")
    logger.debug("Rebalancing...")
    try:
        new_ports, actions_df = rb_local.rebalance(port, state.ppm, state.hs)
    except Exception as e:
        logger.exception("Rebalance failed: %s", e)
        raise
    logger.debug(
        "Rebalance completed [actions_df is None=%s | empty=%s]",
        actions_df is None,
        (False if actions_df is None else actions_df.empty),
    )

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

    action_logs: List[ActionLog] = []
    if actions_df is not None and not actions_df.empty:
        for _, r in actions_df.iterrows():
            # Handle NA values safely for ActionLog
            action_logs.append(
                ActionLog(
                    action=safe_str(r.get("flag"), "rebalance"),
                    step="",
                    trade_type=safe_str(r.get("action"), "trade"),
                    symbol=safe_str(r.get("src_sharecodes"), "UNKNOWN"),
                    amount_thb=safe_float(r.get("amount")),
                    unit=None,
                    price=None,
                    asset_class="Unknown",
                    notes=None,
                )
            )

    # Build proposed portfolio model
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
                    productId=safe_str(r.get("product_id")),
                    desk=safe_str(r.get("desk")),
                    portType=safe_str(r.get("port_type")),
                    currency=safe_str(r.get("currency")),
                    symbol=symbol_val,
                    srcSharecodes=safe_str(r.get("src_sharecodes"), ""),
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
    _metrics = compute_portfolio_health(
        proposed_portfolio_model,
        request.objective.target_alloc,
    )
    health = _metrics.score
    # Wrap detailed metrics into HealthMetrics to satisfy RebalanceResponse schema
    health_metrics = HealthMetrics(score=health, metrics=_metrics.metrics)

    return RebalanceResponse(
        actions=action_logs,
        portfolio=proposed_portfolio_model,
        health_score=health,
        health_metrics=health_metrics,
    )
