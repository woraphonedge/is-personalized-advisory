"""
Core rebalancing logic for the investment advisory model.  This module
implements portfolio health scoring and a heuristic rebalancing engine.  It is
designed to work with Pydantic models defined in `models.py` and supports
multi‑asset funds via the `exposures` attribute.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ..config import BULK_THRESHOLD, CORRIDOR_WIDTH, MIN_SHARPE_RATIO
from ..models import (
    HealthDetailMetrics,
    HealthMetrics,
    Portfolio,
    Position,
    RebalanceRequestMock,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def normalize_asset_label(label: str) -> str:
    """Map various legacy labels to the new four categories."""
    if label in {"Cash", "Cash and Cash Equivalent", "Cash Equivalent"}:
        return "Cash and Cash Equivalent"
    if label in {"Fixed Income", "Bond"}:
        return "Fixed Income"
    # Split Equity into Global and Local buckets; keep backward compatibility
    if label in {"Global Equity", "Equity", "Stocks"}:
        return "Global Equity"
    if label in {"Local Equity", "Thai Equity"}:
        return "Local Equity"
    # Keep Alternative as its own category
    if label in {"Alternative", "Alternatives"}:
        return "Alternative"
    if label in {"Allocation"}:
        return "Allocation"
    return label


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in weights.items():
        nk = normalize_asset_label(k)
        out[nk] = out.get(nk, 0.0) + v
    return out


def effective_exposure(p: Position, norm_asset: str) -> float:
    """Calculate the effective exposure of a position to a normalized asset class."""
    if p.exposures:
        total = 0.0
        for k, w in p.exposures.items():
            if normalize_asset_label(k) == norm_asset:
                total += w
        return total
    return 1.0 if normalize_asset_label(p.asset_class) == norm_asset else 0.0


def compute_portfolio_health(
    portfolio: Portfolio,
    target_alloc: Dict[str, float],
    min_sharpe_ratio: float = MIN_SHARPE_RATIO,
    bulk_threshold: float = BULK_THRESHOLD,
    corridor_width: float = CORRIDOR_WIDTH,
    risk_free_rate: float = 0.02,
) -> HealthMetrics:
    """Compute a simple health score on a scale of 0–10.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio to evaluate.
    target_alloc : List[AssetClassAllocation]
        Target weights for asset classes (must sum to 1).
    min_risk_adjusted_return : float
        Minimum acceptable ratio of expected return to volatility.
    bulk_threshold : float
        Percentage of total portfolio value beyond which a single position is
        penalised.
    corridor_width : float
        Allowed deviation from target allocation before penalty applies.

    Returns
    -------
    HealthMetrics
        Metrics object including 'score' (health score between 0 and 10) and components.
    """
    # Mock other metrics
    expected_return = np.random.uniform(0.0, 0.1)
    volatility = np.random.uniform(0.0, 0.2)
    max_drawdown = np.random.uniform(0.0, 0.1)
    var95 = np.random.uniform(0.0, 0.1)
    cumulative_return_20y = np.random.uniform(0.0, 0.1)
    annualized_return = np.random.uniform(0.0, 0.1)
    backtest_returns = []
    # asset_allocation in HealthDetailMetrics expects Dict[str, float] of current weights
    # We'll populate this later from actual portfolio weights; initialize empty here
    asset_allocation: Dict[str, float] = {}

    score = 10.0
    # logger.debug(f"portfolio: {portfolio}")
    total = portfolio.total_value()
    if total <= 0:
        # Empty portfolio: return zeros and empty allocation, respecting models
        detail = HealthDetailMetrics(
            expected_return=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            var95=0.0,
            cumulative_return_20y=0.0,
            annualized_return=0.0,
            backtest_returns=[],
            asset_allocation={},
        )
        return HealthMetrics(
            sharpe_ratio=0.0,
            mismatch_penalty=0.0,
            bulk_penalty=0.0,
            not_monitored_penalty=0.0,
            score=0.0,
            metrics=detail,
        )
    # Compute expected portfolio return and volatility as weighted averages
    exp_return = 0.0
    exp_vol = 0.0
    for pos in portfolio.positions:
        value = pos.market_value()
        exp_return += pos.expected_return * value
        exp_vol += pos.volatility * value
    exp_return /= total
    exp_vol /= total
    sharpe_ratio = (exp_return - risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
    if sharpe_ratio < min_sharpe_ratio:
        ratio = min(1.0, (min_sharpe_ratio - sharpe_ratio) / (min_sharpe_ratio or 1e-6))
        score -= 2.5 * ratio

    # Asset allocation mismatch penalty
    weights = normalize_weights(portfolio.asset_class_weights())
    mismatch_penalty = 0.0
    deviation_by_asset_class = {}

    # Normalize target allocation dict labels
    target_alloc_dict = {
        normalize_asset_label(k): v for k, v in (target_alloc or {}).items()
    }

    for asset, target in target_alloc_dict.items():
        actual = weights.get(asset, 0.0)
        deviation = abs(actual - target)
        deviation_by_asset_class[asset] = deviation
        if deviation > corridor_width:
            mismatch_penalty += (deviation - corridor_width) / (1.0 - corridor_width)
    score -= min(2.5, mismatch_penalty * 2.5)

    # Bulk risk penalty
    bulk_penalty = 0.0
    bulk_positions = []
    for pos in portfolio.positions:
        weight = pos.market_value() / total
        if weight > bulk_threshold:
            bulk_positions.append(pos)
            bulk_penalty += (weight - bulk_threshold) / (1.0 - bulk_threshold)
    score -= min(2.5, bulk_penalty * 2.5)

    # Monitoring penalty
    not_monitored_penalty = sum(0.5 for p in portfolio.positions if not p.is_monitored)
    score -= min(2.5, not_monitored_penalty)
    # Build flat asset allocation from current weights using new labels
    asset_allocation = {
        "Cash and Cash Equivalent": weights.get("Cash and Cash Equivalent", 0.0),
        "Fixed Income": weights.get("Fixed Income", 0.0),
        "Local Equity": weights.get("Local Equity", 0.0),
        "Global Equity": weights.get("Global Equity", 0.0),
        "Alternative": weights.get("Alternative", 0.0),
        "Allocation": weights.get("Allocation", 0.0),
    }
    # Clamp and build metrics
    final_score = max(0.0, min(10.0, score))
    detail = HealthDetailMetrics(
        expected_return=expected_return,
        volatility=volatility,
        max_drawdown=max_drawdown,
        var95=var95,
        cumulative_return_20y=cumulative_return_20y,
        annualized_return=annualized_return,
        backtest_returns=backtest_returns,
        asset_allocation=asset_allocation,
    )

    return HealthMetrics(
        sharpe_ratio=sharpe_ratio,
        mismatch_penalty=mismatch_penalty,
        bulk_penalty=bulk_penalty,
        not_monitored_penalty=not_monitored_penalty,
        score=final_score,
        metrics=detail,
    )


def propose_rebalance(
    request: RebalanceRequestMock,
    current_portfolio: Portfolio,
    candidate_universe: List[Position],
) -> Tuple[Portfolio, List[Dict[str, Any]]]:
    """Heuristic algorithm to generate a new portfolio according to a request.

    Parameters
    ----------
    request : RebalanceRequest
        The request containing objective, constraints and target allocation.
    current_portfolio : Portfolio
        The client's starting portfolio.
    candidate_universe : List[Position]
        Available securities to buy, each with attributes filled (unit_bal and
        exposures will be ignored on purchase).

    Returns
    -------
    (Portfolio, List[str])
        The rebalanced portfolio and a textual log of actions.
    """
    # Create working copy to mutate (ensure we operate on the same list the Portfolio holds)
    portfolio = Portfolio(
        positions=[Position(**p.model_dump()) for p in current_portfolio.positions]
    )

    # Helpers to manage cash via a dedicated Cash position (unit_price_thb = 1.0)
    def refresh_market_value(p: Position) -> None:
        """Recompute and assign explicit market value when units/prices change.
        We maintain mkt_val_thb explicitly because some upstream payloads may
        include it. Any local mutation of unit_bal/unit_price_thb must keep
        mkt_val_thb consistent to avoid stale values.
        """
        p.mkt_val_thb = (p.unit_bal or 0.0) * (p.unit_price_thb or 0.0)

    def find_cash_position() -> Position | None:
        return next((p for p in portfolio.positions if p.src_symbol == "CASH"), None)

    def ensure_cash_position() -> Position:
        cp = find_cash_position()
        if cp is None:
            cp = Position(
                src_symbol="CASH",
                asset_class="Cash and Cash Equivalent",
                unit_bal=0.0,
                unit_price_thb=1.0,
                unit_cost_thb=1.0,
                mkt_val_thb=0.0,
                expected_return=0.0,
                expected_income_yield=0.0,
                volatility=0.0,
                is_monitored=True,
            )
            portfolio.positions.append(cp)
        return cp

    def get_cash_value() -> float:
        cp = find_cash_position()
        return cp.market_value() if cp else 0.0

    def add_cash(amount: float) -> None:
        if amount <= 0:
            return
        cp = ensure_cash_position()
        # unit_price_thb is 1.0, so unit_bal increases by amount
        cp.unit_bal += amount / (cp.unit_price_thb or 1.0)
        refresh_market_value(cp)

    def spend_cash(amount: float) -> None:
        if amount <= 0:
            return
        cp = ensure_cash_position()
        cp.unit_bal -= amount / (cp.unit_price_thb or 1.0)
        refresh_market_value(cp)
        if cp.unit_bal <= 0:
            # Remove empty cash position to keep portfolio tidy
            try:
                portfolio.positions.remove(cp)
            except ValueError:
                pass

    # Add new money as cash position
    if request.objective.new_money > 0:
        add_cash(request.objective.new_money)
    # Structured actions log
    actions: List[Dict[str, Any]] = []

    def log_action(
        *,
        action: str,
        step: str,
        trade_type: str | None = None,
        symbol: str | None = None,
        amount_thb: float | None = None,
        unit: float | None = None,
        price: float | None = None,
        asset_class: str | None = None,
        notes: str | None = None,
    ) -> None:
        rec: Dict[str, Any] = {
            "action": action,
            "step": step,
        }
        if trade_type is not None:
            rec["trade_type"] = trade_type
        if symbol is not None:
            rec["symbol"] = symbol
        if amount_thb is not None:
            rec["amount_thb"] = amount_thb
        if unit is not None:
            rec["unit"] = unit
        if price is not None:
            rec["price"] = price
        if asset_class is not None:
            rec["asset_class"] = asset_class
        if notes is not None:
            rec["notes"] = notes
        actions.append(rec)

    # Log initial cash addition if any
    if request.objective.new_money > 0:
        log_action(
            action="add_cash",
            step="initialize",
            trade_type=None,
            symbol="CASH",
            amount_thb=request.objective.new_money,
            unit=request.objective.new_money,
            price=1.0,
            asset_class="Cash and Cash Equivalent",
        )

    do_not_sell = set(request.constraints.do_not_sell)
    total_value = portfolio.total_value()

    # Step 1: Sell obviously bad positions: bulk risk or unmonitored
    # Determine bulk risk threshold
    for pos in list(portfolio.positions):
        # Do not "sell" cash in bulk/unmonitored step; cash will be used to fund buys instead
        if normalize_asset_label(pos.asset_class) == "Cash and Cash Equivalent":
            continue
        weight = pos.market_value() / total_value if total_value > 0 else 0
        if (
            weight > BULK_THRESHOLD or not pos.is_monitored
        ) and pos.src_symbol not in do_not_sell:
            if pos.unrealised_gain_pct() < request.constraints.max_unrealised_loss_sell:
                continue
            max_sale_value = request.constraints.max_sale_fraction * total_value
            sale_value = min(pos.market_value(), max_sale_value)
            if sale_value <= 0:
                continue
            qty = sale_value / pos.unit_price_thb
            # Adjust unit_bal
            new_qty = pos.unit_bal - qty
            if new_qty <= 0:
                portfolio.positions.remove(pos)
            else:
                pos.unit_bal = new_qty
                refresh_market_value(pos)
            add_cash(sale_value)
            log_action(
                action="sell",
                step="sell_bad_or_unmonitored",
                trade_type="sell",
                symbol=pos.src_symbol,
                amount_thb=sale_value,
                unit=qty,
                price=pos.unit_price_thb,
                asset_class=normalize_asset_label(pos.asset_class),
            )
            total_value = portfolio.total_value()

    # Step 2: Reduce overweight asset classes
    weights = normalize_weights(portfolio.asset_class_weights())
    # Build dict from target allocations using normalized labels (support legacy 'Equity')
    target_alloc_dict = {
        normalize_asset_label(k): v
        for k, v in (request.objective.target_alloc or {}).items()
    }
    for asset, actual in list(weights.items()):
        target = target_alloc_dict.get(asset, 0.0)
        overweight = actual - max(target + CORRIDOR_WIDTH, target)
        if overweight <= 0:
            continue
        needed = overweight * total_value
        # Sort positions by lowest expected return
        positions = sorted(
            [p for p in portfolio.positions if effective_exposure(p, asset) > 0],
            key=lambda x: x.expected_return,
        )
        for pos in positions:
            if needed <= 0:
                break
            if pos.src_symbol in do_not_sell:
                continue
            if pos.unrealised_gain_pct() < request.constraints.max_unrealised_loss_sell:
                continue
            # Determine effective exposure weight of this position to the overweight asset
            eff_weight = effective_exposure(pos, asset)
            if eff_weight <= 0:
                continue
            # Skip selling cash when the overweight asset is Cash; buying other assets will reduce cash.
            if (
                normalize_asset_label(pos.asset_class) == "Cash and Cash Equivalent"
                and asset == "Cash and Cash Equivalent"
            ):
                continue
            # Compute gross value to sell so that cash added equals the true value sold.
            # If eff_weight < 1, selling quantity Q reduces the overweight by Q*price*eff_weight,
            # but adds cash Q*price. Therefore, cap by needed/eff_weight and by max_sale_fraction/eff_weight.
            pos_gross_value = pos.market_value()
            max_by_needed = needed / (eff_weight or 1.0)
            max_by_fraction = (
                request.constraints.max_sale_fraction
                * total_value
                / (eff_weight or 1.0)
            )
            sale_value_gross = min(pos_gross_value, max_by_needed, max_by_fraction)
            if sale_value_gross <= 0:
                continue
            qty_to_sell = sale_value_gross / (pos.unit_price_thb or 1.0)
            new_qty = pos.unit_bal - qty_to_sell
            if new_qty <= 0:
                portfolio.positions.remove(pos)
            else:
                pos.unit_bal = new_qty
                refresh_market_value(pos)
            add_cash(sale_value_gross)
            log_action(
                action="sell",
                step="reduce_overweight",
                trade_type="sell",
                symbol=pos.src_symbol,
                amount_thb=sale_value_gross,
                unit=qty_to_sell,
                price=pos.unit_price_thb,
                asset_class=asset,
                notes=f"effective_exposure={eff_weight:.4f}",
            )
            # Reduce remaining needed by the contribution of this sale to the overweight asset
            needed -= sale_value_gross * eff_weight
            total_value = portfolio.total_value()
        weights = portfolio.asset_class_weights()
    print(f"portfolio positions: {portfolio.positions}")
    # Step 3: Rebalance to target allocation
    # Use normalized labels across weights and target allocations
    current_weights = normalize_weights(portfolio.asset_class_weights())
    target_allocations = {
        normalize_asset_label(k): v
        for k, v in (request.objective.target_alloc or {}).items()
    }
    deficits = []

    for asset_class, target in target_allocations.items():
        current = current_weights.get(asset_class, 0.0)
        deficit = max(0.0, target - current)
        if deficit > 0:
            deficits.append((asset_class, deficit))
    deficits.sort(key=lambda x: x[1], reverse=True)
    # Build candidate lists by normalized asset class
    cand_map: Dict[str, List[Position]] = {}
    for cand in candidate_universe:
        cand_map.setdefault(normalize_asset_label(cand.asset_class), []).append(cand)
    # Sort candidates based on objective
    for asset, cands in cand_map.items():
        if request.objective.objective == "income":
            cands.sort(key=lambda c: c.expected_income_yield, reverse=True)
        elif request.objective.objective == "risk_adjusted":
            cands.sort(key=lambda c: c.expected_return / c.volatility, reverse=True)
        elif request.objective.objective == "principal":
            cands.sort(key=lambda c: c.expected_return, reverse=True)
        cand_map[asset] = cands

    for asset, deficit in deficits:
        if get_cash_value() <= 0:
            break
        invest_value = min(deficit * total_value, get_cash_value())
        if invest_value <= 0:
            continue
        cands = cand_map.get(asset, [])
        if not cands:
            continue
        cand = cands[0]
        qty = invest_value / cand.unit_price_thb
        if qty <= 0:
            continue
        # Add new position; exposures from candidate may propagate
        new_pos = Position(
            src_symbol=cand.src_symbol,
            asset_class=cand.asset_class,
            unit_bal=qty,
            unit_price_thb=cand.unit_price_thb,
            unit_cost_thb=cand.unit_price_thb,
            mkt_val_thb=cand.unit_price_thb * qty,
            expected_return=cand.expected_return,
            expected_income_yield=cand.expected_income_yield,
            volatility=cand.volatility,
            is_monitored=True,
            exposures=cand.exposures,
        )
        # If existing src_symbol exists, merge by updating unit_bal and weight averages
        existing = next(
            (p for p in portfolio.positions if p.src_symbol == new_pos.src_symbol), None
        )
        if existing:
            total_mv = existing.market_value() + new_pos.market_value()
            # Weighted averages
            existing.expected_return = (
                existing.expected_return * existing.market_value()
                + new_pos.expected_return * new_pos.market_value()
            ) / (total_mv or 1)
            existing.expected_income_yield = (
                existing.expected_income_yield * existing.market_value()
                + new_pos.expected_income_yield * new_pos.market_value()
            ) / (total_mv or 1)
            existing.volatility = (
                existing.volatility * existing.market_value()
                + new_pos.volatility * new_pos.market_value()
            ) / (total_mv or 1)
            if existing.unit_bal + new_pos.unit_bal > 0:
                existing.unit_cost_thb = (
                    existing.unit_cost_thb * existing.unit_bal
                    + new_pos.unit_cost_thb * new_pos.unit_bal
                ) / (existing.unit_bal + new_pos.unit_bal)
            existing.unit_bal += new_pos.unit_bal
            existing.is_monitored = existing.is_monitored and new_pos.is_monitored
            # Keep market value consistent with updated units/price
            refresh_market_value(existing)
            # exposures: if both have exposures, weight average; else use whichever exists
            if existing.exposures or new_pos.exposures:
                ex = existing.exposures or {existing.asset_class: 1.0}
                ne = new_pos.exposures or {new_pos.asset_class: 1.0}
                # convert both to dicts with full support
                combined: Dict[str, float] = {}
                for k, v in ex.items():
                    combined[k] = (
                        combined.get(k, 0.0) + v * existing.market_value() / total_mv
                    )
                for k, v in ne.items():
                    combined[k] = (
                        combined.get(k, 0.0) + v * new_pos.market_value() / total_mv
                    )
                existing.exposures = combined
        else:
            portfolio.positions.append(new_pos)
        spend_cash(qty * cand.unit_price_thb)
        log_action(
            action="buy",
            step="rebalance_to_target",
            trade_type="buy",
            symbol=cand.src_symbol,
            amount_thb=qty * cand.unit_price_thb,
            unit=qty,
            price=cand.unit_price_thb,
            asset_class=asset,
        )
        # remove candidate from list to avoid repeated purchase
        cand_map[asset].pop(0)
        # update values for next step
        total_value = portfolio.total_value()
        weights = portfolio.asset_class_weights()
    return portfolio, actions
