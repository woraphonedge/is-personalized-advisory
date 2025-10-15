"""
Pydantic models representing portfolios and positions for the investment
rebalancing API.  These models use Pydantic v2 to perform type
validation and provide JSON serialisation/deserialisation.
"""

from __future__ import annotations  # postpones the evaluation of type annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.config import DEFAULT_ASSET_ALLOCATION_EXPOSURE


class Position(BaseModel):
    # Explicit camelCase aliases allow the API to accept frontend camelCase while
    # still using snake_case internally. With populate_by_name enabled, both
    # formats are accepted during validation.
    product_id: Optional[str] = Field(alias="productId")
    desk: Optional[str] = Field(default=None, alias="desk")
    port_type: Optional[str] = Field(default=None, alias="portType")
    src_symbol: str = Field(alias="symbol")
    src_sharecodes: Optional[str] = Field(default=None, alias="srcSharecodes")
    asset_class: str = Field(
        alias="assetClass",
        description="(Cash, Fixed Income, Global Equity, Local Equity, Alternative, Asset Allocation)",
    )
    asset_sub_class: Optional[str] = Field(alias="assetSubClass")
    unit_bal: float = Field(alias="unitBal")
    unit_price_thb: float = Field(alias="unitPriceThb")
    unit_cost_thb: float = Field(alias="unitCostThb")
    mkt_val_thb: Optional[float] = Field(
        default=None,
        alias="marketValue",
        description="Explicit market value in THB. When provided by certain APIs (e.g., health-score), prefer this over unit_bal * unit_price_thb without relying on units/prices.",
    )
    currency: Optional[str] = Field(default=None, alias="currency")

    expected_return: float = Field(
        alias="expectedReturn",
        description="Annualised expected return (decimal fraction, e.g. 0.08 for 8%)",
    )
    expected_income_yield: Optional[float] = Field(
        default=None,
        alias="expectedIncomeYield",
        description="Annual income yield (decimal fraction)",
    )
    volatility: float = Field(
        alias="volatility",
        description="Annualised standard deviation (decimal fraction)",
    )
    product_type_desc: Optional[str] = Field(default=None, alias="productTypeDesc")
    coverage_prdtype: Optional[str] = Field(default=None, alias="coveragePrdtype")
    is_monitored: bool = Field(
        alias="isMonitored", description="Whether the position is monitored"
    )
    is_risky_asset: Optional[bool] = Field(default=None, alias="isRiskyAsset")
    is_coverage: Optional[bool] = Field(default=None, alias="isCoverage")
    product_display_name: Optional[str] = Field(default=None, alias="productDisplayName")
    es_core_port: Optional[bool] = Field(default=None, alias="isCorePort")
    es_sell_list: Optional[str | None] = Field(default=None, alias="esSellList")
    flag_top_pick: Optional[str | None] = Field(default=None, alias="flagTopPick")
    flag_tax_saving: Optional[str | None] = Field(default=None, alias="flagTaxSaving")
    exposures: Optional[Dict[str, float]] = Field(
        default=None,
        alias="exposures",
        description="Exposures to asset classes",
    )
    pos_date: Optional[str] = Field(default=None, alias="posDate")

    # Pydantic v2 config: allow population by both field name and alias,
    # and preserve from_attributes and arbitrary_types_allowed behavior.
    model_config = ConfigDict(
        populate_by_name=True,
        validate_by_alias=True,
        extra="ignore",
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    def market_value(self) -> float:
        if self.mkt_val_thb is not None:
            return self.mkt_val_thb
        return self.unit_bal * self.unit_price_thb

    def unrealised_gain_pct(self) -> float:
        if self.unit_cost_thb == 0:
            return 0.0
        return (self.unit_price_thb - self.unit_cost_thb) / self.unit_cost_thb


class Portfolio(BaseModel):
    positions: List[Position] = Field(default_factory=list)

    class Config:
        from_attributes = True

    def total_value(self) -> float:
        return sum(p.market_value() for p in self.positions)

    def asset_class_weights(self) -> Dict[str, float]:
        """Compute asset class weights, taking into account exposures for
        multi‑asset funds (asset allocation).  The `exposures` field of a
        position is a mapping from asset classes to percentages summing to 1.
        """
        total = self.total_value()
        weights: Dict[str, float] = {}
        if total <= 0:
            return weights
        for pos in self.positions:
            value = pos.market_value()
            # Use explicit exposures when provided; otherwise, if this is an
            # Asset Allocation fund with no breakdown, use the fallback mapping.
            exposures = pos.exposures
            if (not exposures) and (pos.asset_class == "Asset Allocation"):
                exposures = DEFAULT_ASSET_ALLOCATION_EXPOSURE

            if exposures:
                for asset, w in exposures.items():
                    weights[asset] = weights.get(asset, 0.0) + (value * w) / total
            else:
                weights[pos.asset_class] = (
                    weights.get(pos.asset_class, 0.0) + value / total
                )
        return weights


class PortfolioResponse(BaseModel):
    """Response model for portfolio endpoint that includes portfolio data and client style."""
    portfolio: Portfolio = Field(description="Portfolio with positions")
    client_style: Optional[str] = Field(
        default=None,
        alias="clientStyle",
        description="Client investment style from df_style (e.g., 'High Risk', 'Conservative', 'Unwavering')"
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


class AssetClass(str, Enum):
    GLOBAL_EQUITY = "Global Equity"
    LOCAL_EQUITY = "Local Equity"
    FIXED_INCOME = "Fixed Income"
    ASSET_ALLOCATION = "Asset Allocation"
    CASH_EQUIVALENT = "Cash and Cash Equivalent"
    ALTERNATIVE = "Alternative"


class AssetClassAllocation(BaseModel):
    asset_class: AssetClass = Field(description="Asset class identifier")
    weight: float = Field(description="Target weight for this asset class (0.0 to 1.0)")


class Objective(BaseModel):
    objective: str = Field(
        description="Objective of the client: income, risk_adjusted or principal"
    )
    new_money: float = Field(0.0, description="New money to add to the portfolio")
    target_alloc: Dict[str, float] = Field(
        description="Target allocation mapping from asset class to target weight (0.0-1.0)"
    )
    client_style: Optional[str] = Field(
        default=None,
        description="Client investment style label (e.g., High Risk, Conservative)"
    )


class Constraints(BaseModel):
    do_not_sell: List[str] = Field(
        default_factory=list, description="List of src_symbols to not sell"
    )
    max_unrealised_loss_sell: float = Field(
        -0.30, description="Maximum unrealised loss to sell"
    )
    max_sale_fraction: float = Field(
        0.10, description="Maximum fraction of portfolio value to sell"
    )


class RebalanceRequestMock(BaseModel):
    customer_id: int
    objective: Objective
    constraints: Constraints
    portfolio: Optional[Portfolio] = None
    # Optional style payload for v2 API; can be a dict or list of dicts convertible to a DataFrame
    style: Optional[List[Dict[str, Any]] | Dict[str, Any]] = None


class ConstraintsV2(BaseModel):
    """Constraints/settings used by the v2 rebalancer.

    These map to the parameters of utils.rebalancer_v2.Rebalancer.__init__.
    """
    private_percent: float = Field(
        0.0, description="Portfolio portion eligible for private products (0-1)"
    )
    cash_percent: Optional[float] = Field(
        default=None, description="Explicit target cash percent (0-1), None to use model"
    )
    offshore_percent: Optional[float] = Field(
        default=None, description="Target offshore exposure percent (0-1)"
    )
    product_restriction: Optional[List[str]] = Field(
        default=None, description="List of product symbols to exclude"
    )
    product_whitelist: Optional[List[str]] = Field(
        default=None, description="List of product symbols explicitly allowed"
    )
    product_blacklist: Optional[List[str]] = Field(
        default=None, description="List of product symbols to exclude (takes precedence over restriction)"
    )
    discretionary_acceptance: float = Field(
        default=0.4,
        description="Cap for total Mandate weight in portfolio (0-1). None -> no cap (treated as 1.0).",
    )
    client_classification: Optional[str] = Field(
        default=None,
        description="Client classification (e.g., UI, Non-UI)",
    )


class RebalanceRequest(BaseModel):
    customer_id: int
    objective: Objective
    constraints: ConstraintsV2
    portfolio: Optional[Portfolio] = None
    style: Optional[List[Dict[str, Any]] | Dict[str, Any]] = None


class HealthMetricsRequest(BaseModel):
    customer_id: int
    target_alloc: Dict[str, float]
    portfolio: Optional[Portfolio] = None
    client_style: Optional[str] = Field(
        default=None,
        description="Client investment style (e.g., 'High Risk', 'Conservative', 'Aggressive Growth'). "
                    "NOTE: Currently not used in health score calculation. The system uses the stored "
                    "style from df_style (port_investment_style column) which is loaded at startup. "
                    "Future implementation should override the stored style with this value when provided."
    )


class ActionLog(BaseModel):
    action: str
    step: str
    trade_type: Optional[str] = None
    symbol: Optional[str] = None
    amount_thb: Optional[float] = None
    unit: Optional[float] = None
    price: Optional[float] = None
    asset_class: Optional[str] = None
    notes: Optional[str] = None


class RebalanceResponse(BaseModel):
    actions: List[ActionLog]
    portfolio: Portfolio
    health_score: float
    health_metrics: HealthMetrics


class HealthDetailMetrics(BaseModel):
    """Detailed components matching the notebook's health score dataframe.

    Columns reference from notebook output:
    - expected_return, expected_return_model, score_ret
    - volatility, volatility_model, score_vol
    - score_portfolio_risk
    - acd (asset class allocation distance), score_acd
    - ged (global equity allocation distance), score_ged
    - score_diversification
    - score_bulk_risk
    - score_issuer_risk
    - score_non_cover_global_stock, score_non_cover_local_stock, score_non_cover_mutual_fund
    - score_not_monitored_product
    """

    model_config = ConfigDict(
        populate_by_name=True,
    )

    port_id: int | None = None

    expected_return: float
    expected_return_model: float | None = None
    score_ret: int | None = None

    volatility: float
    volatility_model: float | None = None
    score_vol: int | None = None

    score_portfolio_risk: int | None = None

    acd: float | None = None
    score_acd: int | None = None

    ged: float | None = None
    score_ged: int | None = None

    score_diversification: int | None = None
    score_bulk_risk: int | None = None
    score_issuer_risk: int | None = None

    score_non_cover_global_stock: int | None = None
    score_non_cover_local_stock: int | None = None
    score_non_cover_mutual_fund: int | None = None
    score_not_monitored_product: float | None = None

    # Optional: include look-through allocation for convenience
    asset_allocation: Dict[str, float] | None = None
    # Model asset allocation based on client investment style (advisory model target)
    model_asset_allocation: Dict[str, float] | None = Field(
        default=None,
        alias="modelAssetAllocation",
        description="Target asset allocation from advisory model matching client investment style"
    )


class HealthMetrics(BaseModel):
    """Health metrics aligned with notebook output.

    We keep a top-level 'score' for compatibility with frontend usage, mapping to
    the notebook's 'health_score'. All component details are in 'metrics'.
    """

    score: float = Field(description="Overall health score (mirrors notebook health_score)")
    metrics: HealthDetailMetrics
