"""
Centralized hard configuration for advisory thresholds.
These are server-side constants and are not user-configurable.
"""

# Minimum acceptable Sharpe ratio threshold used in health scoring
MIN_SHARPE_RATIO: float = 0.0

# Percentage of total portfolio value beyond which a single position is penalized
BULK_THRESHOLD: float = 0.20

# Allowed deviation from target allocation before penalty applies
CORRIDOR_WIDTH: float = 0.05

# Fallback exposure mapping used when a multi-asset/asset allocation fund lacks a
# provided breakdown. Values must sum to 1.0. Adjust as needed.
# This is intentionally simple and opinionated to provide a reasonable default.
DEFAULT_ASSET_ALLOCATION_EXPOSURE = {
    "Cash and Cash Equivalent": 0.05,
    "Fixed Income": 0.30,
    "Local Equity": 0.05,
    "Global Equity": 0.55,
    "Alternative": 0.05,
}
