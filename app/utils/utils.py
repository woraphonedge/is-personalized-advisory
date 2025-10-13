import logging
import warnings

import pandas as pd
from dotenv import load_dotenv

from ..models import Portfolio

load_dotenv()
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def get_latest_eom():
    today = pd.to_datetime('today').normalize()
    eom = today.replace(day=1) + pd.offsets.MonthEnd(-1)
    return str(eom.date())


def convert_portfolio_to_df(portfolio: Portfolio) -> pd.DataFrame:
    # Convert Pydantic models to dicts with frontend aliases to ensure proper columns
    rows = [p.model_dump(by_alias=False) for p in portfolio.positions]
    
    df = pd.DataFrame(rows)
    
    # Map JSON field names to expected internal column names
    column_mapping = {
        "marketValue": "value",
        "mkt_val_thb": "value",  # Handle both possible names
        "assetClass": "asset_class_name", 
        "assetSubClass": "pp_asset_sub_class",
        "asset_class": "asset_class_name",  # Handle snake_case too
        "asset_sub_class": "pp_asset_sub_class",  # Handle snake_case too
        "srcSharecodes": "src_sharecodes",
        "productId": "product_id",
        "portType": "port_type",
    }
    df = df.rename(columns=column_mapping)
    
    # Ensure value column exists (critical for portfolio processing)
    if "value" not in df.columns:
        if "marketValue" in df.columns:
            df["value"] = df["marketValue"]
        elif "mkt_val_thb" in df.columns:
            df["value"] = df["mkt_val_thb"]
        else:
            logger.error(f"No value column found! Available columns: {list(df.columns)}")
            raise ValueError("Portfolio must have a value/marketValue/mkt_val_thb column")
    
    # Validate that src_sharecodes exists - now mandatory from frontend
    if "src_sharecodes" not in df.columns:
        logger.error("src_sharecodes field is missing from portfolio payload")
        raise ValueError("src_sharecodes field is required in portfolio positions")
    
    # Check for any missing src_sharecodes values
    missing_mask = df["src_sharecodes"].isna()
    if missing_mask.any():
        logger.error(f"Found {missing_mask.sum()} positions with missing src_sharecodes")
        raise ValueError("All portfolio positions must have src_sharecodes populated")
    
    # Ensure required columns exist for downstream processing
    # The rebalancer and other components expect these columns
    if "src_symbol" not in df.columns:
        df["src_symbol"] = df["symbol"] if "symbol" in df.columns else df["src_sharecodes"]
    
    if "symbol" not in df.columns:
        df["symbol"] = df["src_symbol"] if "src_symbol" in df.columns else df["src_sharecodes"]
    
    logger.debug(f"convert_portfolio_to_df columns: {list(df.columns)}")
    return df
if __name__ == "__main__":
    print(get_latest_eom())
