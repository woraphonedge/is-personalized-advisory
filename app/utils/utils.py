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
    # df.columns = [col.upper() for col in df.columns]
    # logger.debug(f"columns: {df.columns}")
    column_mapping = {
        "mkt_val_thb": "value",
        "asset_class": "asset_class_name"
    }
    df = df.rename(columns=column_mapping)
    return df
if __name__ == "__main__":
    print(get_latest_eom())
