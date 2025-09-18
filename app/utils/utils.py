import logging
import os
import warnings
from pathlib import Path

import databricks.sql
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..models import Portfolio

load_dotenv()
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

ACCESS_TOKEN = os.getenv("DATABRICKS_ACCESS_TOKEN")
def read_sql(query):
    with databricks.sql.connect(
            server_hostname = "adb-1489929693291322.2.azuredatabricks.net",
            http_path = "sql/protocolv1/o/1489929693291322/0905-061922-m6dixqvc",
            access_token = ACCESS_TOKEN
    ) as connection:
        df = pd.read_sql(query, connection)
        df.columns = [col.upper() for col in df.columns]
        # df.applymap(lambda x: np.nan if type(x).__name__ == 'NAType' else x)
        return df.replace({pd.NA: np.nan})

def get_latest_eom():
    today = pd.to_datetime('today').normalize()
    eom = today.replace(day=1) + pd.offsets.MonthEnd(-1)
    return str(eom.date())


def get_data_dir() -> Path:
    """Return the path to the project's data directory, creating it if needed."""
    # utils/ -> parent is personalized_advisory/, then 'data'
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def write_parquet(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame to Parquet under data/ and return the full path."""
    path = get_data_dir() / filename
    df.to_parquet(path, index=False)
    return path


def read_parquet(filename: str) -> pd.DataFrame:
    """Read a Parquet file from data/ and return a DataFrame."""
    path = get_data_dir() / filename
    return pd.read_parquet(path)

def convert_portfolio_to_df(portfolio: Portfolio) -> pd.DataFrame:
    # Convert Pydantic models to dicts with frontend aliases to ensure proper columns
    rows = [p.model_dump(by_alias=True) for p in portfolio.positions]
    df = pd.DataFrame(rows)
    df.columns = [col.upper() for col in df.columns]
    logger.debug(f"columns: {df.columns}")
    column_mapping = {
        "POSDATE": "AS_OF_DATE",
        "PRODUCTID": "PRODUCT_ID",
        "SRC_SHARECODES": "SYMBOL",
        "ASSETCLASS": "ASSET_CLASS_NAME",
        "ASSETSUBCLASS": "PP_ASSET_SUB_CLASS",
        "MARKETVALUE": "VALUE",
    }
    df = df.rename(columns=column_mapping)
    return df
if __name__ == "__main__":
    print(read_sql('select cast("" as string) union select NULL').dtypes)
