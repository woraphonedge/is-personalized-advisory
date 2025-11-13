# utils/data_loader.py
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

warnings.filterwarnings("ignore", message=".*_user_agent_entry.*")


class DataLoader:
    def __init__(self, load_from_db: bool = None):
        if load_from_db is None:
            self.load_from_db = str(
                os.getenv("LOAD_DATA_FROM_DWH", "")
            ).strip().lower() in {"1", "true", "yes"}
        else:
            self.load_from_db = load_from_db

    def get_databricks_engine(self):
        ACCESS_TOKEN = os.getenv("DATABRICKS_ACCESS_TOKEN")
        HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
        SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
        conn_str = (
            f"databricks://token:{ACCESS_TOKEN}@{SERVER_HOSTNAME}?http_path={HTTP_PATH}"
        )
        return create_engine(conn_str)

    def get_data_dir(self) -> Path:
        """Return the path to the project's data directory, creating it if needed."""
        # utils/ -> parent is personalized_advisory/, then 'data'
        base = Path(__file__).resolve().parents[1]
        data_dir = base / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    def read_sql(self, query: str, params=None) -> pd.DataFrame:
        engine = self.get_databricks_engine()
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)
        df.columns = [col.lower() for col in df.columns]
        return df.replace({pd.NA: np.nan})

    def write_parquet(
        self, df: pd.DataFrame, filename: str, schema: dict = None
    ) -> Path:
        """Write a DataFrame to Parquet under data/ and return the full path."""
        path = self.get_data_dir() / filename
        # Apply schema if provided to enforce correct data types
        if schema:
            df = df.astype(schema)
        # Use pyarrow engine to preserve boolean types correctly
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def read_parquet(self, filename: str) -> pd.DataFrame:
        """Read a Parquet file from data/ and return a DataFrame."""
        path = self.get_data_dir() / filename
        return pd.read_parquet(path)

    def load_data(self, type_dict: dict = None, **kwargs) -> pd.DataFrame:
        query = kwargs["query"]
        cache_file = kwargs["cache_file"]

        if self.load_from_db:
            try:
                df = self.read_sql(query)
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error: failed to execute query {query}: {e}"
                )

            try:
                self.write_parquet(df, cache_file)
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error: failed to persist {cache_file}: {e}"
                )
        else:
            try:
                df = self.read_parquet(cache_file)
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error: failed to read persist {cache_file}: {e}"
                )

        if type_dict:
            try:
                df = df.astype(type_dict)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Warning: casting dtypes: {e}")

        return df


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.read_sql("SELECT 1")
    print(loader.get_data_dir())
