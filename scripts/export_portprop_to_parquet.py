"""
Export PortProp reference tables from DWH to local Parquet files under data/.

Usage (from project root or personalized_advisory/):
  uv run python -m scripts.export_portprop_to_parquet

This uses utils.utils.read_sql which must be configured to access your DWH.
"""
from __future__ import annotations

from pathlib import Path

from app.utils.utils import read_sql

DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Queries matching PortpropMatrices.load_portprop_ref_tables
QUERIES = {
    "portprop_factsheet": "SELECT * FROM user.edg.portprop_factsheet WHERE symbol IS NOT NULL",
    "portprop_bm": "SELECT * FROM user.edg.portprop_bm WHERE pp_asset_sub_class IS NOT NULL",
    "portprop_ge_mapping": "SELECT * FROM user.edg.portprop_ge_mapping",
    "portprop_ret_eow": "SELECT * FROM user.edg.portprop_ret_eow",
    "advisory_health_score": "SELECT * FROM user.edg.advisory_health_score",
}

FILE_NAMES = {
    "portprop_factsheet": "portprop_factsheet.parquet",
    "portprop_bm": "portprop_bm.parquet",
    "portprop_ge_mapping": "portprop_ge_mapping.parquet",
    "portprop_ret_eow": "portprop_ret_eow.parquet",
    "advisory_health_score": "advisory_health_score.parquet",
}


def main() -> None:
    for key, sql in QUERIES.items():
        print(f"[export] Running query for {key} ...")
        df = read_sql(sql)
        out_path = DATA_DIR / FILE_NAMES[key]
        print(f"[export] Writing {len(df):,} rows to {out_path} ...")
        df.to_parquet(out_path, index=False)

    # Derive fallback slice for factsheet (not stored separately; computed on read)
    print("[export] Done.")


if __name__ == "__main__":
    main()
