import pandas as pd

from .data_loader import DataLoader


class PortpropMatricesRepository:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader


    # --- PortProp Factsheet ---
    def load_portprop_factsheet(self) -> pd.DataFrame:
        query = """
            SELECT  symbol, bm_name, asset_class, weight
            FROM user.edg.portprop_factsheet
            WHERE symbol IS NOT NULL
        """
        cache_file = "portprop_factsheet.parquet"
        type_dict = {
            "symbol": "string",
            "bm_name": "string",
            "asset_class": "string",
            "weight": "float64",
        }
        df = self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)
        return df

    def load_portprop_fallback(self) -> pd.DataFrame:
        df_fs = self.load_portprop_factsheet()
        return df_fs[df_fs["symbol"] == "FALLBACK"]

    # --- PortProp Benchmark ---
    def load_portprop_benchmark(self) -> pd.DataFrame:
        query = """
            SELECT volatility, expected_return_rate, asset_class, pp_asset_sub_class, bm_name
            FROM user.edg.portprop_bm
            WHERE pp_asset_sub_class IS NOT NULL
        """
        cache_file = "portprop_bm.parquet"
        type_dict = {
            "volatility": "float64",
            "expected_return_rate": "float64",
            "asset_class": "string",
            "pp_asset_sub_class": "string",
            "bm_name": "string",
        }
        return self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

    # --- Geography Mapping ---
    def load_portprop_ge_mapping(self) -> pd.DataFrame:
        query = """
            SELECT bm_name, EM, Europe, Japan, Other, US
            FROM user.edg.portprop_ge_mapping
        """
        cache_file = "portprop_ge_mapping.parquet"
        type_dict = {
            "bm_name": "string",
            "em": "float64",
            "europe": "float64",
            "japan": "float64",
            "other": "float64",
            "us": "float64",
        }
        return self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

    # --- Weekly Returns ---
    def load_portprop_ret_eow(self) -> pd.DataFrame:
        query = """
            SELECT return_date, bm_name, return
            FROM user.edg.portprop_ret_eow
        """
        cache_file = "portprop_ret_eow.parquet"
        type_dict = {
            "return_date": "datetime64[ns]",
            "bm_name": "string",
            "return": "float64",
        }
        return self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

    # --- Advisory Health Score ---
    def load_advisory_health_score(self) -> pd.DataFrame:
        query = """
            SELECT expected_return, expected_return_w_hedging_cost, volatility, var99,
                AA_cash, AA_fi, AA_le, AA_ge, AA_alt,
                CUR_usd, CUR_eur, CUR_jpy, CUR_thb, CUR_other,
                GE_US, GE_EUR, GE_JP, GE_EM, GE_Other,
                model_name
            FROM user.edg.advisory_health_score
        """
        cache_file = "portprop_advisory_model.parquet"
        type_dict = {
            "expected_return": "float64",
            "expected_return_w_hedging_cost": "float64",
            "volatility": "float64",
            "var99": "float64",
            "aa_cash": "float64",
            "aa_fi": "float64",
            "aa_le": "float64",
            "aa_ge": "float64",
            "aa_alt": "float64",
            "cur_usd": "float64",
            "cur_eur": "float64",
            "cur_jpy": "float64",
            "cur_thb": "float64",
            "cur_other": "float64",
            "ge_us": "float64",
            "ge_eur": "float64",
            "ge_jp": "float64",
            "ge_em": "float64",
            "ge_other": "float64",
            "model_name": "string",
        }
        bm = self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

        # transform % columns from 0-100 to 0-1
        float_cols = bm.select_dtypes(include=["float64"]).columns
        bm[float_cols] = bm[float_cols] / 100

        return bm
