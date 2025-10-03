import pandas as pd

from .data_loader import DataLoader


class RebalancerRepository:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader


    def load_es_sell_list(self) -> pd.DataFrame:

        query = """
            SELECT symbol
            FROM user.kwm.personalized_advisory_es_sell_list
            WHERE upper(RECOMMENDATION) like '%SELL%'
            OR upper(RECOMMENDATION) like '%SWITCH%'
        """
        # prep cache file
        cache_file = "rebalancer_es_sell_list.parquet"

        # prep type dict
        type_dict = {"symbol": "string"}

        return self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)


    def load_product_recommendation_rank_raw(self) -> pd.DataFrame:
        dim_column_select = [
            "PRODUCT_TYPE_CODE",
            "PRODUCT_TYPE_DESC",
            "CLASS_CODE",
            "ASSET_CLASS_NAME",
            "DESK",
            "GEOGRAPHY",
            "CURRENCY",
            "IS_UI",
            "RANK_PRODUCT",
            "SYMBOL",
            "SRC_SHARECODES",
        ]
        cols = ', '.join(dim_column_select)
        query = f"""
            select {cols}
            from user.kwm.personalized_advisory_recommendation_rank
            """
        # prep cache file
        cache_file = "rebalancer_product_recommendation_rank_raw.parquet"

        # prep type dict
        type_dict = {
            "product_type_code": "string",
            "product_type_desc": "string",
            "class_code": "string",
            "asset_class_name": "string",
            "desk": "string",
            "geography": "string",
            "currency": "string",
            "is_ui": "string",
            "rank_product": "int64",
            "symbol": "string",
            "src_sharecodes": "string",
        }

        return self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

    def load_mandate_candidates(self) -> pd.DataFrame:
        ## Comment: Can we use from ports.product_mapping join with portprop fs instead?
        dim_column_select = [
            "pp_asset_sub_class_code", "SYMBOL", "CURRENCY", "AS_OF_DATE", "PRODUCT_ID", "PORT_TYPE", "DESK",
            "SRC_SYMBOL", "SRC_SHARECODES", "PRODUCT_DISPLAY_NAME", "COVERAGE_PRDTYPE", "IS_COVERAGE",
            "IS_RISKY_ASSET", "SRC_ISIN_CODE", "CLASS_DESC", "SUB_CLASS_DESC", "asset_class_code", "MODEL_SYMBOL",
            "MEMBEROF", "PRODUCT_TYPE_DESC", "GEOGRAPHY", "expected_return", "hedging_cost_flag", "asset_class_name",
            "pp_asset_sub_class", "AA_CASH", "AA_FI", "AA_LE", "AA_GE", "AA_ALT"
        ]

        # prep query
        cols = ', '.join(dim_column_select)
        query = f"""
            SELECT DISTINCT {cols}
            FROM user.kwm.personalized_advisory_asset_allocation_weight
        """

        # prep cache file
        cache_file = "rebalancer_mandate_candidates.parquet"

        # prep type dict
        type_dict = {
            "pp_asset_sub_class_code": "string",
            "symbol": "string",
            "currency": "string",
            "as_of_date": "datetime64[ns]",
            "product_id": "string",
            "port_type": "string",
            "desk": "string",
            "src_symbol": "string",
            "src_sharecodes": "string",
            "product_display_name": "string",
            "coverage_prdtype": "string",
            "is_coverage": "boolean",
            "is_risky_asset": "boolean",
            "src_isin_code": "string",
            "class_desc": "string",
            "sub_class_desc": "string",
            "asset_class_code": "string",
            "model_symbol": "string",
            "memberof": "string",
            "product_type_desc": "string",
            "geography": "string",
            "expected_return": "float64",
            "hedging_cost_flag": "boolean",
            "asset_class_name": "string",
            "pp_asset_sub_class": "string",
            "aa_cash": "float64",
            "aa_fi": "float64",
            "aa_le": "float64",
            "aa_ge": "float64",
            "aa_alt": "float64",
        }

        mandate_candidates = self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)
        for c in ["aa_cash","aa_fi","aa_le","aa_ge","aa_alt"]:
            mandate_candidates[c] = pd.to_numeric(mandate_candidates[c], errors="coerce").fillna(0.0)


        return mandate_candidates
