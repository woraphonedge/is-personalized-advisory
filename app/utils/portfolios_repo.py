import pandas as pd

from .data_loader import DataLoader
from .utils import get_latest_eom


class PortfoliosRepository:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def load_client_out_product_enriched(
        self,
        as_of_date: str = "",
        filter_query: str = "",
        value_column: str = "AUMX_THB",
    ) -> pd.DataFrame:

        dim_column_select = [
            "customer_id",
            "as_of_date",
            "product_id",
            "src_sharecodes",
            "desk",
            "port_type",
            "currency",
            "product_display_name",
            "product_type_desc",
            "asset_class_name",
            "symbol",
            "pp_asset_sub_class",
            "is_risky_asset",
            "coverage_prdtype",
            "is_coverage",
            "expected_return",
            "es_core_port",
            "es_sell_list",
            "flag_top_pick",
            "flag_tax_saving",
        ]
        if filter_query != "":
            filter_query = "where " + filter_query
        # prep query
        query = f"""
            SELECT {', '.join(dim_column_select)},
                sum({value_column}) as VALUE
            FROM user.kwm.client_health_score_outstanding_range('{as_of_date}','{as_of_date}')
            {filter_query}
            GROUP BY {', '.join(dim_column_select)}
        """

        # prep cache file
        cache_file = f"portfolios_client_out_enriched_{as_of_date}.parquet"

        # prep type dict
        type_dict = {
            "as_of_date": "datetime64[ns]",
            "customer_id": "int",
            "product_display_name": "string",
            "currency": "string",
            "product_type_desc": "string",
            "asset_class_name": "string",
            "symbol": "string",
            "pp_asset_sub_class": "string",
            "value": "float64",
            "is_risky_asset": "bool",
            "coverage_prdtype": "string",
            "is_coverage": "bool",
            "expected_return": "float64",
            "es_core_port": "bool",
            "es_sell_list": "string",
            "flag_top_pick": "string",
            "flag_tax_saving": "string",
        }

        ports = self.data_loader.load_data(
            type_dict, query=query, cache_file=cache_file
        )

        # Fill missing values
        ports["coverage_prdtype"] = ports["coverage_prdtype"].fillna("N/A")

        return ports

    def load_client_style(
        self,
        as_of_date: str,
        and_query: str = "",
        style_column: str = "INVESTMENT_STYLE_AUMX",
    ) -> pd.DataFrame:
        if and_query != "":
            and_query = "and " + and_query

        # Columns to select
        dim_column_select = [
            "A.AS_OF_DATE",
            "A.customer_id",
            # "CONCAT(SUBSTR(B.CLIENT_FULL_NAME_TH, 1, 1), REPEAT('*', LENGTH(B.CLIENT_FULL_NAME_TH) - 1)) as CLIENT_FULL_NAME_TH",
            # "CONCAT(SUBSTR(B.CLIENT_FIRST_NAME_EN, 1, 1), REPEAT('*', LENGTH(B.CLIENT_FIRST_NAME_EN) - 1)) as CLIENT_FIRST_NAME_EN",
            # "CONCAT(SUBSTR(B.CLIENT_LAST_NAME_EN, 1, 1), REPEAT('*', LENGTH(B.CLIENT_LAST_NAME_EN) - 1)) as CLIENT_LAST_NAME_EN",
            "B.CLIENT_FULL_NAME_TH",
            "B.CLIENT_FIRST_NAME_EN",
            "B.CLIENT_LAST_NAME_EN",
            f"A.{style_column} as port_investment_style",
            "B.CLIENT_TIER",
            "B.BUSINESS_UNIT",
            "B.CLIENT_SEGMENT_BY_INV_AUM",
            "B.CLIENT_SUB_SEGMENT_BY_INV_AUM",
            "B.SALES_ID",
            "B.UI_Client",
            # "CONCAT(SUBSTR(C.SALES_FIRST_NAME_EN, 1, 1), REPEAT('*', LENGTH(C.SALES_FIRST_NAME_EN) - 1)) as SALES_FIRST_NAME_EN",
            # "CONCAT(SUBSTR(C.SALES_TEAM, 1, 1), REPEAT('*', LENGTH(C.SALES_TEAM) - 1)) as SALES_TEAM",
            "C.SALES_FIRST_NAME_EN",
            "C.SALES_TEAM",
        ]

        where_query = (
            f"WHERE A.AS_OF_DATE = '{as_of_date}' {and_query}"
        )

        query = f"""
            SELECT {', '.join(dim_column_select)}
            FROM user.kwm.client_investment_style_as_of A
            LEFT JOIN edp.kkps_vw.v_pii_client_info B
                ON A.customer_id = B.customer_id
                AND B.DATA_DT = (SELECT max(DATA_DT) FROM edp.kkps_vw.v_pii_client_info)
            LEFT JOIN edp.kkps_vw.v_pii_sales_info C
                ON B.SALES_ID = C.SALES_ID
                AND C.DATA_DT = (SELECT max(DATA_DT) FROM edp.kkps_vw.v_pii_sales_info)
            {where_query}
        """

        cache_file = f"portfolios_client_style_{as_of_date}.parquet"

        type_dict = {
            "as_of_date": "datetime64[ns]",
            "customer_id": "int",
            "client_full_name_th": "string",
            "client_first_name_en": "string",
            "client_last_name_en": "string",
            "port_investment_style": "string",
            "client_tier": "string",
            "business_unit": "string",
            "client_segment_by_inv_aum": "string",
            "client_sub_segment_by_inv_aum": "string",
            "sales_id": "string",
            "ui_client": "string",
            "sales_first_name_en": "string",
            "sales_team": "string",
        }

        styles = self.data_loader.load_data(
            type_dict, query=query, cache_file=cache_file
        )

        return styles

    def load_product_mapping(self, as_of_date: str = "") -> pd.DataFrame:

        if as_of_date == "":
            as_of_date = get_latest_eom()

        dim_column_select = [
            "product_id",
            "src_sharecodes",
            "desk",
            "port_type",
            "currency",
            "product_display_name",
            "product_type_desc",
            "asset_class_name",
            "symbol",
            "pp_asset_sub_class",
            "is_risky_asset",
            "coverage_prdtype",
            "is_coverage",
            "expected_return",
            "es_core_port",
            "es_sell_list",
            "flag_top_pick",
            "flag_tax_saving",
        ]
        ## TODO: Will nomalize this (maintain single source of truth in DWH)
        # prep query
        query = f"""
            SELECT DISTINCT {', '.join(dim_column_select)}
            FROM (
            SELECT *
            FROM user.kwm.health_score_dim_product_info_range('{as_of_date}', '{as_of_date}')
            UNION ALL
            SELECT *
            FROM user.kwm.personalized_advisory_cash_proxy
            UNION ALL
            SELECT * EXCEPT (AA_CASH, AA_FI, AA_LE, AA_GE, AA_ALT)
            FROM user.kwm.personalized_advisory_asset_allocation_weight
            )
        """

        # prep cache file
        cache_file = f"portfolios_product_mapping_{as_of_date}.parquet"

        # prep type dict
        type_dict = {
            "product_id": "string",
            "src_sharecodes": "string",
            "desk": "string",
            "port_type": "string",
            "product_display_name": "string",
            "currency": "string",
            "product_type_desc": "string",
            "asset_class_name": "string",
            "symbol": "string",
            "pp_asset_sub_class": "string",
            "is_risky_asset": "bool",
            "coverage_prdtype": "string",
            "is_coverage": "bool",
            "expected_return": "float64",
            "es_core_port": "bool",
            "es_sell_list": "string",
            "flag_top_pick": "string",
            "flag_tax_saving": "string",
        }
        product_mapping = self.data_loader.load_data(
            type_dict, query=query, cache_file=cache_file
        )

        # Fill missing values
        product_mapping["coverage_prdtype"] = product_mapping[
            "coverage_prdtype"
        ].fillna("N/A")

        return product_mapping

    def load_product_underlying(self) -> pd.DataFrame:

        dim_column_select = ["product_id", "underlying_company"]

        # prep query
        query = f"""
            SELECT DISTINCT {', '.join(dim_column_select)}
            FROM edp.kkps_vw.v_npii_prod_undlycomp_map
            WHERE DATA_DT = (select max(DATA_DT) from edp.kkps_vw.v_npii_prod_undlycomp_map)
        """

        # prep cache file
        cache_file = "portfolios_underlying_mapping.parquet"

        # prep type dict
        type_dict = {
            "product_id": "string",
            "underlying_company": "string",
        }
        product_underlying = self.data_loader.load_data(
            type_dict, query=query, cache_file=cache_file
        )

        return product_underlying

    def load_acct_customer_mapping(self) -> pd.DataFrame:

        dim_column_select = ["customer_id", "sub_account_no"]

        # prep query
        query = f"""
            SELECT DISTINCT {', '.join(dim_column_select)}
            FROM edp.kkps_vw.v_pii_acct_subacct_mast
            WHERE DATA_DT = (select max(DATA_DT) from edp.kkps_vw.v_pii_acct_subacct_mast)
        """

        # prep cache file
        cache_file = "acct_customer_mapping.parquet"

        # prep type dict
        type_dict = {
            "customer_id": "string",
            "sub_account_no": "string",
        }
        acct_customer_mapping = self.data_loader.load_data(
            type_dict, query=query, cache_file=cache_file
        )

        return acct_customer_mapping



if __name__ == "__main__":
    repo = PortfoliosRepository(data_loader=DataLoader())
    df = repo.load_product_mapping()
    print(df)
