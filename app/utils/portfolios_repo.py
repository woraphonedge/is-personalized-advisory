import os

import pandas as pd

from .utils import get_latest_eom
from .data_loader import DataLoader



class PortfoliosRepository:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = DataLoader()

    # def get_portfolio_by_id(self, portfolio_id: int) -> pd.DataFrame:
    #     query = "SELECT * FROM portfolios WHERE portfolio_id = %s"
    #     df = self.loader.read_sql(query, params=(portfolio_id,))
    #     return df
    

    def load_client_out_product_enriched(
        self, 
        start_date: str = "", 
        end_date: str = "", 
        where_query: str = "", 
        value_column: str = "AUMX_THB"
    ) -> pd.DataFrame:
        if start_date == '' and end_date == '':
            start_date = get_latest_eom()
            end_date =  get_latest_eom()

        dim_column_select = [
            'customer_id',
            'as_of_date',
            'product_id',
            'src_sharecodes',
            'desk',
            'port_type',
            'currency',
            'product_display_name',
            'product_type_desc',
            'asset_class_name',
            'symbol',
            'pp_asset_sub_class',
            'is_risky_asset',
            'coverage_prdtype',
            'is_coverage',
            'expected_return'
        ]
        
        # prep query
        query = f"""
            SELECT {', '.join(dim_column_select)}, 
                sum({value_column}) as VALUE
            FROM user.kwm.client_health_score_outstanding_range('{start_date}','{end_date}')
            {where_query}
            GROUP BY {', '.join(dim_column_select)}
        """
        
        # prep cache file
        cache_file = f"client_out_{start_date}_{end_date}.parquet"

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
            "expected_return": "float64"
        }

        ports = self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

        # Fill missing values
        ports['coverage_prdtype'] = ports['coverage_prdtype'].fillna('N/A')

        return ports

    def load_client_style(
    # TODO: may be it need to create cal function in local
        self, 
        start_date: str, 
        end_date: str, 
        and_query: str = "", 
        style_column: str = "INVESTMENT_STYLE_AUMX"
    ) -> pd.DataFrame:
        
        if start_date == '' and end_date == '':
            start_date = get_latest_eom()
            end_date =  get_latest_eom()
        dim_column_select = ['customer_id', 'as_of_date']
        where_query = f"WHERE AS_OF_DATE between '{start_date}' and '{end_date}' and {and_query}"
        
        # prep query
        query = f"""
            SELECT  {', '.join(dim_column_select)}
                    , {style_column} as PORT_INVESTMENT_STYLE
            FROM user.kwm.client_investment_style_as_of
            {where_query}
        """

        # prep cache file
        cache_file = f"client_style_{start_date}_{end_date}.parquet"

        # prep type dict
        type_dict = {
            "as_of_date": "datetime64[ns]",
            "customer_id": "int",
            "port_investment_style": "string",
        }

        styles = self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

        return styles

    def load_product_mapping(self, as_of_date: str = '') -> pd.DataFrame:
        if as_of_date == '':
            as_of_date = get_latest_eom()

        dim_column_select = ['product_id', 
            'src_sharecodes',
            'desk',
            'port_type',
            'currency',
            'product_display_name',
            'product_type_desc',
            'asset_class_name',
            'symbol',
            'pp_asset_sub_class',
            'is_risky_asset',
            'coverage_prdtype',
            'is_coverage',
            'expected_return'
        ]
        
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
        cache_file = f"product_mapping_{as_of_date}.parquet"

        # prep type dict
        type_dict = {
                        'product_id': "string", 
                        'src_sharecodes': "string",
                        'desk': "string",
                        'port_type': "string",
                        "product_display_name": "string",
                        "currency": "string",
                        "product_type_desc": "string",
                        "asset_class_name": "string",
                        "symbol": "string",
                        "pp_asset_sub_class": "string",
                        "is_risky_asset": "bool",
                        "coverage_prdtype": "string",
                        "is_coverage": "bool",
                        "expected_return": "float64"
                    }
        product_mapping = self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)

        # Fill missing values
        product_mapping['coverage_prdtype'] = product_mapping['coverage_prdtype'].fillna('N/A')

        return product_mapping
    
    def load_product_underlying(self) -> pd.DataFrame:

        dim_column_select = ['product_id', 
            'underlying_company'
        ]
        
        # prep query
        query = f"""
            SELECT DISTINCT {', '.join(dim_column_select)}
            FROM edp.kkps_vw.v_npii_prod_undlycomp_map
            WHERE DATA_DT = (select max(DATA_DT) from edp.kkps_vw.v_npii_prod_undlycomp_map)
        """
       
        # prep cache file
        cache_file = f"underlying_mapping.parquet"

        # prep type dict
        type_dict = {
                        'product_id': "string", 
                        'underlying_company': "string",
                    }
        product_underlying = self.data_loader.load_data(type_dict, query=query, cache_file=cache_file)



        return product_underlying
    
if __name__ == "__main__":
    repo = PortfoliosRepository(data_loader=DataLoader())
    df = repo.load_product_mapping()
    print(df)