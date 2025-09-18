import os
import warnings

import pandas as pd

from .utils import get_latest_eom, read_parquet, read_sql, write_parquet

warnings.filterwarnings("ignore")


class Portfolios:
    def __init__(self):
        self.product_mapping = None
        self.df_out = None
        self.df_style = None
        self.port_ids = None
        self.port_id_mapping = None
        self.prod_comp_keys = ['PRODUCT_ID', 'SRC_SHARECODES', 'DESK', 'PORT_TYPE']

    def get_client_out_from_query(self, start_date, end_date, where_query="", value_column="AUMX_THB"):
        dim_column_select = [
            'CUSTOMER_ID',
            'AS_OF_DATE'
        ] + self.prod_comp_keys + [
            'PRODUCT_DISPLAY_NAME',
            'CURRENCY',
            'PRODUCT_TYPE_DESC',
            'ASSET_CLASS_NAME',
            'SYMBOL',
            'PP_ASSET_SUB_CLASS',
            'IS_RISKY_ASSET',
            'COVERAGE_PRDTYPE',
            'IS_COVERAGE',
            'EXPECTED_RETURN'
        ]
        query = f"""
            SELECT {', '.join(dim_column_select)},
                sum({value_column}) as VALUE
            FROM user.kwm.client_health_score_outstanding_range('{start_date}','{end_date}')
            {where_query}
            GROUP BY {', '.join(dim_column_select)}
        """
        ports = read_sql(query)
        type_dict = {
            "AS_OF_DATE": "datetime64[ns]",
            "CUSTOMER_ID": "int",
            "PRODUCT_DISPLAY_NAME": "string",
            "CURRENCY": "string",
            "PRODUCT_TYPE_DESC": "string",
            "ASSET_CLASS_NAME": "string",
            "SYMBOL": "string",
            "PP_ASSET_SUB_CLASS": "string",
            "VALUE": "float64",
            "IS_RISKY_ASSET": "bool",
            "COVERAGE_PRDTYPE": "string",
            "IS_COVERAGE": "bool",
            "EXPECTED_RETURN": "float64"
        }
        type_dict.update({key: 'string' for key in self.prod_comp_keys})
        ports = ports.astype(type_dict)
        ports['COVERAGE_PRDTYPE'] = ports['COVERAGE_PRDTYPE'].fillna('N/A')
        return ports

    def get_client_style_from_query(self, start_date, end_date, and_query="", style_column='INVESTMENT_STYLE_AUMX'):
        where_query = f"WHERE AS_OF_DATE between '{start_date}' and '{end_date}' {and_query}"
        query = f"""
            SELECT CUSTOMER_ID, AS_OF_DATE, {style_column} as PORT_INVESTMENT_STYLE
            FROM user.kwm.client_investment_style_as_of
            {where_query}
        """
        styles = read_sql(query)
        return styles.astype({
            "AS_OF_DATE": "datetime64[ns]",
            "CUSTOMER_ID": "int",
            "PORT_INVESTMENT_STYLE": "string",
        })

    def load_product_mapping(self,as_of_date=''):
        if as_of_date == '':
            as_of_date = get_latest_eom()

        dim_column_select = self.prod_comp_keys + [
            'PRODUCT_DISPLAY_NAME',
            'CURRENCY',
            'PRODUCT_TYPE_DESC',
            'ASSET_CLASS_NAME',
            'SYMBOL',
            'PP_ASSET_SUB_CLASS',
            'IS_RISKY_ASSET',
            'COVERAGE_PRDTYPE',
            'IS_COVERAGE',
            'EXPECTED_RETURN'
        ]
        query =  f"SELECT distinct {', '.join(dim_column_select)} FROM (select * from user.kwm.health_score_dim_product_info_range('{as_of_date}','{as_of_date}') union all select * from user.kwm.personalized_advisory_cash_proxy union all select * except (AA_CASH, AA_FI, AA_LE, AA_GE, AA_ALT) from user.kwm.personalized_advisory_asset_allocation_weight)"
        # Cache file is as-of-date specific
        cache_file = f"product_mapping_{as_of_date}.parquet"

        # Toggle: load from DWH and persist, or load from cache
        if str(os.getenv("LOAD_DATA_FROM_DWH", "")).strip().lower() in {"1", "true", "yes"}:
            product_mapping = read_sql(query)
        else:
            product_mapping = read_parquet(cache_file)

        type_dict = {
                        "PRODUCT_DISPLAY_NAME": "string",
                        "CURRENCY": "string",
                        "PRODUCT_TYPE_DESC": "string",
                        "ASSET_CLASS_NAME": "string",
                        "SYMBOL": "string",
                        "PP_ASSET_SUB_CLASS": "string",
                        "IS_RISKY_ASSET": "bool",
                        "COVERAGE_PRDTYPE": "string",
                        "IS_COVERAGE": "bool",
                        "EXPECTED_RETURN": "float64"
                    }
        type_dict.update({key: 'string' for key in self.prod_comp_keys})
        product_mapping = product_mapping.astype(type_dict)
        product_mapping['COVERAGE_PRDTYPE'] = product_mapping['COVERAGE_PRDTYPE'].fillna('N/A')
        # Persist if we fetched from DWH
        if str(os.getenv("LOAD_DATA_FROM_DWH", "")).strip().lower() in {"1", "true", "yes"}:
            try:
                write_parquet(product_mapping, cache_file)
            except Exception as e:
                print(f"[Portfolios] Warning: failed to persist product mapping {as_of_date}: {e}")
        self.product_mapping = product_mapping

    def map_client_out_prod_info(self, df_products):
        ## df_products must have columns PRODUCT_ID, PRODUCT_DISPLAY_NAME, PRODUCT_TYPE_DESC, VALUE

        if self.product_mapping is None:
            raise Exception("Product mapping is not loaded")

        df_out = df_products.merge(self.product_mapping, on=self.prod_comp_keys , how="left")
        lost_records = df_out[df_out['EXPECTED_RETURN'].isnull()]
        if not lost_records.empty:
            raise ValueError(f"Lost records found during merge:\n{lost_records}")

        return df_out

    def create_portfolio_id(self, df_out, df_style, column_mapping=None):
        if column_mapping is None:
            column_mapping = []
        if not column_mapping:
            df_out['PORT_ID'] = 1
            df_style['PORT_ID'] = 1
            port_ids = pd.Series([1], name='PORT_ID')
            return df_out, df_style, port_ids, None
        else:
            df_out['PORT_ID'] = df_out.groupby(column_mapping).ngroup() + 1
            port_id_mapping = df_out[column_mapping + ['PORT_ID']].drop_duplicates().reset_index(drop=True)
            df_style = port_id_mapping.merge(df_style, on=column_mapping, how='left').drop(column_mapping, axis=1)
            port_ids = port_id_mapping['PORT_ID'].sort_values().reset_index(drop=True)
            return df_out, df_style, port_ids, port_id_mapping

    def set_portfolio(self, df_out, df_style, port_ids, port_id_mapping):
        self.df_out = df_out
        self.df_style = df_style
        self.port_ids = port_ids
        self.port_id_mapping = port_id_mapping
        self.process_portfolio()

    def process_portfolio(self):
        asset_class_map = {
            'Alternative': 'AA_ALT',
            'Cash and Cash Equivalent': 'AA_CASH',
            'Fixed Income': 'AA_FI',
            'Global Equity': 'AA_GE',
            'Local Equity': 'AA_LE'
        }

        self.df_out['ASSET_CLASS_CODE'] = self.df_out['ASSET_CLASS_NAME'].map(asset_class_map)

        self.df_out['WEIGHT'] = self.df_out['VALUE'] / self.df_out.groupby(['PORT_ID'])['VALUE'].transform('sum')
        self.df_style['PORT_INVESTMENT_STYLE'] = self.df_style['PORT_INVESTMENT_STYLE'].fillna('Bulletproof')
        style_map = {
            'Bulletproof': 'Conservative',
            'Conservative': 'Conservative',
            'Moderate Low Risk': 'Medium to Moderate Low Risk',
            'Moderate High Risk': 'Medium to Moderate High Risk',
            'High Risk': 'High Risk',
            'Aggressive Growth': 'Aggressive',
            'Unwavering': 'Aggressive'
        }
        self.df_style['PORTPOP_STYLES'] = self.df_style['PORT_INVESTMENT_STYLE'].map(style_map)

    def get_portfolio_asset_allocation(self):
        df_astc_alloc = (
            self.df_out.groupby(['PORT_ID', 'ASSET_CLASS_CODE'])['WEIGHT'].sum()
            .unstack(fill_value=0)
            .reindex(columns=['AA_ALT', 'AA_CASH', 'AA_FI', 'AA_GE', 'AA_LE'], fill_value=0)
            .reset_index()
        )
        return df_astc_alloc

    def get_portfolio_asset_allocation_lookthrough(self, portpropmat):
        portpropmat.cal_df_out_join_bm(self)
        df_astc_alloc_lt = portpropmat.get_asset_class_allocation(self)
        return df_astc_alloc_lt

    def get_model_asset_allocation_lookthrough(self, portpropmat):
        return portpropmat.get_model_asset_class_allocation(self)

    def get_portfolio_buyable_cash(self, portpropmat):
        portpropmat.cal_df_out_join_bm(self)
        model_allocation = portpropmat.get_model_asset_class_allocation(self)

        product_cash = self.df_out[self.df_out['ASSET_CLASS_CODE'] == 'AA_CASH'].groupby(['PORT_ID'])['WEIGHT'].sum().reset_index()
        product_cash.merge(model_allocation, on=['PORT_ID'])
        product_cash['BUYABLE_CASH'] = (product_cash['WEIGHT'] - model_allocation['AA_CASH_MODEL']).clip(lower=0)

        return product_cash[['PORT_ID','BUYABLE_CASH']]

    def get_portfolio_health_score(self, portpropmat, healthscore, cal_comp=True):
        portpropmat.cal_df_out_join_bm(self)
        df_port_matrix, df_comp_risk_div = portpropmat.get_portprop_matrix(self)
        health_score, health_score_comp = healthscore.get_health_score(self, df_port_matrix, df_comp_risk_div, cal_comp)
        return  health_score, health_score_comp

if __name__ == "__main__":
    ports = Portfolios()
    # out_mocked = """
    # PRODUCT_ID	PRODUCT_DISPLAY_NAME	PRODUCT_TYPE_DESC	VALUE
    # S00086648	NTL	Listed Securities	1000
    # S00087551	BTSGIF	Listed Securities	1000
    # S00088553	PTTEP	Listed Securities	1000
    # S00088794	SCCC	Listed Securities	1000
    # S00089185	FTREIT	Listed Securities	1000
    # S00175880	SCGD	Listed Securities	1000
    # S00251431	QHBREIT	Listed Securities	1000
    # S00272592	SolarEdge Technologies Inc.	Listed Securities	1000
    # T00263261	ATCL20241224B	Structured Note	1000
    # T00265320	DNATCL20250106B	Structured Note	1000
    # """
    # # Create DataFrame
    # df_table = pd.read_csv(StringIO(out_mocked), sep='\t')

    # # df_out_mock = pd.DataFrame(df_table).astype({
    # #                         "PRODUCT_ID": "string",
    # #                         "PRODUCT_DISPLAY_NAME": "string",
    # #                         "PRODUCT_TYPE_DESC": "string",
    # #                         "VALUE": "float64"
    # #                     })

    # # df_style_mocked  = pd.DataFrame({'PORT_INVESTMENT_STYLE': ['Unwavering']})
    # # ports.load_product_mapping('2025-07-31')
    # # df_out_mocked = ports.map_client_out_prod_info(df_out_mock)
