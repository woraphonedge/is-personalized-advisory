import warnings

import pandas as pd

warnings.filterwarnings("ignore")


class Portfolios:
    def __init__(self):
        self.product_mapping = None
        self.product_underlying = None

        self.df_out = None
        self.df_style = None
        self.port_ids = None
        self.port_id_mapping = None
        self.prod_comp_keys = ['product_id', 'src_sharecodes', 'desk', 'port_type', 'currency']
        self.asset_class_map = {
            'Alternative': 'AA_ALT',
            'Cash and Cash Equivalent': 'AA_CASH',
            'Fixed Income': 'AA_FI',
            'Global Equity': 'AA_GE',
            'Local Equity': 'AA_LE'
        }
        self.style_map = {
            'Bulletproof': 'Conservative',
            'Conservative': 'Conservative',
            'Moderate Low Risk': 'Medium to Moderate Low Risk',
            'Moderate High Risk': 'Medium to Moderate High Risk',
            'High Risk': 'High Risk',
            'Aggressive Growth': 'Aggressive',
            'Unwavering': 'Aggressive'
        }

    def set_ref_tables(self, ref_dict: dict):

        # Set DataFrames with validation
        self.product_mapping = None if 'product_mapping' not in ref_dict else ref_dict['product_mapping']
        self.product_underlying =  None if 'product_underlying' not in ref_dict else ref_dict['product_underlying']

        # Validate required elements
        required_elements = {
            "product_mapping": self.product_mapping,
            "product_underlying": self.product_underlying,
        }
        for key, value in required_elements.items():
            if value is None:
                warnings.warn(f"'{key}' not provided. Expect errors if used.", stacklevel=2)


    def map_client_out_prod_info(self, df_products):

        if self.product_mapping is None:
            raise Exception("Product mapping is not loaded")

        df_out = df_products.merge(self.product_mapping, on=self.prod_comp_keys , how="left")
        lost_records = df_out[df_out['expected_return'].isnull()]
        if not lost_records.empty:
            raise ValueError(f"Lost records found during merge:\n{lost_records}")

        return df_out

    def create_portfolio_id(self, df_out, df_style, column_mapping=None):
        if column_mapping is None:
            column_mapping = []
        if not column_mapping:
            df_out['port_id'] = 1
            df_style['port_id'] = 1
            port_ids = pd.Series([1], name='port_id')
            return df_out, df_style, port_ids, None
        else:
            df_out['port_id'] = df_out.groupby(column_mapping).ngroup() + 1
            port_id_mapping = df_out[column_mapping + ['port_id']].drop_duplicates().reset_index(drop=True)
            df_style = port_id_mapping.merge(df_style, on=column_mapping, how='left').drop(column_mapping, axis=1)
            port_ids = port_id_mapping['port_id'].sort_values().reset_index(drop=True)
            return df_out, df_style, port_ids, port_id_mapping

    def set_portfolio(self, df_out, df_style, port_ids, port_id_mapping):
        self.df_out = df_out
        self.df_style = df_style
        self.port_ids = port_ids
        self.port_id_mapping = port_id_mapping
        self.process_portfolio()

    def process_portfolio(self):
        self.df_out['asset_class_code'] = self.df_out['asset_class_name'].map(self.asset_class_map)
        self.df_out['weight'] = self.df_out['value'] / self.df_out.groupby(['port_id'])['value'].transform('sum')

        self.df_style['port_investment_style'] = self.df_style['port_investment_style'].fillna('Bulletproof')
        self.df_style['portpop_styles'] = self.df_style['port_investment_style'].map(self.style_map)

    def get_portfolio_asset_allocation(self):
        df_astc_alloc = (
            self.df_out.groupby(['port_id', 'asset_class_code'])['weight'].sum()
            .unstack(fill_value=0)
            .reindex(columns=list(self.asset_class_map.values()), fill_value=0)
            .reset_index()
        )
        return df_astc_alloc

    def get_portfolio_asset_allocation_lookthrough(self, portpropmat):
        portpropmat.cal_df_out_join_bm(self)
        df_astc_alloc_lt,_ = portpropmat.get_asset_class_allocation(self)
        return df_astc_alloc_lt

    def get_model_asset_allocation_lookthrough(self, portpropmat):
        return portpropmat.get_model_asset_class_allocation(self)

    def get_portfolio_buyable_cash(self, portpropmat):
        portpropmat.cal_df_out_join_bm(self)
        model_allocation = portpropmat.get_model_asset_class_allocation(self)

        product_cash = self.df_out[self.df_out['asset_class_code'] == 'AA_CASH'].groupby(['port_id'])['weight'].sum().reset_index()
        product_cash.merge(model_allocation, on=['port_id'])
        product_cash['buyable_cash'] = (product_cash['weight'] - model_allocation['aa_cash_model']).clip(lower=0)

        return product_cash[['port_id','buyable_cash']]

    def get_portfolio_health_score(self, portpropmat, healthscore, cal_comp=True):
        portpropmat.cal_df_out_join_bm(self)
        df_port_matrix, df_comp_risk_div = portpropmat.get_portprop_matrix(self)
        health_score, health_score_comp = healthscore.get_health_score(self, df_port_matrix, df_comp_risk_div, cal_comp)
        return  health_score, health_score_comp

if __name__ == "__main__":
    ports = Portfolios()

