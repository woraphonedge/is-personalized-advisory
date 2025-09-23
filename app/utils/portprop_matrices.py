import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import read_sql

warnings.filterwarnings("ignore")


class PortpropMatrices():
    def __init__(self):
        load_from_dwh = str(os.getenv("LOAD_DATA_FROM_DWH", "")).strip().lower() in {"1", "true", "yes"}
        if load_from_dwh:
            self.load_portprop_ref_tables()
        else:
            self.load_portprop_from_parquet()
        self.cal_ret_cov()
        self.asset_class_list = ['AA_ALT', 'AA_CASH', 'AA_FI', 'AA_GE', 'AA_LE']
        self.equity_geography_list = ['EM', 'EUROPE', 'JAPAN', 'US', 'OTHER']
        self.df_out_join = None

    def load_portprop_ref_tables(self):
        df_port_fs = read_sql("SELECT * FROM user.edg.portprop_factsheet WHERE symbol IS NOT NULL")
        df_port_fb = df_port_fs[df_port_fs['SYMBOL'] == 'FALLBACK']

        df_port_bm = read_sql(
            "SELECT * FROM user.edg.portprop_bm WHERE pp_asset_sub_class IS NOT NULL"
        )

        df_ge_mapping = read_sql(
            "SELECT * FROM user.edg.portprop_ge_mapping"
        )

        df_port_ret = read_sql(
            "SELECT * FROM user.edg.portprop_ret_eow"
        )

        df_model = read_sql(
            "SELECT * FROM user.edg.advisory_health_score"
        )

        self.df_port_fs = df_port_fs
        self.df_port_fb = df_port_fb
        self.df_port_bm = df_port_bm
        self.df_ge_mapping = df_ge_mapping
        self.df_port_ret = df_port_ret
        self.df_model = df_model

    def load_portprop_from_parquet(self):
        """Load PortProp reference tables from local Parquet files under data/."""
        data_dir = (Path(__file__).resolve().parents[1] / "data").resolve()
        factsheet_pq = data_dir / "portprop_factsheet.parquet"
        bm_pq = data_dir / "portprop_bm.parquet"
        ge_map_pq = data_dir / "portprop_ge_mapping.parquet"
        ret_eow_pq = data_dir / "portprop_ret_eow.parquet"
        model_pq = data_dir / "advisory_health_score.parquet"

        df_port_fs = pd.read_parquet(factsheet_pq)
        df_port_fb = df_port_fs[df_port_fs['SYMBOL'] == 'FALLBACK']
        df_port_bm = pd.read_parquet(bm_pq)
        df_ge_mapping = pd.read_parquet(ge_map_pq)
        df_port_ret = pd.read_parquet(ret_eow_pq)
        df_model = pd.read_parquet(model_pq)

        self.df_port_fs = df_port_fs
        self.df_port_fb = df_port_fb
        self.df_port_bm = df_port_bm
        self.df_ge_mapping = df_ge_mapping
        self.df_port_ret = df_port_ret
        self.df_model = df_model

    def cal_ret_cov(self):
        df_ret_pivot = self.df_port_ret.pivot_table(index='RETURN_DATE'
                                               , columns='BM_NAME'
                                               , values='RETURN'
                                               , aggfunc='sum')
        self.df_ret_cov = df_ret_pivot.cov()

    def cal_df_out_join_bm(self, ports):
        df_out_join = ports.df_out.copy()

        df_out_join['ASSET_ALLO_JOIN_KEY'] = np.where(
            df_out_join['ASSET_CLASS_NAME'] == 'Allocation',
            df_out_join['SYMBOL'],
            np.nan
        )

        df_out_join = df_out_join.merge(
            self.df_port_fs,
            how='left',
            left_on=['ASSET_ALLO_JOIN_KEY'],
            right_on=['SYMBOL'],
            suffixes=('', '_FS')
        )

        df_out_join['ASSET_ALLO_FALLBACK_JOIN_KEY'] = np.where(
            (df_out_join['ASSET_ALLO_JOIN_KEY'].notna()) & (df_out_join['SYMBOL_FS'].isna()),
            'FALLBACK',
            np.nan
        )

        df_out_join = df_out_join.merge(
            self.df_port_fs,
            how='left',
            left_on=['ASSET_ALLO_FALLBACK_JOIN_KEY'],
            right_on=['SYMBOL'],
            suffixes=('', '_FB')
        )

        df_out_join = df_out_join.merge(
            self.df_port_bm,
            how='left',
            left_on=['PP_ASSET_SUB_CLASS'],
            right_on=['PP_ASSET_SUB_CLASS'],
            suffixes=('', '_BM')
        )

        # Vectorized assignments for 'ASSET_CLASS', 'BM_NAME', 'WEIGHT'
        is_alloc = df_out_join['ASSET_CLASS_NAME'] == 'Allocation'
        fallback = df_out_join['ASSET_ALLO_FALLBACK_JOIN_KEY'] == 'FALLBACK'

        df_out_join['ASSET_CLASS'] = np.where(
            ~is_alloc,
            df_out_join['ASSET_CLASS_NAME'],
            np.where(
                ~fallback,
                df_out_join['ASSET_CLASS'],
                df_out_join['ASSET_CLASS_FB']
            )
        )
        df_out_join['BM_NAME'] = np.where(
            ~is_alloc,
            df_out_join['BM_NAME_BM'],
            np.where(
                ~fallback,
                df_out_join['BM_NAME'],
                df_out_join['BM_NAME_FB']
            )
        )
        df_out_join['WEIGHT'] = np.where(
            ~is_alloc,
            df_out_join['WEIGHT'],
            np.where(
                ~fallback,
                df_out_join['WEIGHT_FS'] * df_out_join['WEIGHT'],
                df_out_join['WEIGHT_FB'] * df_out_join['WEIGHT']
            )
        )

        df_out_join = df_out_join[list(ports.df_out.columns) + ['ASSET_CLASS','BM_NAME']]

        df_out_join = df_out_join.merge(
            self.df_ge_mapping,
            how='left',
            left_on=['BM_NAME'],
            right_on=['BM_NAME'],
            suffixes=('','_GE'))

        self.df_out_join = df_out_join

    def get_asset_class_allocation(self, ports):
        asset_class_map = {
            'Alternative': 'AA_ALT',
            'Cash and Cash Equivalent': 'AA_CASH',
            'Fixed Income': 'AA_FI',
            'Global Equity': 'AA_GE',
            'Local Equity': 'AA_LE'
        }
        all_classes = list(asset_class_map.keys())

        df = self.df_out_join
        df_astc_alloc = (
            df.groupby(['PORT_ID', 'ASSET_CLASS'])['WEIGHT'].sum()
            .unstack(fill_value=0)
            .reindex(columns=all_classes, fill_value=0)
            .rename(columns=asset_class_map)
            .reset_index()
        )

        df_comp_astc = (
            df.groupby(['PORT_ID'] + ports.prod_comp_keys + ['ASSET_CLASS'])['WEIGHT'].sum()
            .unstack(fill_value=0)
            .reindex(columns=all_classes, fill_value=0)
            .rename(columns=asset_class_map)
            .reset_index()
        )
        return df_astc_alloc, df_comp_astc

    def get_model_asset_class_allocation(self, ports):
        model_allocation =  ports.df_style \
                                    .merge(self.df_model, left_on='PORTPOP_STYLES'
                                        , right_on='MODEL_NAME'
                                        , suffixes=['','_MODEL']
                                        , how='left')
        model_allocation[self.asset_class_list] = model_allocation[self.asset_class_list]/100          
        asset_class_model_map = {x: x + '_MODEL' for x in self.asset_class_list}
        model_allocation = model_allocation.rename(columns=asset_class_model_map)
        select_columns = ['PORT_ID', 'PORT_INVESTMENT_STYLE', 'PORTPOP_STYLES'] \
        + [x+'_MODEL' for x in self.asset_class_list]

        return model_allocation[select_columns]

    def get_model_asset_class_deviation(self, ports):
        df_astc_alloc, df_comp_astc = self.get_asset_class_allocation(ports)
        model_allocation =  ports.df_style \
                                .merge(df_astc_alloc, on='PORT_ID', how='left') \
                                .merge(self.df_model, left_on='PORTPOP_STYLES'
                                    , right_on='MODEL_NAME'
                                    , suffixes=['','_MODEL']
                                    , how='left')
        diff_col = []
        for x in self.asset_class_list:
            model_allocation[x + "_DIF"] =  model_allocation[x] -  model_allocation[x + "_MODEL"]/100
            diff_col.append(x + "_DIF")

        return model_allocation[['PORT_ID', 'PORT_INVESTMENT_STYLE', 'PORTPOP_STYLES'] + diff_col]

    def get_geography_equity_allocation(self, ports):
        df_out_join_ge = self.df_out_join[self.df_out_join['ASSET_CLASS'] == 'Global Equity']
        df_out_join_ge_sum = df_out_join_ge.groupby(['PORT_ID'])['WEIGHT'].sum()
        df_out_join_ge = df_out_join_ge.merge(df_out_join_ge_sum.rename('WEIGHT_SUM')
                                        , left_on='PORT_ID'
                                        , right_index=True
                                        , how='left')

        for col in  self.equity_geography_list:
            df_out_join_ge[f'WEIGHTED_{col}'] = (df_out_join_ge['WEIGHT'] * df_out_join_ge[col])/df_out_join_ge['WEIGHT_SUM']

        alloc_cols = ['WEIGHTED_EM', 'WEIGHTED_EUROPE', 'WEIGHTED_JAPAN', 'WEIGHTED_US', 'WEIGHTED_OTHER']
        df_ge_alloc = (
            df_out_join_ge.groupby('PORT_ID')[alloc_cols].sum()
            .rename(columns={
                'WEIGHTED_EM': 'GE_EM',
                'WEIGHTED_EUROPE': 'GE_EUR',
                'WEIGHTED_JAPAN': 'GE_JP',
                'WEIGHTED_US': 'GE_US',
                'WEIGHTED_OTHER': 'GE_OTHER'
            })
            .reset_index()
        )

        df_ge_alloc.fillna(0, inplace=True)

        df_comp_ge = (
            df_out_join_ge.groupby(['PORT_ID'] + ports.prod_comp_keys)[alloc_cols].sum()
            .rename(columns={
                'WEIGHTED_EM': 'GE_EM',
                'WEIGHTED_EUROPE': 'GE_EUR',
                'WEIGHTED_JAPAN': 'GE_JP',
                'WEIGHTED_US': 'GE_US',
                'WEIGHTED_OTHER': 'GE_OTHER'
            })
            .reset_index()
            .fillna(0)
        )
        return df_ge_alloc, df_comp_ge

    def get_expected_return(self, ports):
        self.df_out_join['WEIGHTED_RETURN'] = self.df_out_join['WEIGHT'] * self.df_out_join['EXPECTED_RETURN']
        df_exp_ret = (
            self.df_out_join.groupby('PORT_ID', as_index=False)['WEIGHTED_RETURN'].sum()
            .rename(columns={'WEIGHTED_RETURN': 'EXPECTED_RETURN'})
        )
        df_comp_exp_ret = (
            self.df_out_join.groupby(['PORT_ID'] + ports.prod_comp_keys, as_index=False)['WEIGHTED_RETURN'].sum()
            .rename(columns={'WEIGHTED_RETURN': 'EXPECTED_RETURN'})
        )
        return df_exp_ret, df_comp_exp_ret

    def get_volatility(self, ports):
        df_ret_cov = self.df_ret_cov
        bm_names = df_ret_cov.columns

        df_weights = (
            self.df_out_join.groupby(['PORT_ID', 'BM_NAME'])['WEIGHT'].sum()
            .unstack(fill_value=0)
            .reindex(columns=bm_names, fill_value=0)
            .reindex(index=ports.port_ids, fill_value=0)
        )

        weight_matrix = df_weights.to_numpy()
        cov_matrix = df_ret_cov.to_numpy()
        stdev_ret_matrix = np.sqrt((weight_matrix @ cov_matrix * weight_matrix).sum(axis=1))
        vols_matrix = np.sqrt(52)*stdev_ret_matrix

        df_vol  = pd.DataFrame({'PORT_ID': ports.port_ids, 'VOLATILITY': vols_matrix})

        marginal_risk = np.sqrt(52)*weight_matrix * (weight_matrix @ cov_matrix) / np.where(stdev_ret_matrix[:, None] == 0, 1, stdev_ret_matrix[:, None])

        df_product_weights = (
            self.df_out_join.groupby(['PORT_ID', 'BM_NAME'] + ports.prod_comp_keys)['WEIGHT'].sum()
            .reset_index()
        )

        risk_contrib_df = pd.DataFrame(
            marginal_risk, index=ports.port_ids, columns=bm_names
        ).stack().rename('RISK_CONTRIBUTION').reset_index()
        risk_contrib_df.columns = ['PORT_ID', 'BM_NAME', 'RISK_CONTRIBUTION']

        df = df_product_weights.merge(risk_contrib_df, on=['PORT_ID', 'BM_NAME'], how='left')

        total_bm_weight = (
            df.groupby(['PORT_ID', 'BM_NAME'])['WEIGHT'].transform('sum')
        )
        df['VOL_CONTRIBUTION'] = df['WEIGHT'] * df['RISK_CONTRIBUTION'] / total_bm_weight

        df_comp_vol = (
            df.groupby(['PORT_ID'] + ports.prod_comp_keys)['VOL_CONTRIBUTION']
            .sum()
            .reset_index()
            .rename(columns={'VOL_CONTRIBUTION': 'VOLATILITY'})
        )

        return df_vol, df_comp_vol

    def get_portprop_matrix(self, ports):
        df_astc_alloc, df_comp_astc   = self.get_asset_class_allocation(ports)
        df_ge_alloc, df_comp_ge       = self.get_geography_equity_allocation(ports)
        df_exp_ret, df_comp_exp_ret   = self.get_expected_return(ports)
        df_vol, df_comp_vol           = self.get_volatility(ports)

        df_port_matrix =  ports.df_style \
            .merge(df_vol, on='PORT_ID', how='left') \
            .merge(df_exp_ret, on='PORT_ID', how='left') \
            .merge(df_astc_alloc, on='PORT_ID', how='left') \
            .merge(df_ge_alloc, on='PORT_ID', how='left')\
            .merge(self.df_model, left_on='PORTPOP_STYLES'
                   , right_on='MODEL_NAME'
                   , suffixes=['','_MODEL']
                   , how='left')
        df_port_matrix = df_port_matrix.fillna(0)

        join_key = ['PORT_ID'] + ports.prod_comp_keys
        df_comp_risk_div = ports.df_out[join_key + ['WEIGHT']] \
            .merge(df_comp_astc, on=join_key, how='left') \
            .merge(df_comp_ge, on=join_key, how='left') \
            .merge(df_comp_exp_ret, on=join_key, how='left') \
            .merge(df_comp_vol, on=join_key, how='left')
        return df_port_matrix, df_comp_risk_div


if __name__ == '__main__':
    ppm = PortpropMatrices()
    asset_class_model_map = {x: x + '_MODEL' for x in ppm.asset_class_list}
    print(asset_class_model_map)