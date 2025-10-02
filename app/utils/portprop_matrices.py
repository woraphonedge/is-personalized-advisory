import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class PortpropMatrices():
    def __init__(self, ref_dict: dict = None):
        self.set_ref_tables(ref_dict)
        self.cal_ret_cov()
        
        self.asset_class_list = ['aa_alt', 'aa_cash', 'aa_fi', 'aa_ge', 'aa_le']
        self.equity_geography_list = ['em', 'europe', 'japan', 'us', 'other']
        self.df_out_join = None

    def set_ref_tables(self, ref_dict: dict):

        # Set DataFrames with validation
        self.df_port_fs = None if 'portprop_factsheet' not in ref_dict else ref_dict['portprop_factsheet']
        self.df_port_fb =  None if 'portprop_fallback' not in ref_dict else ref_dict['portprop_fallback']
        self.df_port_bm =  None if 'portprop_benchmark' not in ref_dict else ref_dict['portprop_benchmark']
        self.df_ge_mapping =  None if 'portprop_ge_mapping' not in ref_dict else ref_dict['portprop_ge_mapping']
        self.df_port_ret =  None if 'portprop_ret_eow' not in ref_dict else ref_dict['portprop_ret_eow']
        self.df_model =  None if 'advisory_health_score' not in ref_dict else ref_dict['advisory_health_score']
        
        # Validate required elements
        required_elements = {
            "df_port_fs": self.df_port_fs,
            "df_port_fb": self.df_port_fb,
            "df_port_bm": self.df_port_bm,
            "df_ge_mapping": self.df_ge_mapping,
            "df_port_ret": self.df_port_ret,
            "df_model": self.df_model
        }
        for key, value in required_elements.items():
            if value is None:
                warnings.warn(f"'{key}' not provided. Expect errors if used.")


    def cal_ret_cov(self):
        df_ret_pivot = self.df_port_ret.pivot_table(index='return_date'
                                               , columns='bm_name'
                                               , values='return'
                                               , aggfunc='sum')
        self.df_ret_cov = df_ret_pivot.cov()

    def cal_df_out_join_bm(self, ports):
        df_out_join = ports.df_out.copy()

        df_out_join['asset_allo_join_key'] = np.where(
            df_out_join['asset_class_name'] == 'Allocation',
            df_out_join['symbol'],
            np.nan
        )

        df_out_join = df_out_join.merge(
            self.df_port_fs,
            how='left',
            left_on=['asset_allo_join_key'],
            right_on=['symbol'],
            suffixes=('', '_fs')
        )

        df_out_join['asset_allo_fallback_join_key'] = np.where(
            (df_out_join['asset_allo_join_key'].notna()) & (df_out_join['symbol_fs'].isna()),
            'FALLBACK',
            pd.NA
        )

        df_out_join = df_out_join.merge(
            self.df_port_fb,
            how='left',
            left_on=['asset_allo_fallback_join_key'],
            right_on=['symbol'],
            suffixes=('', '_fb')
        )

        df_out_join = df_out_join.merge(
            self.df_port_bm,
            how='left',
            left_on=['pp_asset_sub_class'],
            right_on=['pp_asset_sub_class'],
            suffixes=('', '_bm')
        )

        # Vectorized assignments for 'ASSET_CLASS', 'BM_NAME', 'WEIGHT'
        is_alloc = df_out_join['asset_class_name'] == 'Allocation'
        fallback = df_out_join['asset_allo_fallback_join_key'] == 'FALLBACK'

        df_out_join['asset_class'] = np.where(
            ~is_alloc,
            df_out_join['asset_class_name'],
            np.where(
                ~fallback,
                df_out_join['asset_class'],
                df_out_join['asset_class_fb']
            )
        )
        df_out_join['bm_name'] = np.where(
            ~is_alloc,
            df_out_join['bm_name_bm'],
            np.where(
                ~fallback,
                df_out_join['bm_name'],
                df_out_join['bm_name_fb']
            )
        )
        df_out_join['weight'] = np.where(
            ~is_alloc,
            df_out_join['weight'],
            np.where(
                ~fallback,
                df_out_join['weight_fs'] * df_out_join['weight'],
                df_out_join['weight_fb'] * df_out_join['weight']
            )
        )

        df_out_join = df_out_join[list(ports.df_out.columns) + ['asset_class','bm_name']]

        df_out_join = df_out_join.merge(
            self.df_ge_mapping,
            how='left',
            left_on=['bm_name'],
            right_on=['bm_name'],
            suffixes=('','_ge'))

        self.df_out_join = df_out_join

    def get_asset_class_allocation(self, ports):
        asset_class_map = {
            'Alternative': 'aa_alt',
            'Cash and Cash Equivalent': 'aa_cash',
            'Fixed Income': 'aa_fi',
            'Global Equity': 'aa_ge',
            'Local Equity': 'aa_le'
        }
        all_classes = list(asset_class_map.keys())

        df = self.df_out_join
        df_astc_alloc = (
            df.groupby(['port_id', 'asset_class'])['weight'].sum()
            .unstack(fill_value=0)
            .reindex(columns=all_classes, fill_value=0)
            .rename(columns=asset_class_map)
            .reset_index()
        )

        df_comp_astc = (
            df.groupby(['port_id'] + ports.prod_comp_keys + ['asset_class'])['weight'].sum()
            .unstack(fill_value=0)
            .reindex(columns=all_classes, fill_value=0)
            .rename(columns=asset_class_map)
            .reset_index()
        )
        return df_astc_alloc, df_comp_astc

    def get_model_asset_class_allocation(self, ports):
        model_allocation =  ports.df_style \
                                    .merge(self.df_model, left_on='portpop_styles'
                                        , right_on='model_name'
                                        , suffixes=['','_model']
                                        , how='left')
        model_allocation[self.asset_class_list] = model_allocation[self.asset_class_list]         
        asset_class_model_map = {x: x + '_model' for x in self.asset_class_list}
        model_allocation = model_allocation.rename(columns=asset_class_model_map)
        select_columns = ['port_id', 'port_investment_style', 'portpop_styles'] \
        + [x+'_model' for x in self.asset_class_list]

        return model_allocation[select_columns]

    def get_model_asset_class_deviation(self, ports):
        df_astc_alloc, df_comp_astc = self.get_asset_class_allocation(ports)
        model_allocation =  ports.df_style \
                                .merge(df_astc_alloc, on='port_id', how='left') \
                                .merge(self.df_model, left_on='portpop_styles'
                                    , right_on='model_name'
                                    , suffixes=['','_model']
                                    , how='left')
        diff_col = []
        for x in self.asset_class_list:
            model_allocation[x + "_dif"] =  model_allocation[x] -  model_allocation[x + "_model"]
            diff_col.append(x + "_dif")

        return model_allocation[['port_id', 'port_investment_style', 'portpop_styles'] + diff_col]

    def get_geography_equity_allocation(self, ports):
        df_out_join_ge = self.df_out_join[self.df_out_join['asset_class'] == 'Global Equity']
        df_out_join_ge_sum = df_out_join_ge.groupby(['port_id'])['weight'].sum()
        df_out_join_ge = df_out_join_ge.merge(df_out_join_ge_sum.rename('weight_sum')
                                        , left_on='port_id'
                                        , right_index=True
                                        , how='left')

        for col in  self.equity_geography_list:
            df_out_join_ge[f'weighted_{col}'] = (df_out_join_ge['weight'] * df_out_join_ge[col])/df_out_join_ge['weight_sum']

        alloc_cols = ['weighted_em', 'weighted_europe', 'weighted_japan', 'weighted_us', 'weighted_other']
        df_ge_alloc = (
            df_out_join_ge.groupby('port_id')[alloc_cols].sum()
            .rename(columns={
                'weighted_em': 'ge_em',
                'weighted_europe': 'ge_eur',
                'weighted_japan': 'ge_jp',
                'weighted_us': 'ge_us',
                'weighted_other': 'ge_other'
            })
            .reset_index()
        )

        df_ge_alloc.fillna(0, inplace=True)

        df_comp_ge = (
            df_out_join_ge.groupby(['port_id'] + ports.prod_comp_keys)[alloc_cols].sum()
            .rename(columns={
                'weighted_em': 'ge_em',
                'weighted_europe': 'ge_eur',
                'weighted_japan': 'ge_jp',
                'weighted_us': 'ge_us',
                'weighted_other': 'ge_other'
            })
            .reset_index()
            .fillna(0)
        )
        return df_ge_alloc, df_comp_ge

    def get_expected_return(self, ports):
        self.df_out_join['weighted_return'] = self.df_out_join['weight'] * self.df_out_join['expected_return']
        df_exp_ret = (
            self.df_out_join.groupby('port_id', as_index=False)['weighted_return'].sum()
            .rename(columns={'weighted_return': 'expected_return'})
        )
        df_comp_exp_ret = (
            self.df_out_join.groupby(['port_id'] + ports.prod_comp_keys, as_index=False)['weighted_return'].sum()
            .rename(columns={'weighted_return': 'expected_return'})
        )
        return df_exp_ret, df_comp_exp_ret

    def get_volatility(self, ports):
        df_ret_cov = self.df_ret_cov
        bm_names = df_ret_cov.columns

        df_weights = (
            self.df_out_join.groupby(['port_id', 'bm_name'])['weight'].sum()
            .unstack(fill_value=0)
            .reindex(columns=bm_names, fill_value=0)
            .reindex(index=ports.port_ids, fill_value=0)
        )

        weight_matrix = df_weights.to_numpy()
        cov_matrix = df_ret_cov.to_numpy()
        stdev_ret_matrix = np.sqrt((weight_matrix @ cov_matrix * weight_matrix).sum(axis=1))
        vols_matrix = np.sqrt(52)*stdev_ret_matrix

        df_vol  = pd.DataFrame({'port_id': ports.port_ids, 'volatility': vols_matrix})

        marginal_risk = np.sqrt(52)*weight_matrix * (weight_matrix @ cov_matrix) / np.where(stdev_ret_matrix[:, None] == 0, 1, stdev_ret_matrix[:, None])

        df_product_weights = (
            self.df_out_join.groupby(['port_id', 'bm_name'] + ports.prod_comp_keys)['weight'].sum()
            .reset_index()
        )

        risk_contrib_df = pd.DataFrame(
            marginal_risk, index=ports.port_ids, columns=bm_names
        ).stack().rename('risk_contribution').reset_index()
        risk_contrib_df.columns = ['port_id', 'bm_name', 'risk_contribution']

        df = df_product_weights.merge(risk_contrib_df, on=['port_id', 'bm_name'], how='left')

        total_bm_weight = (
            df.groupby(['port_id', 'bm_name'])['weight'].transform('sum')
        )
        df['vol_contribution'] = df['weight'] * df['risk_contribution'] / total_bm_weight

        df_comp_vol = (
            df.groupby(['port_id'] + ports.prod_comp_keys)['vol_contribution']
            .sum()
            .reset_index()
            .rename(columns={'vol_contribution': 'volatility'})
        )

        return df_vol, df_comp_vol

    def get_portprop_matrix(self, ports):
        df_astc_alloc, df_comp_astc   = self.get_asset_class_allocation(ports)
        df_ge_alloc, df_comp_ge       = self.get_geography_equity_allocation(ports)
        df_exp_ret, df_comp_exp_ret   = self.get_expected_return(ports)
        df_vol, df_comp_vol           = self.get_volatility(ports)

        df_port_matrix =  ports.df_style \
            .merge(df_vol, on='port_id', how='left') \
            .merge(df_exp_ret, on='port_id', how='left') \
            .merge(df_astc_alloc, on='port_id', how='left') \
            .merge(df_ge_alloc, on='port_id', how='left')\
            .merge(self.df_model, left_on='portpop_styles'
                   , right_on='model_name'
                   , suffixes=['','_model']
                   , how='left')
        df_port_matrix = df_port_matrix.fillna(0)

        join_key = ['port_id'] + ports.prod_comp_keys
        df_comp_risk_div = ports.df_out[join_key + ['weight']] \
            .merge(df_comp_astc, on=join_key, how='left') \
            .merge(df_comp_ge, on=join_key, how='left') \
            .merge(df_comp_exp_ret, on=join_key, how='left') \
            .merge(df_comp_vol, on=join_key, how='left')
        return df_port_matrix, df_comp_risk_div


if __name__ == '__main__':
    ppm = PortpropMatrices()
    asset_class_model_map = {x: x + '_model' for x in ppm.asset_class_list}
    print(asset_class_model_map)