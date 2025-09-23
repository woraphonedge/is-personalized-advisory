import os
import time
import warnings
from functools import wraps

import numpy as np
import pandas as pd

from .utils import read_parquet, read_sql, write_parquet

warnings.filterwarnings("ignore")


class HealthScore:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.load_reference_tables()

    def log_time_usage(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, "verbose", True):
                start = time.time()
                print(f"[{func.__name__}] started...")
                result = func(self, *args, **kwargs)
                end = time.time()
                print(f"[{func.__name__}] finished in {end - start:.4f} seconds.")
                return result
            else:
                return func(self, *args, **kwargs)
        return wrapper

    @log_time_usage
    def load_reference_tables(self):
        # Cache file name for the underlying mapping reference table
        UNDERLYING_MAPPING_FILE = "underlying_mapping.parquet"

        if str(os.getenv("LOAD_DATA_FROM_DWH", "")).strip().lower() in {"1", "true", "yes"}:
            df_underlying_mapping = read_sql(
                """
                SELECT * FROM edp.kkps_vw.v_npii_prod_undlycomp_map
                WHERE DATA_DT = (SELECT max(DATA_DT) FROM edp.kkps_vw.v_npii_prod_undlycomp_map)
                """
            )
            try:
                write_parquet(df_underlying_mapping, UNDERLYING_MAPPING_FILE)
            except Exception as e:
                print(f"[HealthScore] Warning: failed to persist underlying mapping: {e}")
        else:
            df_underlying_mapping = read_parquet(UNDERLYING_MAPPING_FILE)

        self.df_underlying_mapping = df_underlying_mapping

    @log_time_usage
    def get_score_port_risk_diversification(self, df_port_matrix, df_comp_risk_div, cal_comp=True):
        # df_port_matrix, df_comp_risk_div = self.get_portprop_matrix(cal_comp)
        # Vectorized scoring
        df_port_matrix["SCORE_RET"] = np.where(
            df_port_matrix["EXPECTED_RETURN"] < 0.8 * df_port_matrix["EXPECTED_RETURN_MODEL"], -1, 0
        )
        df_port_matrix["SCORE_VOL"] = np.where(
            df_port_matrix["VOLATILITY"]*100 > 1.2 * df_port_matrix["VOLATILITY_MODEL"], -1, 0
        )
        df_port_matrix["SCORE_PORTFOLIO_RISK"] = df_port_matrix["SCORE_VOL"] + df_port_matrix["SCORE_RET"]

        df_port_matrix["ACD"] = (
            (
                (df_port_matrix["AA_CASH"]*100 - df_port_matrix["AA_CASH_MODEL"]) ** 2 +
                (df_port_matrix["AA_FI"]*100 - df_port_matrix["AA_FI_MODEL"]) ** 2 +
                (df_port_matrix["AA_LE"]*100 - df_port_matrix["AA_LE_MODEL"]) ** 2 +
                (df_port_matrix["AA_GE"]*100 - df_port_matrix["AA_GE_MODEL"]) ** 2 +
                (df_port_matrix["AA_ALT"]*100 - df_port_matrix["AA_ALT_MODEL"]) ** 2
            ) ** 0.5
        )
        df_port_matrix["SCORE_ACD"] = np.where(df_port_matrix["ACD"]/5 > 10, -1, 0)

        df_port_matrix["GED"] = (
            (
                (df_port_matrix["GE_US"]*100 - df_port_matrix["GE_US_MODEL"]) ** 2 +
                (df_port_matrix["GE_EUR"]*100 - df_port_matrix["GE_EUR_MODEL"]) ** 2 +
                (df_port_matrix["GE_JP"]*100 - df_port_matrix["GE_JP_MODEL"]) ** 2 +
                (df_port_matrix["GE_EM"]*100 - df_port_matrix["GE_EM_MODEL"]) ** 2 +
                (df_port_matrix["GE_OTHER"]*100 - df_port_matrix["GE_OTHER_MODEL"]) ** 2
            ) ** 0.5
        )
        df_port_matrix["SCORE_GED"] = np.where(df_port_matrix["GED"]/5 > 4, -1, 0)

        df_port_matrix["SCORE_DIVERSIFICATION"] = df_port_matrix["SCORE_ACD"] + df_port_matrix["SCORE_GED"]

        df_risk_diver = df_port_matrix[["PORT_ID"
                                        , "EXPECTED_RETURN", "EXPECTED_RETURN_MODEL","SCORE_RET"
                                        , "VOLATILITY", "VOLATILITY_MODEL", "SCORE_VOL"
                                        , "SCORE_PORTFOLIO_RISK"
                                        , "ACD", "SCORE_ACD"
                                        , "GED", "SCORE_GED"
                                        , "SCORE_DIVERSIFICATION"]]

        if cal_comp:
            return df_risk_diver, df_comp_risk_div
        else:
            return df_risk_diver, None

    @log_time_usage
    def get_score_bulk_risk(self, ports, cal_comp=True):
        df_out_sel = ports.df_out.copy()
        df_out_sel['IS_BULK_RISK'] = (ports.df_out['IS_RISKY_ASSET']) & (df_out_sel['WEIGHT'] > 0.2)
        df_bulk_risk = (df_out_sel.groupby(['PORT_ID'])['IS_BULK_RISK']
                        .max()
                        .astype(int)
                        .reset_index()
                        .rename({'IS_BULK_RISK':"SCORE_BULK_RISK"}
                                , axis=1))
        df_bulk_risk['SCORE_BULK_RISK'] = -2 * df_bulk_risk['SCORE_BULK_RISK']

        if cal_comp:
            df_comp_bulk_risk = df_out_sel[['PORT_ID'] + ports.prod_comp_keys + ['IS_BULK_RISK']]
            return df_bulk_risk, df_comp_bulk_risk
        else:
            return df_bulk_risk, None

    @log_time_usage
    def get_score_issuer_risk(self, ports, cal_comp=True):
        df_out_underlying = ports.df_out.merge(self.df_underlying_mapping, on='PRODUCT_ID', how='left')

        # Filter GOV Bond Out
        df_out_underlying_filter = df_out_underlying[~( (df_out_underlying['PRODUCT_TYPE_DESC'] == 'Fixed Income') 
                                                        & (df_out_underlying['UNDERLYING_COMPANY'].isin(['BOT', 'MOF'])) )]


        df_out_underlying_group = (df_out_underlying_filter.groupby(['PORT_ID', 'UNDERLYING_COMPANY'])['WEIGHT']
                                .sum()
                                .reset_index())

        df_out_underlying_group['IS_ISSURE_RISK'] = df_out_underlying_group['WEIGHT'] > 0.2
        df_out_underlying_risk = df_out_underlying_group.groupby(['PORT_ID'])['IS_ISSURE_RISK'].max().reset_index()

        df_issuer_risk  = (pd.DataFrame({'PORT_ID': ports.port_ids})
                        .merge(df_out_underlying_risk, on='PORT_ID', how='left')
                        .fillna(False)
                        .rename({'IS_ISSURE_RISK':"SCORE_ISSUER_RISK"}, axis=1)
                        .astype(int)
                        )
        df_issuer_risk['SCORE_ISSUER_RISK'] = -2 * df_issuer_risk['SCORE_ISSUER_RISK']

        if cal_comp:
            df_comp_issuer_risk =  df_out_underlying_group[df_out_underlying_group['IS_ISSURE_RISK']]
            df_comp_issuer_risk['ISSURE_RISK_GROUP'] = df_comp_issuer_risk.groupby('PORT_ID').cumcount() + 1
            df_comp_issuer_risk = df_comp_issuer_risk[['PORT_ID', 'UNDERLYING_COMPANY', 'ISSURE_RISK_GROUP']]
            df_comp_issuer_risk = df_out_underlying_filter.merge(df_comp_issuer_risk, on=['PORT_ID', 'UNDERLYING_COMPANY'], how='left')
            df_comp_issuer_risk = df_comp_issuer_risk[['PORT_ID'] + ports.prod_comp_keys + ['UNDERLYING_COMPANY', 'ISSURE_RISK_GROUP']]
            return df_issuer_risk, df_comp_issuer_risk
        else:
            return df_issuer_risk, None

    @log_time_usage
    def get_score_not_monitor_risk(self, ports, cal_comp=True):
        df_not_monitor_weight = (
            ports.df_out.groupby(['PORT_ID', 'COVERAGE_PRDTYPE', 'IS_COVERAGE'])['VALUE'].sum() /
            ports.df_out.groupby(['PORT_ID', 'COVERAGE_PRDTYPE'])['VALUE'].sum()
        ) > 0.5
        df_not_monitor_weight = df_not_monitor_weight.reset_index(name='NOT_MONITOR_RISK')
        # Add total value per PORT_ID, COVERAGE_PRDTYPE for thresholding
        total_value = ports.df_out.groupby(['PORT_ID', 'COVERAGE_PRDTYPE'])['WEIGHT'].sum().reset_index()
        total_value = total_value.rename(columns={'WEIGHT': 'TOTAL_PRDTYPE_WEIGHT'})

        df_nm = df_not_monitor_weight.copy()
        df_nm = df_nm.merge(total_value, on=['PORT_ID', 'COVERAGE_PRDTYPE'], how='left')

        # For each prdtype, create a boolean mask
        for prdtype, col in [
            ('GLOBAL_STOCK', 'SCORE_NON_COVER_GLOBAL_STOCK'),
            ('LOCAL_STOCK', 'SCORE_NON_COVER_LOCAL_STOCK'),
            ('MUTUAL_FUND', 'SCORE_NON_COVER_MUTUAL_FUND'),
        ]:
            mask = (
                (df_nm['COVERAGE_PRDTYPE'] == prdtype) &
                (~df_nm['IS_COVERAGE']) &
                (df_nm['NOT_MONITOR_RISK']) &
                (df_nm['TOTAL_PRDTYPE_WEIGHT'] > 0.05)
            )
            df_nm[col] = mask.astype(int) * -1  # -1 if True, 0 if False

        # Group by PORT_ID, then for each column, take min (will be -1 if any are -1, else 0)
        score_cols = [
            'SCORE_NON_COVER_GLOBAL_STOCK',
            'SCORE_NON_COVER_LOCAL_STOCK',
            'SCORE_NON_COVER_MUTUAL_FUND'
        ]

        df_not_monitor_risk = (
            df_nm.groupby('PORT_ID')[score_cols].min().reset_index()
        )

        # Vectorized mapping for total score
        total = df_not_monitor_risk['SCORE_NON_COVER_GLOBAL_STOCK'] \
                + df_not_monitor_risk['SCORE_NON_COVER_LOCAL_STOCK'] \
                + df_not_monitor_risk['SCORE_NON_COVER_MUTUAL_FUND']
        df_not_monitor_risk['SCORE_NOT_MONITORED_PRODUCT'] = total.fillna(0).map({-3:-2, -2:-1.5, -1:-1, 0:0})

        if cal_comp:
            df_comp_not_monitor_product =  ports.df_out.merge(
                df_not_monitor_risk,
                on=['PORT_ID'],
                how='left'
            )

            df_comp_not_monitor_product['SCORE_NON_COVER_GLOBAL_STOCK'] = (
                ((df_comp_not_monitor_product['COVERAGE_PRDTYPE'] == 'GLOBAL_STOCK') &
                 (~df_comp_not_monitor_product['IS_COVERAGE']))
                * df_comp_not_monitor_product['SCORE_NON_COVER_GLOBAL_STOCK']
            )
            df_comp_not_monitor_product['SCORE_NON_COVER_LOCAL_STOCK'] = (
                ((df_comp_not_monitor_product['COVERAGE_PRDTYPE'] == 'LOCAL_STOCK') & 
                 (~df_comp_not_monitor_product['IS_COVERAGE']))
                * df_comp_not_monitor_product['SCORE_NON_COVER_LOCAL_STOCK']
            )
            df_comp_not_monitor_product['SCORE_NON_COVER_MUTUAL_FUND'] = (
                ((df_comp_not_monitor_product['COVERAGE_PRDTYPE'] == 'MUTUAL_FUND') & 
                 (~df_comp_not_monitor_product['IS_COVERAGE']))
                * df_comp_not_monitor_product['SCORE_NON_COVER_MUTUAL_FUND']
            )

            select_cols = ['PORT_ID'] + ports.prod_comp_keys + ['COVERAGE_PRDTYPE'
                            , 'SCORE_NON_COVER_GLOBAL_STOCK'
                            , 'SCORE_NON_COVER_LOCAL_STOCK'
                            ,'SCORE_NON_COVER_MUTUAL_FUND']

            df_comp_not_monitor_product = df_comp_not_monitor_product[select_cols]

            return df_not_monitor_risk, df_comp_not_monitor_product
        else:
            return df_not_monitor_risk, None

    @log_time_usage
    def get_health_score(self, ports, df_port_matrix, df_comp_risk_div, cal_comp=True):
        if ports.df_out is None:
            raise Exception('please set port first')

        df_port_risk_diversification, df_comp_risk_diversification = self.get_score_port_risk_diversification(df_port_matrix, df_comp_risk_div, cal_comp)
        df_bulk_risk, df_comp_bulk_risk = self.get_score_bulk_risk(ports, cal_comp)
        df_issuer_risk, df_comp_issuer_risk = self.get_score_issuer_risk(ports, cal_comp)
        df_not_monitor_risk, df_comp_not_monitor_product= self.get_score_not_monitor_risk(ports, cal_comp)

        df_health_score = pd.DataFrame(ports.port_ids)\
            .merge(df_port_risk_diversification, on='PORT_ID', how='left') \
            .merge(df_bulk_risk, on='PORT_ID', how='left') \
            .merge(df_issuer_risk, on='PORT_ID', how='left') \
            .merge(df_not_monitor_risk, on='PORT_ID', how='left')

        matrix_cols = ['SCORE_PORTFOLIO_RISK', 'SCORE_DIVERSIFICATION', 'SCORE_BULK_RISK', 'SCORE_ISSUER_RISK', 'SCORE_NOT_MONITORED_PRODUCT']
        df_health_score[matrix_cols] = df_health_score[matrix_cols].fillna(0)

        df_health_score['HEALTH_SCORE'] =  10 + (df_health_score["SCORE_PORTFOLIO_RISK"]
                                    + df_health_score["SCORE_DIVERSIFICATION"]
                                    + df_health_score["SCORE_BULK_RISK"]
                                    + df_health_score["SCORE_ISSUER_RISK"]
                                    + df_health_score["SCORE_NOT_MONITORED_PRODUCT"])

        if cal_comp:
            join_key = ['PORT_ID'] + ports.prod_comp_keys
            df_joined_comp = ports.df_out[join_key + ['PRODUCT_TYPE_DESC', 'VALUE']] \
                .merge(df_comp_risk_diversification, on=join_key, how='left') \
                .merge(df_comp_bulk_risk, on=join_key, how='left') \
                .merge(df_comp_issuer_risk, on=join_key, how='left') \
                .merge(df_comp_not_monitor_product, on=join_key, how='left')

            ## treat null for .display()
            df_joined_comp['COVERAGE_PRDTYPE'] = np.where(df_joined_comp['COVERAGE_PRDTYPE'].isna(),
                                                            np.nan,
                                                            df_joined_comp['COVERAGE_PRDTYPE'])
            # df_joined_comp['PRODUCT_DISPLAY_NAME'] = np.where(df_joined_comp['PRODUCT_DISPLAY_NAME'].isna(),
            #                                                 np.nan,
            #                                                 df_joined_comp['PRODUCT_DISPLAY_NAME'])

            return df_health_score, df_joined_comp
        else:
            return df_health_score, None

if __name__ == "__main__":
    hs = HealthScore(verbose=True)
    # ref_tables = hs.load_reference_tables()
    # hs.set_reference_tables(ref_tables)
