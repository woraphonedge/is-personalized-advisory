import time
import warnings
from functools import wraps

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class HealthScore:
    def __init__(self, verbose=False):
        self.verbose = verbose

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
    def get_score_port_risk_diversification(self, df_port_matrix, df_comp_risk_div, cal_comp=True):
        df_port_matrix["score_ret"] = np.where(
            df_port_matrix["expected_return"] < 0.8 * df_port_matrix["expected_return_model"], -1, 0
        )
        df_port_matrix["score_vol"] = np.where(
            df_port_matrix["volatility"] > 1.2 * df_port_matrix["volatility_model"], -1, 0
        )
        df_port_matrix["score_portfolio_risk"] = df_port_matrix["score_vol"] + df_port_matrix["score_ret"]

        df_port_matrix["acd"] = (
            (
                (df_port_matrix["aa_cash"]  - df_port_matrix["aa_cash_model"]) ** 2 +
                (df_port_matrix["aa_fi"]  - df_port_matrix["aa_fi_model"]) ** 2 +
                (df_port_matrix["aa_le"]  - df_port_matrix["aa_le_model"]) ** 2 +
                (df_port_matrix["aa_ge"]  - df_port_matrix["aa_ge_model"]) ** 2 +
                (df_port_matrix["aa_alt"]  - df_port_matrix["aa_alt_model"]) ** 2
            ) ** 0.5
        )
        df_port_matrix["score_acd"] = np.where(df_port_matrix["acd"] / 5 > 10, -1, 0)

        df_port_matrix["ged"] = (
            (
                (df_port_matrix["ge_us"]  - df_port_matrix["ge_us_model"]) ** 2 +
                (df_port_matrix["ge_eur"]  - df_port_matrix["ge_eur_model"]) ** 2 +
                (df_port_matrix["ge_jp"]  - df_port_matrix["ge_jp_model"]) ** 2 +
                (df_port_matrix["ge_em"]  - df_port_matrix["ge_em_model"]) ** 2 +
                (df_port_matrix["ge_other"]  - df_port_matrix["ge_other_model"]) ** 2
            ) ** 0.5
        )
        df_port_matrix["score_ged"] = np.where(df_port_matrix["ged"] / 5 > 4, -1, 0)

        df_port_matrix["score_diversification"] = df_port_matrix["score_acd"] + df_port_matrix["score_ged"]

        df_risk_diver = df_port_matrix[[
            "port_id", "expected_return", "expected_return_model", "score_ret",
            "volatility", "volatility_model", "score_vol",
            "score_portfolio_risk", "acd", "score_acd",
            "ged", "score_ged", "score_diversification"
        ]]

        if cal_comp:
            return df_risk_diver, df_comp_risk_div
        else:
            return df_risk_diver, None

    @log_time_usage
    def get_score_bulk_risk(self, ports, cal_comp=True):
        df_out_sel = ports.df_out.copy()
        df_out_sel["is_bulk_risk"] = (ports.df_out["is_risky_asset"]) & (df_out_sel["weight"] > 0.2)

        df_bulk_risk = (
            df_out_sel.groupby(["port_id"])["is_bulk_risk"]
            .max()
            .astype(int)
            .reset_index()
            .rename({"is_bulk_risk": "score_bulk_risk"}, axis=1)
        )
        df_bulk_risk["score_bulk_risk"] = -2 * df_bulk_risk["score_bulk_risk"]

        if cal_comp:
            df_comp_bulk_risk = df_out_sel[["port_id"] + ports.prod_comp_keys + ["is_bulk_risk"]]
            return df_bulk_risk, df_comp_bulk_risk
        else:
            return df_bulk_risk, None

    @log_time_usage
    def get_score_issuer_risk(self, ports, cal_comp=True):
        df_out_underlying = ports.df_out.merge(ports.product_underlying, on="product_id", how="left")

        df_out_underlying_filter = df_out_underlying[~(
            (df_out_underlying["product_type_desc"] == "Fixed Income") &
            (df_out_underlying["underlying_company"].isin(["BOT", "MOF"]))
        )]

        df_out_underlying_group = (
            df_out_underlying_filter.groupby(["port_id", "underlying_company"])["weight"]
            .sum()
            .reset_index()
        )

        df_out_underlying_group["is_issure_risk"] = df_out_underlying_group["weight"] > 0.2
        df_out_underlying_risk = df_out_underlying_group.groupby(["port_id"])["is_issure_risk"].max().reset_index()

        df_issuer_risk = (
            pd.DataFrame({"port_id": ports.port_ids})
            .merge(df_out_underlying_risk, on="port_id", how="left")
            .fillna(False)
            .rename({"is_issure_risk": "score_issuer_risk"}, axis=1)
            .astype(int)
        )
        df_issuer_risk["score_issuer_risk"] = -2 * df_issuer_risk["score_issuer_risk"]

        if cal_comp:
            df_comp_issuer_risk = df_out_underlying_group[df_out_underlying_group["is_issure_risk"]]
            df_comp_issuer_risk["issure_risk_group"] = df_comp_issuer_risk.groupby("port_id").cumcount() + 1
            df_comp_issuer_risk = df_comp_issuer_risk[["port_id", "underlying_company", "issure_risk_group"]]
            df_comp_issuer_risk = df_out_underlying_filter.merge(
                df_comp_issuer_risk, on=["port_id", "underlying_company"], how="left"
            )
            df_comp_issuer_risk = df_comp_issuer_risk[["port_id"] + ports.prod_comp_keys + ["underlying_company", "issure_risk_group"]]
            return df_issuer_risk, df_comp_issuer_risk
        else:
            return df_issuer_risk, None

    @log_time_usage
    def get_score_not_monitor_risk(self, ports, cal_comp=True):
        df_not_monitor_weight = (
            ports.df_out.groupby(["port_id", "coverage_prdtype", "is_coverage"])["value"].sum() /
            ports.df_out.groupby(["port_id", "coverage_prdtype"])["value"].sum()
        ) > 0.5
        df_not_monitor_weight = df_not_monitor_weight.reset_index(name="not_monitor_risk")

        total_value = ports.df_out.groupby(["port_id", "coverage_prdtype"])["weight"].sum().reset_index()
        total_value = total_value.rename(columns={"weight": "total_prdtype_weight"})

        df_nm = df_not_monitor_weight.merge(total_value, on=["port_id", "coverage_prdtype"], how="left")

        for prdtype, col in [
            ("GLOBAL_STOCK", "score_non_cover_global_stock"),
            ("LOCAL_STOCK", "score_non_cover_local_stock"),
            ("MUTUAL_FUND", "score_non_cover_mutual_fund"),
        ]:
            mask = (
                (df_nm["coverage_prdtype"] == prdtype) &
                (~df_nm["is_coverage"]) &
                (df_nm["not_monitor_risk"]) &
                (df_nm["total_prdtype_weight"] > 0.05)
            )
            df_nm[col] = mask.astype(int) * -1

        score_cols = [
            "score_non_cover_global_stock",
            "score_non_cover_local_stock",
            "score_non_cover_mutual_fund",
        ]
        df_not_monitor_risk = df_nm.groupby("port_id")[score_cols].min().reset_index()

        total = (
            df_not_monitor_risk["score_non_cover_global_stock"]
            + df_not_monitor_risk["score_non_cover_local_stock"]
            + df_not_monitor_risk["score_non_cover_mutual_fund"]
        )
        df_not_monitor_risk["score_not_monitored_product"] = total.fillna(0).map({-3: -2, -2: -1.5, -1: -1, 0: 0})

        if cal_comp:
            df_comp_not_monitor_product = ports.df_out.merge(
                df_not_monitor_risk, on=["port_id"], how="left"
            )

            for prdtype, col in [
                ("GLOBAL_STOCK", "score_non_cover_global_stock"),
                ("LOCAL_STOCK", "score_non_cover_local_stock"),
                ("MUTUAL_FUND", "score_non_cover_mutual_fund"),
            ]:
                df_comp_not_monitor_product[col] = (
                    ((df_comp_not_monitor_product["coverage_prdtype"] == prdtype) &
                     (~df_comp_not_monitor_product["is_coverage"]))
                    * df_comp_not_monitor_product[col]
                )

            select_cols = ["port_id"] + ports.prod_comp_keys + [
                "coverage_prdtype",
                "score_non_cover_global_stock",
                "score_non_cover_local_stock",
                "score_non_cover_mutual_fund",
            ]
            df_comp_not_monitor_product = df_comp_not_monitor_product[select_cols]

            return df_not_monitor_risk, df_comp_not_monitor_product
        else:
            return df_not_monitor_risk, None

    @log_time_usage
    def get_health_score(self, ports, df_port_matrix, df_comp_risk_div, cal_comp=True):
        if ports.df_out is None:
            raise Exception("please set port first")

        df_port_risk_diversification, df_comp_risk_diversification = self.get_score_port_risk_diversification(
            df_port_matrix, df_comp_risk_div, cal_comp
        )
        df_bulk_risk, df_comp_bulk_risk = self.get_score_bulk_risk(ports, cal_comp)
        df_issuer_risk, df_comp_issuer_risk = self.get_score_issuer_risk(ports, cal_comp)
        df_not_monitor_risk, df_comp_not_monitor_product = self.get_score_not_monitor_risk(ports, cal_comp)

        df_health_score = (
            pd.DataFrame(ports.port_ids)
            .merge(df_port_risk_diversification, on="port_id", how="left")
            .merge(df_bulk_risk, on="port_id", how="left")
            .merge(df_issuer_risk, on="port_id", how="left")
            .merge(df_not_monitor_risk, on="port_id", how="left")
        )

        matrix_cols = [
            "score_portfolio_risk",
            "score_diversification",
            "score_bulk_risk",
            "score_issuer_risk",
            "score_not_monitored_product",
        ]
        df_health_score[matrix_cols] = df_health_score[matrix_cols].fillna(0)

        df_health_score["health_score"] = 10 + (
            df_health_score["score_portfolio_risk"]
            + df_health_score["score_diversification"]
            + df_health_score["score_bulk_risk"]
            + df_health_score["score_issuer_risk"]
            + df_health_score["score_not_monitored_product"]
        )

        if cal_comp:
            join_key = ["port_id"] + ports.prod_comp_keys
            df_joined_comp = (
                ports.df_out[join_key + ["product_type_desc", "value"]]
                .merge(df_comp_risk_diversification, on=join_key, how="left")
                .merge(df_comp_bulk_risk, on=join_key, how="left")
                .merge(df_comp_issuer_risk, on=join_key, how="left")
                .merge(df_comp_not_monitor_product, on=join_key, how="left")
            )

            df_joined_comp["coverage_prdtype"] = np.where(
                df_joined_comp["coverage_prdtype"].isna(), np.nan, df_joined_comp["coverage_prdtype"]
            )

            return df_health_score, df_joined_comp
        else:
            return df_health_score, None


if __name__ == "__main__":
    hs = HealthScore(verbose=True)
    # ref_tables = hs.load_reference_tables()
    # hs.set_reference_tables(ref_tables)
