import copy
import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from .portfolios import Portfolios


class Rebalancer:
    """
    A portfolio rebalancer that
    (1) sells to remove/trim risky or non-monitored holdings and
    (2) buys model-aligned products using ranked recommendations until the portfolio is within
    tolerance vs model weights.

    Expected dependencies provided by caller:
      - ports: Portfolios object (with df_out, df_style, port_ids, product_mapping loaded)
      - ppm:   PortpropMatrices object
      - hs:    HealthScore object
    """

    def __init__(
        self,
        customer_id: int | None = None,
        client_investment_style: str | None = None,
        client_classification: str | None = None,
        new_money: float = 0.0,

        # -------- constraints --------
        private_percent: float = 0.0,
        cash_percent: float | None = None,
        offshore_percent: float | None = None,
        product_whitelist: list[str] | None = None,
        product_blacklist: list[str] | None = None,
        discretionary_acceptance: float | None = None,

        # -------- params --------
        eps: float = 1e-8,
        eps_tiny: float = 1e-12,
        underweight_threshold: float = 0.05,
        buy_step_weight_max: float = 0.10,
        single_line_cap_non_mandate: float = 0.20,
        max_rebalance_loops: int = 50,
        min_amount_per_row: float = 1_000.0,
        min_weight_per_row: float = 0.05,
        min_mandate_amount: float = 1_000_000.0,
    ) -> None:

        self.customer_id = customer_id
        self.client_investment_style = client_investment_style
        self.client_classification = client_classification
        self.new_money = new_money

        self.private_percent = private_percent
        self.cash_percent = cash_percent
        self.offshore_percent = offshore_percent
        self.product_whitelist = [p.lower() for p in product_whitelist] if product_whitelist else []
        self.product_blacklist = [p.lower() for p in product_blacklist] if product_blacklist else []
        self.discretionary_acceptance = 0.5 if discretionary_acceptance is None else float(discretionary_acceptance)

        self.eps = float(eps)
        self.eps_tiny = float(eps_tiny)
        self.underweight_threshold = float(underweight_threshold)
        self.buy_step_weight_max = float(buy_step_weight_max)
        self.single_line_cap_non_mandate = float(single_line_cap_non_mandate)
        self.max_rebalance_loops = int(max_rebalance_loops)
        self.min_amount_per_row = float(min_amount_per_row)
        self.min_weight_per_row = float(min_weight_per_row)
        self.min_mandate_amount = float(min_mandate_amount)

        # sequences for transaction/batch ids
        self.txn_seq = 0
        self.batch_seq = 0

        # keep a clean recommendations frame template for easy resets
        self.reco_cols = [
            "transaction_no", "batch_no", "port_id", "product_id", "src_sharecodes", "desk",
            "port_type", "currency","product_display_name", "product_type_desc", "asset_class_name",
            "value", "weight", "flag", "expected_weight", "action", "amount",
        ]
        self.recommendations = pd.DataFrame(columns=self.reco_cols)

        # preload raw tables
        self.es_sell_list: pd.DataFrame | None = None
        self.prod_reco_rank_raw: pd.DataFrame | None = None
        self.discretionary_allo_weight: pd.DataFrame | None = None


    # ---------- state mgmt ----------
    def reset_state(self) -> None:
        """Clear per-run state so the same instance can be reused safely."""
        self.txn_seq = 0
        self.batch_seq = 0
        self.recommendations = pd.DataFrame(columns=self.reco_cols)

    # ---------- cached loaders ----------
    def set_ref_tables(self, ref_dict: dict):
        """Reload cached reference tables (if you expect DB to have changed)."""
        # Set DataFrames with validation
        self.es_sell_list = None if "es_sell_list" not in ref_dict else ref_dict["es_sell_list"]
        self.prod_reco_rank_raw = None if "product_recommendation_rank_raw" not in ref_dict else ref_dict["product_recommendation_rank_raw"]
        self.discretionary_allo_weight = None if "mandate_allocation" not in ref_dict else ref_dict["mandate_allocation"]

        # Validate required elements
        required_elements = {
            "es_sell_list": self.es_sell_list,
            "product_recommendation_rank_raw": self.prod_reco_rank_raw,
            "mandate_allocation": self.discretionary_allo_weight,
        }
        for key, value in required_elements.items():
            if value is None:
                warnings.warn(f"'{key}' not provided. Expect errors if used.", UserWarning, stacklevel=2)

    def get_product_recommendation_rank(self, ports) -> pd.DataFrame:
        """Merge the (preloaded) recommendation rank with current product mapping."""
        # Guard: if reference not loaded, warn and use empty frame with expected columns
        if not hasattr(self, "prod_reco_rank_raw") or self.prod_reco_rank_raw is None:
            logger = logging.getLogger(__name__)
            logger.warning("prod_reco_rank_raw is None; continuing with empty recommendations")
            self.prod_reco_rank_raw = pd.DataFrame(columns=["src_sharecodes", "desk", "currency"])  # minimal expected keys
        prod_reco_rank = self.prod_reco_rank_raw.merge(
            ports.product_mapping,
            on=["src_sharecodes", "desk", "currency"],
            how="left",
            suffixes=("", "_mapping")
        )
        return prod_reco_rank

    # ---------------------------
    # Cash Proxy funding
    # ---------------------------
    def build_cash_proxy_funding(self, ports, action: pd.DataFrame, per_row: bool = True) -> pd.DataFrame:
        """
        Unified builder for cash proxy funding.
        - per_row=True  -> 1:1 funding per input row (use for sell / cash overweight)
        - per_row=False -> aggregate by currency then fund (use for buy)
        """
        if action is None or action.empty:
            return pd.DataFrame(columns=self.reco_cols)
        if ports.product_mapping is None:
            raise RuntimeError("Product mapping must be loaded before mapping cash proxy.")

        cash_map = ports.product_mapping[ports.product_mapping["symbol"] == "CASH PROXY"][
            ports.prod_comp_keys + ["product_display_name", "product_type_desc", "asset_class_name"]
        ].copy()

        funding = action[["port_id","currency","amount"]].merge(cash_map, on="currency", how="left")
        funding["amount"] = -funding["amount"].astype(float)
        funding["value"] = np.nan
        funding["weight"] = np.nan
        funding["flag"] = "cash_proxy_funding"
        funding["expected_weight"] = np.nan
        funding["action"] = "funding"

        return funding[action.columns]

    def get_cash_proxy(self, ports) -> pd.DataFrame:
        ## Comment: Should it belong to ports?
        cash = ports.df_out[ports.df_out["symbol"] == "CASH PROXY"]
        if cash.empty:
            return cash
        cash_ccy = (cash.groupby(["port_id","currency"], as_index=False)
                        .agg(weight=("weight","sum"), value=("value","sum"))).sort_values("weight", ascending=False)
        return cash_ccy

    # ---------------------------
    # Portfolio updater
    # ---------------------------
    # to be deleted later after ensure that it is not needed
    # def _ensure_symbol_column(self, ports: "Portfolios") -> None:
    #     """Ensure ports.df_out has a 'symbol' column for downstream consumers.
    #     Uses 'src_sharecodes' or 'src_symbol' fallback if needed.
    #     """
    #     logger = logging.getLogger(__name__)
    #     try:
    #         if hasattr(ports, "df_out") and ports.df_out is not None:
    #             cols = list(ports.df_out.columns)
    #             if "symbol" not in cols:
    #                 if "src_sharecodes" in cols:
    #                     ports.df_out["symbol"] = ports.df_out["src_sharecodes"].astype(str)
    #                 elif "src_symbol" in cols:
    #                     ports.df_out["symbol"] = ports.df_out["src_symbol"].astype(str)
    #     except Exception as e:
    #         logger.warning("Failed to ensure symbol column exists: %s", e, exc_info=True)
    def update_portfolio(self, ports: "Portfolios", actions: pd.DataFrame) -> None:
        # Create new_ports if not already created
        if not hasattr(self, "new_ports") or self.new_ports is None:
            self.new_ports = copy.deepcopy(ports)
            # Ensure symbol col exists
            # self._ensure_symbol_column(self.new_ports)

        # Prepare actions as position changes
        action_positions = (
            actions[["port_id", "amount"] + self.new_ports.prod_comp_keys]
            .rename(columns={"amount": "value"})
            .copy()
        )

        # Get current portfolio positions
        current_positions = self.new_ports.df_out[["port_id", "value"] + self.new_ports.prod_comp_keys].copy()

        # Combine current and actions positions
        new_positions = pd.concat([current_positions, action_positions], ignore_index=True, sort=False)

        # Aggregate positions by product keys
        new_positions = (
            new_positions.groupby(
                ["port_id"] + self.new_ports.prod_comp_keys,
                dropna=False,
                as_index=False,
            )["value"]
            .sum()
        )

        # Merge product mapping
        df_out = new_positions.merge(
            self.new_ports.product_mapping,
            on=self.new_ports.prod_comp_keys,
            how="left",
        )

        # Filter out zero positions
        df_out = df_out[df_out["value"] != 0]

        # Update new ports
        self.new_ports.set_portfolio(df_out, self.new_ports.df_style, self.new_ports.port_ids, self.new_ports.port_id_mapping)
        # Ensure symbol after recompute
        # self._ensure_symbol_column(self.new_ports)


    # ----- Discretionary helpers -----
    def select_product_discretionary_by_need(
        self,
        ports,
        ppm,
        currency: str,
        headroom_amount: float = 0.0,
    ) -> pd.Series | None:
        port_allo = ports.get_portfolio_asset_allocation_lookthrough(ppm)
        model_allo = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_allo.merge(model_allo, on="port_id", how="left")

        aa_cols = ["aa_cash","aa_fi","aa_le","aa_ge","aa_alt"]
        need = np.array([max(float(diffs[f"{c}_model"].iloc[0] - diffs[c].iloc[0]), 0.0) for c in aa_cols], dtype=float)
        if need.sum() <= self.eps_tiny:
            return None
        need = need / need.sum()

        cands = self.discretionary_allo_weight[self.discretionary_allo_weight["currency"] == currency].copy()
        if cands.empty:
            return None

        # filter universe by headroom amount
        if headroom_amount + self.eps >= self.min_mandate_amount:
            cands = cands[cands["product_type_desc"] == "Mandate"]
        else:
            cands = cands[cands["product_type_desc"] != "Mandate"]
        if cands.empty:
            return None

        def norm_row(row):
            v = np.array([row[c] for c in aa_cols], dtype=float)
            s = v.sum()
            return v / s if s > 0 else np.zeros_like(v)

        mat = np.vstack([norm_row(r) for _, r in cands.iterrows()])
        dists = np.linalg.norm(mat - need.reshape(1, -1), axis=1)
        min_dist = dists.min()
        winners = cands.iloc[np.where(np.isclose(dists, min_dist, atol=self.eps_tiny))[0]]
        return winners.sample(n=1, random_state=np.random.randint(0, 1_000_000)).iloc[0]

    def select_product_discretionary_by_style(
        self,
        ports,
        currency: str,
        headroom_amount: float = 0.0,
    ) -> pd.Series | None:
        if self.client_investment_style is None and ports.df_style is not None:
            self.client_investment_style = ports.df_style.iloc[0]["port_investment_style"]
        elif self.client_investment_style is None and (ports.df_style is None or ports.df_style.empty):
            return None

        cands = self.discretionary_allo_weight[self.discretionary_allo_weight["currency"] == currency].copy()
        if cands.empty:
            return None

        # filter universe by headroom amount
        if headroom_amount + self.eps >= self.min_mandate_amount:
            cands = cands[cands["product_type_desc"] == "Mandate"]
        else:
            cands = cands[cands["product_type_desc"] != "Mandate"]
        if cands.empty:
            return None

        style_discretionary_mapping = {
            "Bulletproof": ["KKPFITHB", "KKPMOP", "KKP CorePath Ultra Light"],
            "Conservative": ["KKPMOTHB", "KKPMOP", "KKP CorePath Light"],
            "Moderate Low Risk": ["KKPMOTHB", "KKPMOP", "KKP CorePath Light"],
            "Moderate High Risk": ["KKPBATHB", "KKPBAP", "KKP CorePath Balanced"],
            "High Risk": ["KKPBATHB", "KKPBAP", "KKP CorePath Balanced"],
            "Aggressive Growth": ["KKPAGTHB", "KKPAGP", "KKP CorePath Extra"],
            "Unwavering": ["KKPEQTHB", "KKPEQP", "KKP CorePath Extra"],
        }
        # Filter candidates by client style mapping
        if self.client_investment_style in style_discretionary_mapping:
            allowed_sharecodes = set(style_discretionary_mapping[self.client_investment_style])
            winners = cands[cands["src_sharecodes"].isin(allowed_sharecodes)]
        if winners.empty:
            return None
        return winners.iloc[0]

    def get_current_discretionary_weight(self, ports) -> float:
        discretionary_syms = set(self.discretionary_allo_weight["src_symbol"].astype(str).unique())
        if "symbol" in ports.df_out.columns:
            return float(ports.df_out.loc[ports.df_out["symbol"].astype(str).isin(discretionary_syms), "weight"].sum() or 0.0)
        return 0.0

    # ---------------------------
    # Sell phase
    # ---------------------------
    ## Comment: Can we delegate theses tasks to healthscore?
    def check_not_monitored(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        mask = (
            (comp["score_non_cover_global_stock"] < 0)
            | (comp["score_non_cover_local_stock"] < 0)
            | (comp["score_non_cover_mutual_fund"] < 0)
        )
        sub = comp.loc[mask].copy()
        sub["flag"] = "not_monitored_product"
        sub["expected_weight"] = 0.0
        return sub

    def check_issuer_risk(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        sub = comp[~comp["issure_risk_group"].isna()].copy()
        sub["flag"] = "issuer_risk"

        sum_weight = comp.groupby("issure_risk_group").agg({"weight": "sum"})
        sub = sub.merge(sum_weight, on="issure_risk_group", how="left", suffixes=('', '_issure_risk_group'))
        sub["expected_weight"] = (0.19 / sub["weight_issure_risk_group"]) * sub["weight"]
        return sub

    def check_bulk_risk(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        sub = comp[comp["is_bulk_risk"]].copy()
        sub["flag"] = "bulk_risk"
        sub["expected_weight"] = 0.19
        return sub

    def check_es_sell_list(self, ports, ppm, hs) -> pd.DataFrame:
        # [TEMP-DEBUG] Guard: if es_sell_list ref is missing, skip
        if not hasattr(self, "es_sell_list") or self.es_sell_list is None:
            logger = logging.getLogger(__name__)
            logger.warning("es_sell_list reference is None; skipping sell-list filter")
            return pd.DataFrame(columns=[
                "port_id","src_sharecodes","flag","expected_weight","value","weight"
            ])
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        comp = comp.merge(self.es_sell_list, left_on="src_sharecodes", right_on="symbol", how="left")
        sub = comp[~comp["symbol"].isna()].copy()
        sub["flag"] = "sell_list"
        sub["expected_weight"] = 0
        return sub

    def build_sell_recommendations(self, ports, ppm, hs) -> pd.DataFrame:

        bad_products = [
            self.check_not_monitored(ports, ppm, hs),
            self.check_issuer_risk(ports, ppm, hs),
            self.check_bulk_risk(ports, ppm, hs),
            # self.check_es_sell_list(ports, ppm, hs)
        ]
        bad_products = [p for p in bad_products if not p.empty]
        if not bad_products:
            return pd.DataFrame(columns=self.reco_cols)

        to_sell = pd.concat(bad_products, ignore_index=True)

        # aggregate by key_cols, combine flags, take min expected_weight
        to_sells = (
            to_sell.groupby(["port_id","value","weight"] + ports.prod_comp_keys, dropna=False, as_index=False)
            .agg(
                flag=("flag", lambda x: ", ".join(sorted(set(x)))),
                expected_weight=("expected_weight", "min"),
            )
        )
        to_sells["action"] = "sell"
        with np.errstate(divide="ignore", invalid="ignore"):
            to_sells["amount"] = -1.0 * ((to_sells["weight"] - to_sells["expected_weight"]) / to_sells["weight"]) * to_sells["value"]
        to_sells["amount"] = to_sells["amount"].fillna(0.0)

        to_sells = to_sells.loc[to_sells["amount"] != 0].copy()
        if to_sells.empty:
            return pd.DataFrame(columns=self.reco_cols)

        # Merge product mapping
        to_sells = to_sells.merge(
            ports.product_mapping,
            on=ports.prod_comp_keys,
            how="left",
        )

        # Debugging
        # _na_mask = to_sells[["product_type_desc", "src_sharecodes", "symbol", "asset_class_name"]].isna()
        # if _na_mask.any(axis=1).any():
        #     cols_dbg = ["port_id"] + list(ports.prod_comp_keys) + [
        #         "flag", "expected_weight", "weight", "value",
        #         "src_sharecodes", "symbol", "product_type_desc", "asset_class_name"
        #     ]
        #     logger = logging.getLogger(__name__)
        #     logger.warning(
        #         "Rows with NA after mapping (sample):\n%s",
        #         to_sells.loc[_na_mask.any(axis=1), cols_dbg].head(10)
        #     )
        #     try:
        #         # Show what exists in product_mapping for these src_sharecodes to compare join keys
        #         na_syms = to_sells.loc[_na_mask.any(axis=1), "src_sharecodes"].dropna().astype(str).unique().tolist()
        #         pm_cols = list(ports.prod_comp_keys) + ["symbol", "product_type_desc", "asset_class_name"]
        #         pm_slice = ports.product_mapping[ports.product_mapping["src_sharecodes"].astype(str).isin(na_syms)][pm_cols].drop_duplicates()
        #         logger.warning(
        #             "product_mapping candidates for NA symbols (sample):\n%s\njoin keys used: %s",
        #             pm_slice.head(20),
        #             ports.prod_comp_keys,
        #         )
        #     except Exception:
        #         pass


        # build row-by-row (to allow per-row min checks)
        recommendations = []
        for _, to_sell in to_sells.iterrows():
            to_sell = pd.DataFrame([to_sell])
            ft = to_sell["flag_tax_saving"].iloc[0] if "flag_tax_saving" in to_sell.columns else None
            is_tax_saving = bool(pd.notna(ft) and ft not in (False, "", 0))
            # removed noisy row-level debug prints
            # Handle NA values safely in boolean conditions
            ptype = to_sell["product_type_desc"].iloc[0]
            ptype_restricted = (not pd.isna(ptype)) and (ptype in ["Private Market", "Hedge Fund", "Structured Note"])
            src_code = to_sell["src_sharecodes"].iloc[0]
            src_whitelisted = (not pd.isna(src_code)) and (src_code.lower() in self.product_whitelist if self.product_whitelist else False)

            if ptype_restricted or is_tax_saving or src_whitelisted:
                continue

            # row-level minimum checks (BOTH must pass)
            # try:
            #     w_change = abs(float(to_sell["weight"].iloc[0] - to_sell["expected_weight"].iloc[0]))
            # except Exception:
            #     w_change = 0.0
            # a_abs = abs(float(to_sell["amount"].iloc[0] or 0.0))
            # if (w_change < self.min_weight_per_row) or (a_abs < self.min_amount_per_row):
            #     continue

            # 1-to-1 funding per sell row
            funding = self.build_cash_proxy_funding(ports, to_sell[[c for c in self.reco_cols if c not in ["transaction_no", "batch_no"]]], per_row=True)

            self.batch_seq += 1
            batch_no = self.batch_seq

            n = len(to_sell)
            start = self.txn_seq + 1
            self.txn_seq += n
            to_sell["batch_no"] = batch_no
            to_sell["transaction_no"] = list(range(start, start + n))

            n2 = len(funding)
            start2 = self.txn_seq + 1
            self.txn_seq += n2
            funding["batch_no"] = batch_no
            funding["transaction_no"] = list(range(start2, start2 + n2))

            cols = list(self.reco_cols)
            for dfp in (to_sell, funding):
                for c in cols:
                    if c not in dfp.columns:
                        dfp[c] = np.nan

            recommendations.append(pd.concat([to_sell[cols], funding[cols]], ignore_index=True))

        if not recommendations:
            return pd.DataFrame(columns=self.reco_cols)

        recommendations = pd.concat(recommendations, ignore_index=True)
        recommendations = recommendations[self.reco_cols]
        self.update_portfolio(ports, recommendations)
        return recommendations

    # ---------------------------
    # Buy phase
    # ---------------------------
    @staticmethod
    ## Comment: Can we delegate this task to portprop?
    def get_port_model_allocation_diff(ports, ppm) -> pd.DataFrame:
        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on=["port_id"], how="left")
        for col in ["aa_cash", "aa_fi", "aa_le", "aa_ge", "aa_alt"]:
            diffs[f"{col}_diff"] = diffs[col] - diffs[f"{col}_model"]
        return diffs

    def build_buy_recommendations(self, ports, ppm) -> pd.DataFrame:
        """
        Buy logic:
        1) ONE-OFF mandate trade up to portfolio mandate headroom (self.discretionary_acceptance - current mandate weight),
           limited by: available cash in the chosen currency and absolute underweight.
           -> single batch (1 security + its 1:1 cash-proxy funding).
        2) Then proceed with iterative ranked-product buys by currency (mandates excluded),
           step size = min(self.buy_step_weight_max, cash in that currency, absolute underweight),
           single-line cap self.single_line_cap_non_mandate (for non-mandates).
        """
        prod_reco_rank = self.get_product_recommendation_rank(ports)
        # Exclude blacklisted products from recommendations
        if self.product_blacklist:
            prod_reco_rank = prod_reco_rank[~prod_reco_rank["src_sharecodes"].str.lower().isin(self.product_blacklist)]

        asset_map = {
            "aa_cash_diff": "Cash and Cash Equivalent",
            "aa_fi_diff": "Fixed Income",
            "aa_le_diff": "Local Equity",
            "aa_ge_diff": "Global Equity",
            "aa_alt_diff": "Alternative",
        }

        buys = []
        total_value = float(ports.df_out["value"].sum())

        # --------- Phase 1: ONE-OFF mandate trade ----------
        diffs = self.get_port_model_allocation_diff(ports, ppm)
        under_cols = list(asset_map.keys())
        min_under_col = diffs[under_cols].idxmin(axis=1).iloc[0]
        min_under_val = float(diffs[min_under_col].iloc[0])     # negative if underweight

        cash_buckets = self.get_cash_proxy(ports)
        total_cash_w = float(cash_buckets["weight"].sum()) if not cash_buckets.empty else 0.0

        mandate_w_now = self.get_current_discretionary_weight(ports)
        mandate_headroom = max(0.0, self.discretionary_acceptance - mandate_w_now)

        if (mandate_headroom > self.eps and
            total_cash_w > self.eps and
            not cash_buckets.empty):

            for i in range(len(cash_buckets)):
                if i>=len(cash_buckets):
                    break
                fund_ccy = cash_buckets.iloc[i]["currency"]
                cash_w_ccy = float(cash_buckets.iloc[i]["weight"] or 0.0)

                add_w = min(mandate_headroom, cash_w_ccy)
                amount = add_w * total_value

                # mandate buy must meet BOTH thresholds
                if not (add_w >= self.min_weight_per_row and abs(amount) >= self.min_amount_per_row):
                    pass  # skip
                else:
                    chosen_md = self.select_product_discretionary_by_style(
                        ports=ports,
                        currency=fund_ccy,
                        headroom_amount=amount
                    )

                    if chosen_md is not None and add_w > self.eps:
                        sym = chosen_md["symbol"]
                        existing = ports.df_out[ports.df_out["symbol"] == sym][["weight"]]
                        cur_weight = float(existing["weight"].sum()) if not existing.empty else 0.0

                        chosen = pd.DataFrame([{
                            "port_id":        ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0],
                            "product_id":     chosen_md["product_id"],
                            "src_sharecodes": chosen_md["src_sharecodes"],
                            "desk":           chosen_md["desk"],
                            "port_type":      chosen_md["port_type"],
                            "currency":       chosen_md["currency"],
                            "value":          0.0,
                            "weight":         cur_weight,
                            "flag":           "discretionary_buy",
                            "expected_weight":cur_weight + add_w,
                            "action":         "buy",
                            "amount":         amount,
                        }])

                        # funding per row to keep 1-1 pair
                        cash_proxy = self.build_cash_proxy_funding(ports, chosen, per_row=True)

                        # assign IDs inline
                        self.batch_seq += 1
                        batch_no = self.batch_seq

                        n = len(chosen)
                        start = self.txn_seq + 1
                        self.txn_seq += n
                        chosen["batch_no"] = batch_no
                        chosen["transaction_no"] = list(range(start, start + n))

                        n2 = len(cash_proxy)
                        start2 = self.txn_seq + 1
                        self.txn_seq += n2
                        cash_proxy["batch_no"] = batch_no
                        cash_proxy["transaction_no"] = list(range(start2, start2 + n2))

                        cols = list(self.recommendations.columns)
                        for dfp in (chosen, cash_proxy):
                            for c in cols:
                                if c not in dfp.columns:
                                    dfp[c] = np.nan

                        trade = pd.concat([chosen[cols], cash_proxy[cols]], ignore_index=True)
                        buys.append(trade)
                        self.update_portfolio(ports, trade)

                        # refresh after the one-off trade
                        diffs = self.get_port_model_allocation_diff(ports, ppm)
                        cash_buckets = self.get_cash_proxy(ports)

        # --------- Phase 2: Ranked product loop (mandates excluded) ----------
        if cash_buckets is None or cash_buckets.empty:
            return pd.concat(buys, ignore_index=True) if buys else pd.DataFrame(columns=self.reco_cols)

        for _ in range(self.max_rebalance_loops):
            diffs = self.get_port_model_allocation_diff(ports, ppm)
            under_cols = list(asset_map.keys())
            min_under_col = diffs[under_cols].idxmin(axis=1).iloc[0]
            min_under_val = float(diffs[min_under_col].iloc[0])     # negative if underweight
            min_under_asset = asset_map[min_under_col]

            if min_under_val > -self.underweight_threshold:
                break  # underweight gate

            cash_buckets = self.get_cash_proxy(ports)
            total_cash_w = float(cash_buckets["weight"].sum()) if not cash_buckets.empty else 0.0

            if total_cash_w < self.buy_step_weight_max or cash_buckets.empty:
                break  # cash gate
            made_trade = False

            for _, cb in cash_buckets.iterrows():
                fund_ccy = cb["currency"]
                cash_w_ccy = float(cb["weight"] or 0.0)
                if cash_w_ccy <= self.eps:
                    continue

                step_w = float(min(self.buy_step_weight_max, cash_w_ccy))
                step_amount = step_w * total_value

                # BOTH thresholds must pass for buys
                if (step_w < self.min_weight_per_row) or (abs(step_amount) < self.min_amount_per_row):
                    continue
                if step_w <= self.eps:
                    continue

                df_out = ports.df_out[["port_id", "src_sharecodes", "currency", "value", "weight"]]

                cand = prod_reco_rank[
                    (prod_reco_rank["asset_class_name"] == min_under_asset) &
                    (prod_reco_rank["currency"] == fund_ccy)
                ]
                if "product_type_desc" in cand.columns:
                    cand = cand[cand["product_type_desc"] != "Mandate"]
                if self.client_classification == "UI":
                    cand = cand[~((cand["is_ui"] == "N") & (cand["asset_class_name"] == "Alternative"))]
                else:
                    cand = cand[~((cand["is_ui"] == "Y") & (cand["asset_class_name"] == "Alternative"))]

                candidates = (
                    cand.merge(df_out, on="src_sharecodes", how="left", suffixes=("", "_port"))
                        .fillna({"value": 0.0, "weight": 0.0})
                )

                if candidates.empty:
                    continue

                candidates["port_id"] = (
                    ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0]
                )
                candidates["flag"] = f"{min_under_asset.lower().replace(" ","_")}_buy"
                candidates["expected_weight"] = candidates["weight"] + step_w
                candidates["action"] = "buy"
                candidates["amount"] = step_amount
                candidates = candidates[candidates["expected_weight"] < self.single_line_cap_non_mandate]
                if candidates.empty:
                    continue

                best_rank = candidates["rank_product"].min()
                chosen = candidates[candidates["rank_product"] == best_rank].copy()
                if "currency" not in chosen.columns and "currency_port" in chosen.columns:
                    chosen["currency"] = chosen["currency_port"]
                if "currency" not in chosen.columns:
                    chosen["currency"] = fund_ccy

                # Select only 1 row (highest rank, random if tie)
                if len(chosen) > 1:
                    chosen = chosen.sample(n=1, random_state=np.random.randint(0, 1_000_000))
                chosen = chosen[[c for c in self.reco_cols if c not in ["transaction_no", "batch_no"]]]
                cash_proxy = self.build_cash_proxy_funding(ports, chosen, per_row=True)

                # Exclude already chosen products from candidates
                prod_reco_rank = prod_reco_rank[~prod_reco_rank["src_sharecodes"].isin(chosen["src_sharecodes"])]

                # assign IDs inline
                self.batch_seq += 1
                batch_no = self.batch_seq

                n = len(chosen)
                start = self.txn_seq + 1
                self.txn_seq += n
                chosen["batch_no"] = batch_no
                chosen["transaction_no"] = list(range(start, start + n))

                n2 = len(cash_proxy)
                start2 = self.txn_seq + 1
                self.txn_seq += n2
                cash_proxy["batch_no"] = batch_no
                cash_proxy["transaction_no"] = list(range(start2, start2 + n2))

                cols = list(self.recommendations.columns)
                for dfp in (chosen, cash_proxy):
                    for c in cols:
                        if c not in dfp.columns:
                            dfp[c] = np.nan

                trade = pd.concat([chosen[cols], cash_proxy[cols]], ignore_index=True)

                buys.append(trade)
                self.update_portfolio(ports, trade)
                made_trade = True
                break

            if not made_trade:
                break

        recommendations = pd.concat(buys, ignore_index=True) if buys else pd.DataFrame(columns=self.reco_cols)

        return recommendations

    # ---------------------------
    # Cash overweight -> Cash Proxy (same currency) — 1-to-1 pairing
    # ---------------------------
    def move_cash_overweight_to_proxy(self, ports, ppm) -> pd.DataFrame:
        cash_overweight = (
            ((ports.df_out["product_type_desc"] == "Cash") | (ports.df_out["asset_class_name"] == "Cash and Cash Equivalent"))
            & (ports.df_out["symbol"] != "CASH PROXY")
        )

        cash_rows = ports.df_out[cash_overweight].copy()
        if cash_rows.empty:
            return pd.DataFrame(columns=self.reco_cols)

        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on="port_id", how="left")

        w_cash_now = float(cash_rows["weight"].sum() or 0.0)
        w_cash_model = float(diffs["aa_cash_model"].iloc[0])
        w_over = max(0.0, w_cash_now - w_cash_model)
        if w_over <= self.eps:
            return pd.DataFrame(columns=self.reco_cols)

        total_value = float(ports.df_out["value"].sum())
        if w_cash_now <= self.eps_tiny:
            return pd.DataFrame(columns=self.reco_cols)

        cash_map = ports.product_mapping[ports.product_mapping["symbol"] == "CASH PROXY"][
            ports.prod_comp_keys + ["product_display_name", "product_type_desc", "asset_class_name"]
        ]

        pairs = []
        for _, r in cash_rows.iterrows():
            share = float(r["weight"]) / w_cash_now
            move_w = w_over * share
            amt_sell = - move_w * total_value

            # SELL leg must meet BOTH thresholds
            if (move_w < self.min_weight_per_row) or (abs(amt_sell) < self.min_amount_per_row):
                continue
            if abs(amt_sell) <= self.eps:
                continue

            sell_row = pd.DataFrame([{
                "port_id":        r["port_id"],
                "product_id":     r["product_id"],
                "src_sharecodes": r["src_sharecodes"],
                "desk":           r["desk"],
                "port_type":      r["port_type"],
                "currency":       r.get("currency", np.nan),
                "product_display_name": r.get("product_display_name", np.nan),
                "product_type_desc":    r.get("product_type_desc", np.nan),
                "asset_class_name":     r.get("asset_class_name", np.nan),
                "value":          r.get("value", np.nan),
                "weight":         r.get("weight", np.nan),
                "flag":           "cash_overweight",
                "expected_weight": (r.get("weight", np.nan) - move_w) if pd.notna(r.get("weight", np.nan)) else np.nan,
                "action":         "sell",
                "amount":         amt_sell,
            }])

            # funding buy (CASH PROXY) 1:1 for this sell row
            funding_buy = sell_row[["port_id","currency","amount"]].copy()
            funding_buy["amount"] = -funding_buy["amount"]
            funding_buy = funding_buy.merge(cash_map, on="currency", how="left")
            funding_buy["value"] = np.nan
            funding_buy["weight"] = np.nan
            funding_buy["flag"] = "cash_proxy_funding"
            funding_buy["expected_weight"] = np.nan
            funding_buy["action"] = "funding"
            funding_buy = funding_buy[[c for c in self.reco_cols if c not in ["transaction_no", "batch_no"]]]

            # assign IDs inline
            self.batch_seq += 1
            batch_no = self.batch_seq

            n = len(sell_row)
            start = self.txn_seq + 1
            self.txn_seq += n
            sell_row["batch_no"] = batch_no
            sell_row["transaction_no"] = list(range(start, start + n))

            n2 = len(funding_buy)
            start2 = self.txn_seq + 1
            self.txn_seq += n2
            funding_buy["batch_no"] = batch_no
            funding_buy["transaction_no"] = list(range(start2, start2 + n2))

            cols = list(self.recommendations.columns)
            for dfp in (sell_row, funding_buy):
                for c in cols:
                    if c not in dfp.columns:
                        dfp[c] = np.nan

            pairs.append(pd.concat([sell_row[cols], funding_buy[cols]], ignore_index=True))

        if not pairs:
            return pd.DataFrame(columns=self.recommendations.columns)

        trade = pd.concat(pairs, ignore_index=True)
        self.update_portfolio(ports, trade)
        return trade

    def add_new_money(self, ports) -> pd.DataFrame:
        """
        Add new money by mapping to CASH PROXY (THB).
        """
        if self.new_money is None or self.new_money <= self.eps:
            return pd.DataFrame(columns=self.reco_cols)
        cash_proxy = ports.product_mapping[
            (ports.product_mapping["symbol"] == "CASH PROXY") &
            (ports.product_mapping["currency"] == "THB")
        ].copy()
        if cash_proxy.empty:
            # Could not find cash proxy for THB
            return pd.DataFrame(columns=self.reco_cols)
        # only pick the first match if multiple
        proxy = cash_proxy.iloc[0]
        reco = pd.DataFrame([{
            "transaction_no": self.txn_seq + 1,
            "batch_no": self.batch_seq + 1,
            "port_id": ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0],
            "product_id": proxy["product_id"],
            "src_sharecodes": proxy["src_sharecodes"],
            "desk": proxy["desk"],
            "port_type": proxy["port_type"],
            "currency": proxy["currency"],
            "product_display_name": proxy.get("product_display_name", np.nan),
            "product_type_desc": proxy.get("product_type_desc", np.nan),
            "asset_class_name": proxy.get("asset_class_name", np.nan),
            "value": np.nan,
            "weight": np.nan,
            "flag": "new_money",
            "expected_weight": np.nan,
            "action": "funding",
            "amount": float(self.new_money),
        }])
        self.txn_seq += 1
        self.batch_seq += 1
        return reco[self.reco_cols]

    def convert_cash_proxy_currency(self, ports) -> pd.DataFrame:
        # Find all CASH PROXY positions in non-THB and non-USD currencies
        cash_proxy_rows = ports.df_out[
            (ports.df_out["symbol"] == "CASH PROXY") &
            (~ports.df_out["currency"].isin(["THB", "USD"]))
        ].copy()
        if cash_proxy_rows.empty:
            return pd.DataFrame(columns=self.reco_cols)

        # Find USD CASH PROXY mapping
        usd_cash_proxy = ports.product_mapping[
            (ports.product_mapping["symbol"] == "CASH PROXY") &
            (ports.product_mapping["currency"] == "USD")
        ]
        if usd_cash_proxy.empty:
            return pd.DataFrame(columns=self.reco_cols)
        usd_proxy = usd_cash_proxy.iloc[0]

        recommendations = []
        for _, row in cash_proxy_rows.iterrows():
            # Remove the non-THB/USD cash proxy (sell)
            sell_row = pd.DataFrame([{
                "port_id": row["port_id"],
                "product_id": row["product_id"],
                "src_sharecodes": row["src_sharecodes"],
                "desk": row["desk"],
                "port_type": row["port_type"],
                "currency": row["currency"],
                "product_display_name": row.get("product_display_name", np.nan),
                "product_type_desc": row.get("product_type_desc", np.nan),
                "asset_class_name": row.get("asset_class_name", np.nan),
                "value": row.get("value", np.nan),
                "weight": row.get("weight", np.nan),
                "flag": "convert_currency",
                "expected_weight": 0.0,
                "action": "sell",
                "amount": -abs(float(row.get("value", 0.0))),
            }])

            # Add USD cash proxy (buy)
            buy_row = pd.DataFrame([{
                "port_id": row["port_id"],
                "product_id": usd_proxy["product_id"],
                "src_sharecodes": usd_proxy["src_sharecodes"],
                "desk": usd_proxy["desk"],
                "port_type": usd_proxy["port_type"],
                "currency": "USD",
                "product_display_name": usd_proxy.get("product_display_name", np.nan),
                "product_type_desc": usd_proxy.get("product_type_desc", np.nan),
                "asset_class_name": usd_proxy.get("asset_class_name", np.nan),
                "value": np.nan,
                "weight": np.nan,
                "flag": "convert_currency",
                "expected_weight": np.nan,
                "action": "funding",
                "amount": abs(float(row.get("value", 0.0))),
            }])

            self.batch_seq += 1
            batch_no = self.batch_seq

            n_sell = len(sell_row)
            start_sell = self.txn_seq + 1
            self.txn_seq += n_sell
            sell_row["batch_no"] = batch_no
            sell_row["transaction_no"] = list(range(start_sell, start_sell + n_sell))

            n_buy = len(buy_row)
            start_buy = self.txn_seq + 1
            self.txn_seq += n_buy
            buy_row["batch_no"] = batch_no
            buy_row["transaction_no"] = list(range(start_buy, start_buy + n_buy))

            cols = list(self.reco_cols)
            for dfp in (sell_row, buy_row):
                for c in cols:
                    if c not in dfp.columns:
                        dfp[c] = np.nan

            recommendations.append(pd.concat([sell_row[cols], buy_row[cols]], ignore_index=True))

        if not recommendations:
            return pd.DataFrame(columns=self.reco_cols)

        trade = pd.concat(recommendations, ignore_index=True)
        self.update_portfolio(ports, trade)
        return trade

    @staticmethod
    def readable_flag(flag: str) -> str:
        if not flag or not isinstance(flag, str):
            return ""

        flags = [f.strip().lower() for f in flag.split(",") if f.strip()]
        unique_flags = sorted(set(flags))

        # --- SELL REASONS ---
        sell_reasons = {
            "bulk_risk": "overconcentration in a single product",
            "issuer_risk": "high exposure to a single issuer",
            "not_monitored_product": "product not actively monitored by research/investment solutions",
            "sell_list": "internal sell recommendation",
        }

        # --- BUY REASONS ---
        buy_reasons = {
            "alternative_buy": "alternative",
            "fixed_income_buy": "fixed income",
            "local_equity_buy": "local equity",
            "global_equity_buy": "global equity",
            "discretionary_buy": "discretionary product aligned with the investment style",
        }

        # --- OTHER ACTIONS ---
        other_actions = {
            "cash_overweight": "Reduce cash holdings above the model target.",
            "cash_proxy_funding": "Fund transactions through the cash proxy position.",
            "new_money": "Add new money to the portfolio.",
            "convert_currency": "Convert foreign cash into USD.",
        }

        # --- Detect category ---
        is_buy = any(f.endswith("_buy") for f in unique_flags)
        is_risk = any(f in sell_reasons for f in unique_flags)
        is_cash = any(f.startswith("cash_") for f in unique_flags)
        is_new_money = "new_money" in unique_flags
        is_convert_ccy = "convert_currency" in unique_flags

        # --- Sell Message ---
        if is_risk:
            reasons = [sell_reasons[f] for f in unique_flags if f in sell_reasons]
            joined = "; ".join(reasons)
            return f"Reduce or exit to manage {joined}."

        # --- Buy Message ---
        if is_buy:
            buys = [buy_reasons[f] for f in unique_flags if f in buy_reasons]
            buys_text = ", ".join(buys)
            return f"Increase allocation to {buys_text} to align the portfolio with the model allocation."

        # --- Cash Management ---
        if is_cash:
            actions = [other_actions[f] for f in unique_flags if f in other_actions]
            return " ".join(actions)

        # --- New Money ---
        if is_new_money:
            return other_actions["new_money"]

        # --- Currency Conversion ---
        if is_convert_ccy:
            return other_actions["convert_currency"]

        # --- Default Fallback ---
        return f"Rebalance portfolio according to internal investment guidelines ({flag})."

    # ---------------------------
    # Orchestrator
    # ---------------------------
    def rebalance(self, ports: Portfolios, ppm, hs, reset_state: bool = True, refresh_refs: bool = False) -> Tuple[Portfolios, pd.DataFrame]:
        try:
            logger = logging.getLogger(__name__)
            logger.debug("[rebalance] start | new_money=%s", self.new_money)
            # optionally refresh DB-backed reference tables
            # if refresh_refs:
            #     self.refresh_reference_data()

            # clear ephemeral state so we don't carry over from previous runs
            if reset_state:
                self.reset_state()

            # Create a new ports instance with copied portfolio data
            # Deep copy is unnecessary - we only need to modify the portfolio, not reference tables
            self.new_ports = copy.copy(ports)
            # Copy only the mutable portfolio data, not the reference tables
            self.new_ports.df_out = ports.df_out.copy()
            self.new_ports.df_style = ports.df_style.copy()
            self.new_ports.port_ids = ports.port_ids.copy()
            self.new_ports.port_id_mapping = ports.port_id_mapping.copy()

            health_score_before = self.new_ports.get_portfolio_health_score(ppm, hs)[0]["health_score"].values[0]

            new_money = self.add_new_money(self.new_ports)
            if not new_money.empty:
                self.recommendations = pd.concat([self.recommendations, new_money], ignore_index=True)
                self.update_portfolio(self.new_ports, new_money)
            logger.debug("[rebalance] new_money rows=%s", 0 if new_money is None else len(new_money))

            sells = self.build_sell_recommendations(self.new_ports, ppm, hs)
            if not sells.empty:
                self.recommendations = pd.concat([self.recommendations, sells], ignore_index=True)
            logger.debug("[rebalance] sells rows=%s", 0 if sells is None else len(sells))

            cash_shift = self.move_cash_overweight_to_proxy(self.new_ports, ppm)
            if not cash_shift.empty:
                self.recommendations = pd.concat([self.recommendations, cash_shift], ignore_index=True)
            logger.debug("[rebalance] cash_shift rows=%s", 0 if cash_shift is None else len(cash_shift))

            convert_ccy = self.convert_cash_proxy_currency(self.new_ports)
            if not convert_ccy.empty:
                self.recommendations = pd.concat([self.recommendations, convert_ccy], ignore_index=True)
            logger.debug("[rebalance] convert_ccy rows=%s", 0 if convert_ccy is None else len(convert_ccy))

            buys = self.build_buy_recommendations(self.new_ports, ppm)
            if not buys.empty:
                self.recommendations = pd.concat([self.recommendations, buys], ignore_index=True)
            logger.debug("[rebalance] buys rows=%s", 0 if buys is None else len(buys))

            logger.debug("[rebalance] final recommendations rows=%s", len(self.recommendations))

            # Merge product mapping for final output
            self.recommendations = self.recommendations.merge(
                self.new_ports.product_mapping,
                on=self.new_ports.prod_comp_keys,
                how="left",
                suffixes=("_reco", "")
            )
            self.recommendations = self.recommendations[self.reco_cols]

            health_score_after = self.new_ports.get_portfolio_health_score(ppm, hs)[0]["health_score"].values[0]

            # Do not rebalance if health score decreased
            if health_score_after < health_score_before:
                logger.warning(
                    "[rebalance] health score decreased after rebalance: before=%.2f after=%.2f",
                    health_score_before, health_score_after
                )
                return ports, pd.DataFrame(columns=self.reco_cols)

            # Add human-readable flag message
            self.recommendations["flag_msg"] = self.recommendations["flag"].apply(type(self).readable_flag)

            try:
                logger.debug("[rebalance] final df_out rows=%s", len(self.new_ports.df_out))
            except Exception:
                logger.debug("[rebalance] final df_out unavailable")
            return self.new_ports, self.recommendations

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.exception("[rebalance][exception] %s", e)
            # if not self.recommendations.empty:
            #     return self.recommendations

            return self.new_ports, pd.DataFrame(columns=self.reco_cols+["flag_msg"])

        # ---------------------------
        # For Debug
        # ---------------------------
        # # optionally refresh DB-backed reference tables
        # if refresh_refs:
        #     self.refresh_reference_data()

        # # clear ephemeral state so we don't carry over from previous runs
        # if reset_state:
        #     self.reset_state()

        # self.new_ports = copy.deepcopy(ports)

        # # --- Add new money (as cash proxy THB) if any ---
        # new_money = self.add_new_money(self.new_ports)
        # if not new_money.empty:
        #     self.recommendations = pd.concat([self.recommendations, new_money], ignore_index=True)
        #     self.update_portfolio(self.new_ports, new_money)

        # sells = self.build_sell_recommendations(self.new_ports, ppm, hs)
        # if not sells.empty:
        #     self.recommendations = pd.concat([self.recommendations, sells], ignore_index=True)

        # cash_shift = self.move_cash_overweight_to_proxy(self.new_ports, ppm)
        # if not cash_shift.empty:
        #     self.recommendations = pd.concat([self.recommendations, cash_shift], ignore_index=True)

        # convert_ccy = self.convert_cash_proxy_currency(self.new_ports)
        # if not convert_ccy.empty:
        #     self.recommendations = pd.concat([self.recommendations, convert_ccy], ignore_index=True)

        # buys = self.build_buy_recommendations(self.new_ports, ppm)
        # if not buys.empty:
        #     self.recommendations = pd.concat([self.recommendations, buys], ignore_index=True)

        # return self.new_ports, self.recommendations
