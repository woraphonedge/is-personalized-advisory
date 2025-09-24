import pandas as pd
import numpy as np
from utils import *


class Rebalancer:
    """
    A portfolio rebalancer that (1) sells to remove/trim risky or non-monitored holdings and
    (2) buys model-aligned products using ranked recommendations until the portfolio is within
    tolerance vs model weights.

    Expected dependencies provided by caller:
      - ports: Portfolios object (with df_out, df_style, port_ids, product_mapping loaded)
      - ppm:   PortpropMatrices object
      - hs:    HealthScore object
    """

    def __init__(
        self,
        as_of_date: str | None = None,
        customer_id: int | None = None,
        client_investment_style: str | None = None,
        client_classification: str | None = None,
        new_money: float = 0.0,

        # -------- constraints --------
        private_percent: float = 0.0,
        cash_percent: float | None = None,
        offshore_percent: float | None = None,
        product_restriction: list[str] | None = None,
        discretionary_acceptance: float | None = None,
        
        # -------- params --------
        eps: float = 1e-8,
        eps_tiny: float = 1e-12,
        underweight_threshold: float = 0.05,
        buy_step_weight_max: float = 0.10,
        single_line_cap_non_mandate: float = 0.20,
        max_rebalance_loops: int = 50,
        min_amount_per_row: float = 1_000.0,
        min_weight_per_row: float = 0.0,
        min_mandate_amount: float = 1_000_000.0,
    ) -> None:
        
        self.as_of_date = (
            pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d") if as_of_date is None else as_of_date

        self.customer_id = customer_id
        self.client_investment_style = client_investment_style
        self.client_classification = client_classification
        self.new_money = new_money

        self.private_percent = private_percent
        self.cash_percent = cash_percent
        self.offshore_percent = offshore_percent
        self.product_restriction = product_restriction or []
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
        self._txn_seq = 0
        self._batch_seq = 0

        # keep a clean recommendations frame template for easy resets
        self._reco_cols = [
            "TRANSACTION_NO", "BATCH_NO", "PORT_ID", "PRODUCT_ID", "SRC_SHARECODES", "DESK",
            "PORT_TYPE", "CURRENCY", "VALUE", "WEIGHT", "FLAG", "EXPECTED_WEIGHT", "ACTION", "AMOUNT",
        ]
        self.recommendations = pd.DataFrame(columns=self._reco_cols)

        # preload raw tables
        self.es_sell_list: pd.DataFrame | None = None
        self.product_recommendation_rank_raw: pd.DataFrame | None = None
        self.mandate_allocation: pd.DataFrame | None = None
        self.load_es_sell_list()
        self.load_product_recommendation_rank_raw()
        self.load_mandate_candidates()

    # ---------- state mgmt ----------
    def reset_state(self) -> None:
        """Clear per-run state so the same instance can be reused safely."""
        self._txn_seq = 0
        self._batch_seq = 0
        self.recommendations = pd.DataFrame(columns=self._reco_cols)

    def refresh_reference_data(self) -> None:
        """Reload cached reference tables (if you expect DB to have changed)."""
        self.load_es_sell_list()
        self.load_product_recommendation_rank_raw()
        self.mandate_allocation = None
        self.load_mandate_candidates()

    # ---------- cached loaders ----------
    def load_es_sell_list(self) -> None:
        self.es_sell_list = read_sql("select * from user.kwm.personalized_advisory_es_sell_list where upper(RECOMMENDATION) like '%SELL%' or upper(RECOMMENDATION) like '%SWITCH%'")

    def load_product_recommendation_rank_raw(self) -> None:
        self.product_recommendation_rank_raw = read_sql(
            "select * from user.kwm.personalized_advisory_recommendation_rank"
        )

    def load_mandate_candidates(self) -> pd.DataFrame:
        """Preload (and cache) mandate universe once; return cached frame."""
        if self.mandate_allocation is None:
            sql = "select * from user.kwm.personalized_advisory_asset_allocation_weight"
            df = read_sql(sql)
            for c in ["AA_CASH","AA_FI","AA_LE","AA_GE","AA_ALT"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            self.mandate_allocation = df
        return self.mandate_allocation

    def get_product_recommendation_rank(self, ports) -> pd.DataFrame:
        """Merge the (preloaded) recommendation rank with current product mapping."""
        if self.product_recommendation_rank_raw is None:
            self.load_product_recommendation_rank_raw()
        df = self.product_recommendation_rank_raw.merge(
            ports.product_mapping,
            on=["SRC_SHARECODES", "DESK", "CURRENCY"],
            how="left",
            suffixes=("", "_MAPPING")
        )
        return df

    # ---------------------------
    # Cash Proxy funding (unified)
    # ---------------------------
    def build_cash_proxy_funding(self, ports, df: pd.DataFrame, per_row: bool = True) -> pd.DataFrame:
        """
        Unified builder for CASH PROXY funding.
        - per_row=True  -> 1:1 funding per input row (use for SELL / cash overweight)
        - per_row=False -> aggregate by currency then fund (usable for BUY)
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY",
                "VALUE","WEIGHT","FLAG","EXPECTED_WEIGHT","ACTION","AMOUNT"
            ])
        if ports.product_mapping is None:
            raise RuntimeError("Product mapping must be loaded before mapping cash proxy.")

        cash_map = ports.product_mapping[ports.product_mapping["SYMBOL"] == "CASH PROXY"][
            ["PRODUCT_ID", "SRC_SHARECODES", "DESK", "PORT_TYPE", "CURRENCY"]
        ].copy()

        if per_row:
            base = df[["PORT_ID","CURRENCY","AMOUNT"]].copy()
        else:
            base = df.groupby(["PORT_ID","CURRENCY"], dropna=False, as_index=False)["AMOUNT"].sum()

        out = base.merge(cash_map, on="CURRENCY", how="left")
        out["AMOUNT"] = -out["AMOUNT"].astype(float)
        out["VALUE"] = np.nan
        out["WEIGHT"] = np.nan
        out["FLAG"] = "cash_proxy_funding"  # hardcoded
        out["EXPECTED_WEIGHT"] = np.nan
        out["ACTION"] = "funding"

        cols = ["PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY",
                "VALUE","WEIGHT","FLAG","EXPECTED_WEIGHT","ACTION","AMOUNT"]
        out = out.loc[:, ~out.columns.duplicated()]
        return out[cols]

    # ---------------------------
    # Portfolio updater
    # ---------------------------
    def update_portfolio(self, ports: "Portfolios", trade_rows: pd.DataFrame) -> None:
        trade_as_positions = (
            trade_rows[["PORT_ID","AMOUNT"] + ports.prod_comp_keys]
            .rename(columns={"AMOUNT": "VALUE"})
            .copy()
        )

        base_positions = ports.df_out[["PORT_ID","VALUE"] + ports.prod_comp_keys].copy()
        df_products = pd.concat([base_positions, trade_as_positions], ignore_index=True, sort=False)

        df_products = (
            df_products.groupby(
                ["PORT_ID"] + ports.prod_comp_keys,
                dropna=False,
                as_index=False,
            )["VALUE"]
            .sum()
        )
        product_mapping = ports.product_mapping
        df_out = df_products.merge(
            product_mapping,
            on=ports.prod_comp_keys,
            how="left",
        )

        df_out = df_out[df_out["VALUE"] != 0]
        ports.set_portfolio(df_out, ports.df_style, ports.port_ids, ports.port_id_mapping)

    # ----- Mandate helpers -----
    def select_best_mandate_for_currency(
        self,
        ports,
        ppm,
        mandate_df: pd.DataFrame,
        currency: str,
        mandate_headroom_w: float = 0.0,   # headroom as WEIGHT (0–1)
        total_value: float | None = None,  # to compare against min_mandate_amount
    ) -> pd.Series | None:
        # convert headroom weight to currency amount
        tv = float(total_value or 0.0)
        headroom_amt = float(mandate_headroom_w) * tv

        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)[0]
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on="PORT_ID", how="left")

        aa_cols = ["AA_CASH","AA_FI","AA_LE","AA_GE","AA_ALT"]
        need = np.array([max(float(diffs[f"{c}_MODEL"].iloc[0] - diffs[c].iloc[0]), 0.0) for c in aa_cols], dtype=float)
        if need.sum() <= self.eps_tiny:
            return None
        need = need / need.sum()

        cands = mandate_df[mandate_df["CURRENCY"] == currency].copy()
        if cands.empty:
            return None

        # filter universe by headroom amount
        if "PRODUCT_TYPE_DESC" in cands.columns:
            if headroom_amt + self.eps >= self.min_mandate_amount:
                cands = cands[cands["PRODUCT_TYPE_DESC"] == "Mandate"]
            else:
                cands = cands[cands["PRODUCT_TYPE_DESC"] != "Mandate"]
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

    def get_cash_buckets(self, ports) -> pd.DataFrame:
        cash = ports.df_out[ports.df_out["SYMBOL"] == "CASH PROXY"]
        if cash.empty:
            return cash
        out = (cash.groupby(["PORT_ID","CURRENCY"], as_index=False)
                    .agg(WEIGHT=("WEIGHT","sum"), VALUE=("VALUE","sum"))).sort_values("WEIGHT", ascending=False)
        return out

    def get_mandate_weight(self, ports) -> float:
        if "PRODUCT_TYPE_DESC" in ports.df_out.columns:
            return float(ports.df_out.loc[ports.df_out["PRODUCT_TYPE_DESC"] == "Mandate", "WEIGHT"].sum() or 0.0)
        mandates = self.load_mandate_candidates()
        mandate_syms = set(mandates["SYMBOL"].astype(str).unique())
        if "SYMBOL" in ports.df_out.columns:
            return float(ports.df_out.loc[ports.df_out["SYMBOL"].astype(str).isin(mandate_syms), "WEIGHT"].sum() or 0.0)
        return 0.0

    # ---------------------------
    # Sell phase
    # ---------------------------
    def check_not_monitored(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        mask = (
            (comp["SCORE_NON_COVER_GLOBAL_STOCK"] < 0)
            | (comp["SCORE_NON_COVER_LOCAL_STOCK"] < 0)
            | (comp["SCORE_NON_COVER_MUTUAL_FUND"] < 0)
        )
        sub = comp.loc[mask].copy()
        sub["FLAG"] = "not_monitored_product"
        sub["EXPECTED_WEIGHT"] = 0.0
        return sub

    def check_issuer_risk(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        sub = comp[~comp["ISSURE_RISK_GROUP"].isna()].copy()
        sub["FLAG"] = "issuer_risk"

        sum_weight = comp.groupby('ISSURE_RISK_GROUP').agg({'WEIGHT': 'sum'})
        sub = sub.merge(sum_weight, on='ISSURE_RISK_GROUP', how='left', suffixes=('', '_ISSURE_RISK_GROUP'))
        sub['EXPECTED_WEIGHT'] = (0.19 / sub['WEIGHT_ISSURE_RISK_GROUP']) * sub['WEIGHT']
        return sub

    def check_bulk_risk(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        sub = comp[comp["IS_BULK_RISK"]].copy()
        sub["FLAG"] = "bulk_risk"
        sub["EXPECTED_WEIGHT"] = 0.19
        return sub

    def check_es_sell_list(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        comp = comp.merge(self.es_sell_list, left_on="SRC_SHARECODES", right_on="SYMBOL", how="left")
        sub = comp[~comp["SYMBOL"].isna()].copy()
        sub["FLAG"] = "sell_list"
        sub["EXPECTED_WEIGHT"] = 0
        return sub

    def build_sell_recommendations(self, ports, ppm, hs) -> pd.DataFrame:
        parts = [
            self.check_not_monitored(ports, ppm, hs),
            self.check_issuer_risk(ports, ppm, hs),
            self.check_bulk_risk(ports, ppm, hs),
            self.check_es_sell_list(ports, ppm, hs)
        ]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(columns=self.recommendations.columns)

        rec = pd.concat(parts, ignore_index=True)

        key_cols = [
            "PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY","VALUE","WEIGHT",
        ]
        rec = (
            rec.groupby(key_cols, dropna=False, as_index=False)
              .agg(FLAG=("FLAG", lambda x: ", ".join(sorted(set(x)))),
                   EXPECTED_WEIGHT=("EXPECTED_WEIGHT", "min"))
        )

        rec["ACTION"] = "sell"
        with np.errstate(divide="ignore", invalid="ignore"):
            rec["AMOUNT"] = -1.0 * ((rec["WEIGHT"] - rec["EXPECTED_WEIGHT"]) / rec["WEIGHT"]) * rec["VALUE"]
        rec["AMOUNT"] = rec["AMOUNT"].fillna(0.0)

        chosen = rec.loc[rec["AMOUNT"] != 0].copy()
        if chosen.empty:
            return pd.DataFrame(columns=self.recommendations.columns)

        # build row-by-row (to allow per-row min checks)
        trades = []
        for _, row in chosen.iterrows():
            row_df = pd.DataFrame([row[
                ["PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY",
                 "VALUE","WEIGHT","FLAG","EXPECTED_WEIGHT","ACTION","AMOUNT"]
            ]])

            # row-level minimum checks (BOTH must pass)
            try:
                w_change = abs(float(row_df["WEIGHT"].iloc[0] - row_df["EXPECTED_WEIGHT"].iloc[0]))
            except Exception:
                w_change = 0.0
            a_abs = abs(float(row_df["AMOUNT"].iloc[0] or 0.0))
            if (w_change < self.min_weight_per_row) or (a_abs < self.min_amount_per_row):
                continue

            # 1-to-1 funding per sell row
            funding_df = self.build_cash_proxy_funding(ports, row_df, per_row=True)

            # assign IDs inline
            self._batch_seq += 1
            batch_no = self._batch_seq

            n = len(row_df)
            start = self._txn_seq + 1
            self._txn_seq += n
            row_df["BATCH_NO"] = batch_no
            row_df["TRANSACTION_NO"] = list(range(start, start + n))

            n2 = len(funding_df)
            start2 = self._txn_seq + 1
            self._txn_seq += n2
            funding_df["BATCH_NO"] = batch_no
            funding_df["TRANSACTION_NO"] = list(range(start2, start2 + n2))

            cols = list(self.recommendations.columns)
            for dfp in (row_df, funding_df):
                for c in cols:
                    if c not in dfp.columns:
                        dfp[c] = np.nan

            trades.append(pd.concat([row_df[cols], funding_df[cols]], ignore_index=True))

        if not trades:
            return pd.DataFrame(columns=self.recommendations.columns)
        
        trade = pd.concat(trades, ignore_index=True)
        self.update_portfolio(ports, trade)
        return trade

    # ---------------------------
    # Buy phase
    # ---------------------------
    @staticmethod
    def get_port_model_allocation_diff(ports, ppm) -> pd.DataFrame:
        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)[0]
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        out = port_alloc.merge(model_alloc, on=["PORT_ID"], how="left")
        for col in ["AA_CASH", "AA_FI", "AA_LE", "AA_GE", "AA_ALT"]:
            out[f"{col}_DIFF"] = out[col] - out[f"{col}_MODEL"]
        return out

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
        product_recommendation_rank = self.get_product_recommendation_rank(ports)
        mandates = self.load_mandate_candidates()

        asset_map = {
            "AA_CASH_DIFF": "Cash and Cash Equivalent",
            "AA_FI_DIFF": "Fixed Income",
            "AA_LE_DIFF": "Local Equity",
            "AA_GE_DIFF": "Global Equity",
            "AA_ALT_DIFF": "Alternative",
        }
    
        buys = []
        total_value = float(ports.df_out["VALUE"].sum())

        # --------- Phase 1: ONE-OFF mandate trade ----------
        diffs = self.get_port_model_allocation_diff(ports, ppm)
        under_cols = list(asset_map.keys())
        min_col = diffs[under_cols].idxmin(axis=1).iloc[0]
        min_val = float(diffs[min_col].iloc[0])     # negative if underweight
        abs_under = abs(min_val)

        cash_buckets = self.get_cash_buckets(ports)
        total_cash_w = float(cash_buckets["WEIGHT"].sum()) if not cash_buckets.empty else 0.0

        mandate_w_now = self.get_mandate_weight(ports)
        mandate_headroom = max(0.0, self.discretionary_acceptance - mandate_w_now)

        if (mandate_headroom > self.eps and
            abs_under > self.eps and
            total_cash_w > self.eps and
            not cash_buckets.empty):

            for i in range(len(cash_buckets)):
                if i>=len(cash_buckets):
                    break
                fund_ccy = cash_buckets.iloc[i]["CURRENCY"]
                cash_w_ccy = float(cash_buckets.iloc[i]["WEIGHT"] or 0.0)
                
                add_w = min(mandate_headroom, cash_w_ccy, abs_under)
                amt = add_w * total_value

                # mandate buy must meet BOTH thresholds
                if not (add_w >= self.min_weight_per_row and abs(amt) >= self.min_amount_per_row):
                    pass  # skip
                else:
                    chosen_md = self.select_best_mandate_for_currency(
                        ports=ports,
                        ppm=ppm,
                        mandate_df=mandates,
                        currency=fund_ccy,
                        mandate_headroom_w=mandate_headroom,  # universe switch uses headroom, not step
                        total_value=total_value,
                    )

                    if chosen_md is not None and add_w > self.eps:
                        sym = chosen_md["SYMBOL"]
                        existing = ports.df_out[ports.df_out["SYMBOL"] == sym][["WEIGHT"]]
                        cur_weight = float(existing["WEIGHT"].sum()) if not existing.empty else 0.0

                        chosen = pd.DataFrame([{
                            "PORT_ID":        ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0],
                            "PRODUCT_ID":     chosen_md["PRODUCT_ID"],
                            "SRC_SHARECODES": chosen_md["SRC_SHARECODES"],
                            "DESK":           chosen_md["DESK"],
                            "PORT_TYPE":      chosen_md["PORT_TYPE"],
                            "CURRENCY":       chosen_md["CURRENCY"],
                            "VALUE":          0.0,
                            "WEIGHT":         cur_weight,
                            "FLAG":           "discretionary_buy",  # hardcoded
                            "EXPECTED_WEIGHT":cur_weight + add_w,
                            "ACTION":         "buy",
                            "AMOUNT":         amt,
                        }])

                        # funding per row to keep 1-1 pair
                        cash_proxy = self.build_cash_proxy_funding(ports, chosen, per_row=True)

                        # assign IDs inline
                        self._batch_seq += 1
                        batch_no = self._batch_seq

                        n = len(chosen)
                        start = self._txn_seq + 1
                        self._txn_seq += n
                        chosen["BATCH_NO"] = batch_no
                        chosen["TRANSACTION_NO"] = list(range(start, start + n))

                        n2 = len(cash_proxy)
                        start2 = self._txn_seq + 1
                        self._txn_seq += n2
                        cash_proxy["BATCH_NO"] = batch_no
                        cash_proxy["TRANSACTION_NO"] = list(range(start2, start2 + n2))

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
                        cash_buckets = self.get_cash_buckets(ports)

        # --------- Phase 2: Ranked product loop (mandates excluded) ----------
        if cash_buckets is None or cash_buckets.empty:
            return pd.concat(buys, ignore_index=True) if buys else pd.DataFrame(columns=self.recommendations.columns)

        for _ in range(self.max_rebalance_loops):
            diffs = self.get_port_model_allocation_diff(ports, ppm)
            under_cols = list(asset_map.keys())
            min_col = diffs[under_cols].idxmin(axis=1).iloc[0]
            min_val = float(diffs[min_col].iloc[0])     # negative if underweight
            min_asset = asset_map[min_col]

            if min_val > -self.underweight_threshold:
                break  # underweight gate

            cash_buckets = self.get_cash_buckets(ports)
            total_cash_w = float(cash_buckets["WEIGHT"].sum()) if not cash_buckets.empty else 0.0
            if total_cash_w < self.buy_step_weight_max or cash_buckets.empty:
                break  # cash gate
            made_trade = False

            for _, cb in cash_buckets.iterrows():
                fund_ccy = cb["CURRENCY"]
                cash_w_ccy = float(cb["WEIGHT"] or 0.0)
                if cash_w_ccy <= self.eps:
                    continue

                step_w = float(min(self.buy_step_weight_max, cash_w_ccy, abs(min_val)))
                step_amount = step_w * total_value

                # BOTH thresholds must pass for buys
                if (step_w < self.min_weight_per_row) or (abs(step_amount) < self.min_amount_per_row):
                    continue
                if step_w <= self.eps:
                    continue

                df_out = ports.df_out[["PORT_ID", "SRC_SHARECODES", "CURRENCY", "VALUE", "WEIGHT"]]

                cand = product_recommendation_rank[
                    (product_recommendation_rank["ASSET_CLASS_NAME"] == min_asset) &
                    (product_recommendation_rank["CURRENCY"] == fund_ccy)
                ]
                if "PRODUCT_TYPE_DESC" in cand.columns:
                    cand = cand[cand["PRODUCT_TYPE_DESC"] != "Mandate"]

                candidates = (
                    cand.merge(df_out, on="SRC_SHARECODES", how="left", suffixes=("", "_PORT"))
                       .fillna({"VALUE": 0.0, "WEIGHT": 0.0})
                )

                if candidates.empty:
                    continue

                candidates["PORT_ID"] = (
                    ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0]
                )
                candidates["FLAG"] = f"{min_asset.lower().replace(' ','_')}_buy"
                candidates["EXPECTED_WEIGHT"] = candidates["WEIGHT"] + step_w
                candidates["ACTION"] = "buy"
                candidates["AMOUNT"] = step_amount
                candidates = candidates[candidates["EXPECTED_WEIGHT"] < self.single_line_cap_non_mandate]
                if candidates.empty:
                    continue

                best_rank = candidates["RANK_PRODUCT"].min()
                chosen = candidates[candidates["RANK_PRODUCT"] == best_rank].copy()
                if "CURRENCY" not in chosen.columns and "CURRENCY_PORT" in chosen.columns:
                    chosen["CURRENCY"] = chosen["CURRENCY_PORT"]
                if "CURRENCY" not in chosen.columns:
                    chosen["CURRENCY"] = fund_ccy

                chosen = chosen[
                    [
                        "PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY",
                        "VALUE","WEIGHT","FLAG","EXPECTED_WEIGHT","ACTION","AMOUNT",
                    ]
                ]
                cash_proxy = self.build_cash_proxy_funding(ports, chosen, per_row=True)
                
                # assign IDs inline
                self._batch_seq += 1
                batch_no = self._batch_seq

                n = len(chosen)
                start = self._txn_seq + 1
                self._txn_seq += n
                chosen["BATCH_NO"] = batch_no
                chosen["TRANSACTION_NO"] = list(range(start, start + n))

                n2 = len(cash_proxy)
                start2 = self._txn_seq + 1
                self._txn_seq += n2
                cash_proxy["BATCH_NO"] = batch_no
                cash_proxy["TRANSACTION_NO"] = list(range(start2, start2 + n2))

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

        return pd.concat(buys, ignore_index=True) if buys else pd.DataFrame(columns=self.recommendations.columns)

    # ---------------------------
    # Cash overweight -> Cash Proxy (same currency) — 1-to-1 pairing
    # ---------------------------
    def move_cash_overweight_to_proxy(self, ports, ppm) -> pd.DataFrame:
        if "PRODUCT_TYPE_DESC" not in ports.df_out.columns:
            return pd.DataFrame(columns=self.recommendations.columns)

        if "SYMBOL" in ports.df_out.columns:
            mask = (ports.df_out["PRODUCT_TYPE_DESC"] == "Cash") & (ports.df_out["SYMBOL"] != "CASH PROXY")
        else:
            mask = (ports.df_out["PRODUCT_TYPE_DESC"] == "Cash")

        cash_rows = ports.df_out[mask].copy()
        if cash_rows.empty:
            return pd.DataFrame(columns=self.recommendations.columns)

        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)[0]
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on="PORT_ID", how="left")

        w_cash_now = float(cash_rows["WEIGHT"].sum() or 0.0)
        w_cash_model = float(diffs["AA_CASH_MODEL"].iloc[0])
        w_over = max(0.0, w_cash_now - w_cash_model)
        if w_over <= self.eps:
            return pd.DataFrame(columns=self.recommendations.columns)

        total_value = float(ports.df_out["VALUE"].sum())
        if w_cash_now <= self.eps_tiny:
            return pd.DataFrame(columns=self.recommendations.columns)

        cash_map = ports.product_mapping[ports.product_mapping["SYMBOL"] == "CASH PROXY"][
            ["PRODUCT_ID", "SRC_SHARECODES", "DESK", "PORT_TYPE", "CURRENCY"]
        ]

        pairs = []
        for _, r in cash_rows.iterrows():
            share = float(r["WEIGHT"]) / w_cash_now
            move_w = w_over * share
            amt_sell = - move_w * total_value

            # SELL leg must meet BOTH thresholds
            if (move_w < self.min_weight_per_row) or (abs(amt_sell) < self.min_amount_per_row):
                continue
            if abs(amt_sell) <= self.eps:
                continue

            sell_row = pd.DataFrame([{
                "PORT_ID":        r["PORT_ID"],
                "PRODUCT_ID":     r["PRODUCT_ID"],
                "SRC_SHARECODES": r["SRC_SHARECODES"],
                "DESK":           r["DESK"],
                "PORT_TYPE":      r["PORT_TYPE"],
                "CURRENCY":       r.get("CURRENCY", np.nan),
                "VALUE":          r.get("VALUE", np.nan),
                "WEIGHT":         r.get("WEIGHT", np.nan),
                "FLAG":           "cash_overweight",  # hardcoded
                "EXPECTED_WEIGHT": (r.get("WEIGHT", np.nan) - move_w) if pd.notna(r.get("WEIGHT", np.nan)) else np.nan,
                "ACTION":         "sell",
                "AMOUNT":         amt_sell,
            }])

            # funding buy (CASH PROXY) 1:1 for this sell row
            funding_buy = sell_row[["PORT_ID","CURRENCY","AMOUNT"]].copy()
            funding_buy["AMOUNT"] = -funding_buy["AMOUNT"]
            funding_buy = funding_buy.merge(cash_map, on="CURRENCY", how="left")
            funding_buy["VALUE"] = np.nan
            funding_buy["WEIGHT"] = np.nan
            funding_buy["FLAG"] = "cash_overweight_to_proxy"  # hardcoded
            funding_buy["EXPECTED_WEIGHT"] = np.nan
            funding_buy["ACTION"] = "funding"
            funding_buy = funding_buy[
                ["PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY",
                 "VALUE","WEIGHT","FLAG","EXPECTED_WEIGHT","ACTION","AMOUNT"]
            ]

            # assign IDs inline
            self._batch_seq += 1
            batch_no = self._batch_seq

            n = len(sell_row)
            start = self._txn_seq + 1
            self._txn_seq += n
            sell_row["BATCH_NO"] = batch_no
            sell_row["TRANSACTION_NO"] = list(range(start, start + n))

            n2 = len(funding_buy)
            start2 = self._txn_seq + 1
            self._txn_seq += n2
            funding_buy["BATCH_NO"] = batch_no
            funding_buy["TRANSACTION_NO"] = list(range(start2, start2 + n2))

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

    # ---------------------------
    # Orchestrator
    # ---------------------------
    def rebalance(self, ports, ppm, hs, reset_state: bool = True, refresh_refs: bool = False) -> pd.DataFrame:
        try:
            # optionally refresh DB-backed reference tables
            if refresh_refs:
                self.refresh_reference_data()

            # clear ephemeral state so we don't carry over from previous runs
            if reset_state:
                self.reset_state()

            sells = self.build_sell_recommendations(ports, ppm, hs)
            if not sells.empty:
                self.recommendations = pd.concat([self.recommendations, sells], ignore_index=True)

            cash_shift = self.move_cash_overweight_to_proxy(ports, ppm)
            if not cash_shift.empty:
                self.recommendations = pd.concat([self.recommendations, cash_shift], ignore_index=True)

            buys = self.build_buy_recommendations(ports, ppm)
            if not buys.empty:
                self.recommendations = pd.concat([self.recommendations, buys], ignore_index=True)

            return self.recommendations
        
        except Exception as e:
            print(e)
            # if not self.recommendations.empty:
            #     return self.recommendations
            
            return pd.DataFrame(columns=self.recommendations.columns)
