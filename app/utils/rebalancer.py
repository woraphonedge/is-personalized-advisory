import copy
import warnings

import numpy as np
import pandas as pd

from .portfolios import Portfolios


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
            "transaction_no", "batch_no", "port_id", "product_id", "src_sharecodes", "desk",
            "port_type", "currency","product_display_name", "product_type_desc", "asset_class_name",
            "value", "weight", "flag", "expected_weight", "action", "amount",
        ]
        self.recommendations = pd.DataFrame(columns=self._reco_cols)

        # preload raw tables
        self.es_sell_list: pd.DataFrame | None = None
        self.product_recommendation_rank_raw: pd.DataFrame | None = None
        self.mandate_allocation: pd.DataFrame | None = None


    # ---------- state mgmt ----------
    def reset_state(self) -> None:
        """Clear per-run state so the same instance can be reused safely."""
        self._txn_seq = 0
        self._batch_seq = 0
        self.recommendations = pd.DataFrame(columns=self._reco_cols)

    # ---------- cached loaders ----------
    def set_ref_tables(self, ref_dict: dict):
        """Reload cached reference tables (if you expect DB to have changed)."""
        # Set DataFrames with validation
        self.es_sell_list = None if "es_sell_list" not in ref_dict else ref_dict["es_sell_list"]
        self.product_recommendation_rank_raw = None if "product_recommendation_rank_raw" not in ref_dict else ref_dict["product_recommendation_rank_raw"]
        self.mandate_allocation = None if "mandate_allocation" not in ref_dict else ref_dict["mandate_allocation"]

        # Validate required elements
        required_elements = {
            "es_sell_list": self.es_sell_list,
            "product_recommendation_rank_raw": self.product_recommendation_rank_raw,
            "mandate_allocation": self.mandate_allocation,
        }
        for key, value in required_elements.items():
            if value is None:
                warnings.warn(f"'{key}' not provided. Expect errors if used.", UserWarning, stacklevel=2)

    def get_product_recommendation_rank(self, ports) -> pd.DataFrame:
        """Merge the (preloaded) recommendation rank with current product mapping."""
        if self.product_recommendation_rank_raw is None:
            self.load_product_recommendation_rank_raw()
        df = self.product_recommendation_rank_raw.merge(
            ports.product_mapping,
            on=["src_sharecodes", "desk", "currency"],
            how="left",
            suffixes=("", "_mapping")
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
                "port_id","product_id","src_sharecodes","desk","port_type","currency","product_display_name", "product_type_desc", "asset_class_name",
                "value","weight","flag","expected_weight","action","amount"
            ])
        if ports.product_mapping is None:
            raise RuntimeError("Product mapping must be loaded before mapping cash proxy.")

        cash_map = ports.product_mapping[ports.product_mapping["symbol"] == "CASH PROXY"][
            ["product_id", "src_sharecodes", "desk", "port_type", "currency","product_display_name", "product_type_desc", "asset_class_name"]
        ].copy()

        if per_row:
            base = df[["port_id","currency","amount"]].copy()
        else:
            base = df.groupby(["port_id","currency"], dropna=False, as_index=False)["amount"].sum()

        out = base.merge(cash_map, on="currency", how="left")
        out["amount"] = -out["amount"].astype(float)
        out["value"] = np.nan
        out["weight"] = np.nan
        out["flag"] = "cash_proxy_funding"  # hardcoded
        out["expected_weight"] = np.nan
        out["action"] = "funding"

        cols = ["port_id","product_id","src_sharecodes","desk","port_type","currency","product_display_name", "product_type_desc", "asset_class_name",
                "value","weight","flag","expected_weight","action","amount"]
        out = out.loc[:, ~out.columns.duplicated()]
        return out[cols]

    # ---------------------------
    # Portfolio updater
    # ---------------------------
    def update_portfolio(self, ports: "Portfolios", trade_rows: pd.DataFrame) -> None:
        # Create new_ports if not already created
        if not hasattr(self, "new_ports") or self.new_ports is None:
            # Make a deep copy of ports to preserve its structure and data
            self.new_ports = copy.deepcopy(ports)

        # Work with new_ports instead of original ports
        new_ports = self.new_ports

        # Prepare trade rows as position changes
        trade_as_positions = (
            trade_rows[["port_id", "amount"] + new_ports.prod_comp_keys]
            .rename(columns={"amount": "value"})
            .copy()
        )

        # Get current portfolio positions
        base_positions = new_ports.df_out[["port_id", "value"] + new_ports.prod_comp_keys].copy()

        # Combine base and trade positions
        df_products = pd.concat([base_positions, trade_as_positions], ignore_index=True, sort=False)

        # Aggregate positions by product keys
        df_products = (
            df_products.groupby(
                ["port_id"] + new_ports.prod_comp_keys,
                dropna=False,
                as_index=False,
            )["value"]
            .sum()
        )

        # Merge product mapping
        df_out = df_products.merge(
            new_ports.product_mapping,
            on=new_ports.prod_comp_keys,
            how="left",
        )

        # Filter out zero positions
        df_out = df_out[df_out["value"] != 0]

        # Update new_ports (not the original ports)
        new_ports.set_portfolio(df_out, new_ports.df_style, new_ports.port_ids, new_ports.port_id_mapping)


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

        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on="port_id", how="left")

        aa_cols = ["aa_cash","aa_fi","aa_le","aa_ge","aa_alt"]
        need = np.array([max(float(diffs[f"{c}_model"].iloc[0] - diffs[c].iloc[0]), 0.0) for c in aa_cols], dtype=float)
        if need.sum() <= self.eps_tiny:
            return None
        need = need / need.sum()

        cands = mandate_df[mandate_df["currency"] == currency].copy()
        if cands.empty:
            return None

        # filter universe by headroom amount
        if "product_type_desc" in cands.columns:
            if headroom_amt + self.eps >= self.min_mandate_amount:
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

    def get_cash_buckets(self, ports) -> pd.DataFrame:
        ## Comment: Should it belong to ports?
        cash = ports.df_out[ports.df_out["symbol"] == "CASH PROXY"]
        if cash.empty:
            return cash
        out = (cash.groupby(["port_id","currency"], as_index=False)
                    .agg(weight=("weight","sum"), value=("value","sum"))).sort_values("weight", ascending=False)
        return out

    def get_mandate_weight(self, ports) -> float:
        if "product_type_desc" in ports.df_out.columns:
            return float(ports.df_out.loc[ports.df_out["product_type_desc"] == "Mandate", "weight"].sum() or 0.0)
        mandates = self.load_mandate_candidates()
        mandate_syms = set(mandates["symbol"].astype(str).unique())
        if "symbol" in ports.df_out.columns:
            return float(ports.df_out.loc[ports.df_out["symbol"].astype(str).isin(mandate_syms), "weight"].sum() or 0.0)
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
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        comp = comp.merge(self.es_sell_list, left_on="src_sharecodes", right_on="symbol", how="left")
        sub = comp[~comp["symbol"].isna()].copy()
        sub["flag"] = "sell_list"
        sub["expected_weight"] = 0
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
            "port_id","product_id","src_sharecodes","desk","port_type","currency","product_display_name", "product_type_desc", "asset_class_name","value","weight",
        ]
        rec = (
            rec.groupby(key_cols, dropna=False, as_index=False)
              .agg(flag=("flag", lambda x: ", ".join(sorted(set(x)))),
                   expected_weight=("expected_weight", "min"))
        )

        rec["action"] = "sell"
        with np.errstate(divide="ignore", invalid="ignore"):
            rec["amount"] = -1.0 * ((rec["weight"] - rec["expected_weight"]) / rec["weight"]) * rec["value"]
        rec["amount"] = rec["amount"].fillna(0.0)

        chosen = rec.loc[rec["amount"] != 0].copy()
        if chosen.empty:
            return pd.DataFrame(columns=self.recommendations.columns)

        # build row-by-row (to allow per-row min checks)
        trades = []
        for _, row in chosen.iterrows():
            row_df = pd.DataFrame([row[
                ["port_id","product_id","src_sharecodes","desk","port_type","currency","product_display_name", "product_type_desc", "asset_class_name",
                 "value","weight","flag","expected_weight","action","amount"]
            ]])

            # row-level minimum checks (BOTH must pass)
            try:
                w_change = abs(float(row_df["weight"].iloc[0] - row_df["expected_weight"].iloc[0]))
            except Exception:
                w_change = 0.0
            a_abs = abs(float(row_df["amount"].iloc[0] or 0.0))
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
            row_df["batch_no"] = batch_no
            row_df["transaction_no"] = list(range(start, start + n))

            n2 = len(funding_df)
            start2 = self._txn_seq + 1
            self._txn_seq += n2
            funding_df["batch_no"] = batch_no
            funding_df["transaction_no"] = list(range(start2, start2 + n2))

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
    ## Comment: Can we delegate this task to portprop?
    def get_port_model_allocation_diff(ports, ppm) -> pd.DataFrame:
        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        out = port_alloc.merge(model_alloc, on=["port_id"], how="left")
        for col in ["aa_cash", "aa_fi", "aa_le", "aa_ge", "aa_alt"]:
            out[f"{col}_diff"] = out[col] - out[f"{col}_model"]
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

        ## Comment: Why we dont use self.mandate_allocation instead of creating
        # new variables mandates, it is easier to track.
        mandates = self.mandate_allocation

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
        min_col = diffs[under_cols].idxmin(axis=1).iloc[0]
        min_val = float(diffs[min_col].iloc[0])     # negative if underweight
        abs_under = abs(min_val)

        cash_buckets = self.get_cash_buckets(ports)
        total_cash_w = float(cash_buckets["weight"].sum()) if not cash_buckets.empty else 0.0

        mandate_w_now = self.get_mandate_weight(ports)
        mandate_headroom = max(0.0, self.discretionary_acceptance - mandate_w_now)

        if (mandate_headroom > self.eps and
            abs_under > self.eps and
            total_cash_w > self.eps and
            not cash_buckets.empty):

            for i in range(len(cash_buckets)):
                if i>=len(cash_buckets):
                    break
                fund_ccy = cash_buckets.iloc[i]["currency"]
                cash_w_ccy = float(cash_buckets.iloc[i]["weight"] or 0.0)

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
                            "flag":           "discretionary_buy",  # hardcoded
                            "expected_weight":cur_weight + add_w,
                            "action":         "buy",
                            "amount":         amt,
                        }])

                        # funding per row to keep 1-1 pair
                        cash_proxy = self.build_cash_proxy_funding(ports, chosen, per_row=True)

                        # assign IDs inline
                        self._batch_seq += 1
                        batch_no = self._batch_seq

                        n = len(chosen)
                        start = self._txn_seq + 1
                        self._txn_seq += n
                        chosen["batch_no"] = batch_no
                        chosen["transaction_no"] = list(range(start, start + n))

                        n2 = len(cash_proxy)
                        start2 = self._txn_seq + 1
                        self._txn_seq += n2
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
            total_cash_w = float(cash_buckets["weight"].sum()) if not cash_buckets.empty else 0.0
            if total_cash_w < self.buy_step_weight_max or cash_buckets.empty:
                break  # cash gate
            made_trade = False

            for _, cb in cash_buckets.iterrows():
                fund_ccy = cb["currency"]
                cash_w_ccy = float(cb["weight"] or 0.0)
                if cash_w_ccy <= self.eps:
                    continue

                step_w = float(min(self.buy_step_weight_max, cash_w_ccy, abs(min_val)))
                step_amount = step_w * total_value

                # BOTH thresholds must pass for buys
                if (step_w < self.min_weight_per_row) or (abs(step_amount) < self.min_amount_per_row):
                    continue
                if step_w <= self.eps:
                    continue

                df_out = ports.df_out[["port_id", "src_sharecodes", "currency", "value", "weight"]]

                cand = product_recommendation_rank[
                    (product_recommendation_rank["asset_class_name"] == min_asset) &
                    (product_recommendation_rank["currency"] == fund_ccy)
                ]
                if "product_type_desc" in cand.columns:
                    cand = cand[cand["product_type_desc"] != "Mandate"]

                candidates = (
                    cand.merge(df_out, on="src_sharecodes", how="left", suffixes=("", "_port"))
                        .fillna({"value": 0.0, "weight": 0.0})
                )

                if candidates.empty:
                    continue

                candidates["port_id"] = (
                    ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0]
                )
                candidates["flag"] = f"{min_asset.lower().replace(" ","_")}_buy"
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

                chosen = chosen[
                    [
                        "port_id","product_id","src_sharecodes","desk","port_type","currency","product_display_name", "product_type_desc", "asset_class_name",
                        "value","weight","flag","expected_weight","action","amount",
                    ]
                ]
                cash_proxy = self.build_cash_proxy_funding(ports, chosen, per_row=True)

                # assign IDs inline
                self._batch_seq += 1
                batch_no = self._batch_seq

                n = len(chosen)
                start = self._txn_seq + 1
                self._txn_seq += n
                chosen["batch_no"] = batch_no
                chosen["transaction_no"] = list(range(start, start + n))

                n2 = len(cash_proxy)
                start2 = self._txn_seq + 1
                self._txn_seq += n2
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

        return pd.concat(buys, ignore_index=True) if buys else pd.DataFrame(columns=self.recommendations.columns)

    # ---------------------------
    # Cash overweight -> Cash Proxy (same currency) — 1-to-1 pairing
    # ---------------------------
    def move_cash_overweight_to_proxy(self, ports, ppm) -> pd.DataFrame:
        if "product_type_desc" not in ports.df_out.columns:
            return pd.DataFrame(columns=self.recommendations.columns)

        if "symbol" in ports.df_out.columns:
            mask = (ports.df_out["product_type_desc"] == "Cash") & (ports.df_out["symbol"] != "CASH PROXY")
        else:
            mask = (ports.df_out["product_type_desc"] == "Cash")

        cash_rows = ports.df_out[mask].copy()
        if cash_rows.empty:
            return pd.DataFrame(columns=self.recommendations.columns)

        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on="port_id", how="left")

        w_cash_now = float(cash_rows["weight"].sum() or 0.0)
        w_cash_model = float(diffs["aa_cash_model"].iloc[0])
        w_over = max(0.0, w_cash_now - w_cash_model)
        if w_over <= self.eps:
            return pd.DataFrame(columns=self.recommendations.columns)

        total_value = float(ports.df_out["value"].sum())
        if w_cash_now <= self.eps_tiny:
            return pd.DataFrame(columns=self.recommendations.columns)

        cash_map = ports.product_mapping[ports.product_mapping["symbol"] == "CASH PROXY"][
            ["product_id", "src_sharecodes", "desk", "port_type", "currency"]
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
                "value":          r.get("value", np.nan),
                "weight":         r.get("weight", np.nan),
                "flag":           "cash_overweight",  # hardcoded
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
            funding_buy["flag"] = "cash_overweight_to_proxy"  # hardcoded
            funding_buy["expected_weight"] = np.nan
            funding_buy["action"] = "funding"
            funding_buy = funding_buy[
                [
                    "port_id","product_id","src_sharecodes","desk","port_type","currency","product_display_name", "product_type_desc", "asset_class_name",
                    "value","weight","flag","expected_weight","action","amount"
                ]
            ]

            # assign IDs inline
            self._batch_seq += 1
            batch_no = self._batch_seq

            n = len(sell_row)
            start = self._txn_seq + 1
            self._txn_seq += n
            sell_row["batch_no"] = batch_no
            sell_row["transaction_no"] = list(range(start, start + n))

            n2 = len(funding_buy)
            start2 = self._txn_seq + 1
            self._txn_seq += n2
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

            self.new_ports = copy.deepcopy(ports)

            sells = self.build_sell_recommendations(self.new_ports, ppm, hs)
            if not sells.empty:
                self.recommendations = pd.concat([self.recommendations, sells], ignore_index=True)

            cash_shift = self.move_cash_overweight_to_proxy(self.new_ports, ppm)
            if not cash_shift.empty:
                self.recommendations = pd.concat([self.recommendations, cash_shift], ignore_index=True)

            buys = self.build_buy_recommendations(self.new_ports, ppm)
            if not buys.empty:
                self.recommendations = pd.concat([self.recommendations, buys], ignore_index=True)

            return self.new_ports, self.recommendations

        except Exception as e:
            print(e)
            # if not self.recommendations.empty:
            #     return self.recommendations

            return self.new_ports, pd.DataFrame(columns=self.recommendations.columns)
