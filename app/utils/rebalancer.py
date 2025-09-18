import os

import numpy as np
import pandas as pd

from .portfolios import Portfolios
from .utils import read_parquet, read_sql, write_parquet

# Cache filenames (module-level constants)
ES_SELL_LIST_FILE = "es_sell_list.parquet"
RECO_RANK_FILE = "product_recommendation_rank.parquet"


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
        as_of_date: str = "2025-07-31",
        customer_id: int = None,
        client_investment_style: str = None,
        client_classification: str = None,
        new_money: float = 0.0,
        discretionary_percent: float = 0.5,
        private_percent: float = 0.0,
        cash_percent: float | None = None,
        offshore_percent: float | None = None,
        product_restriction: list[str] | None = None,
        # Mandate exposure cap as a weight fraction (e.g. 0.4 = 40%). None -> no cap (treated as 1.0)
        discretionary_acceptance: float | None = None,
    ) -> None:
        self.as_of_date = as_of_date
        self.customer_id = customer_id
        self.client_investment_style = client_investment_style
        self.client_classification = client_classification
        self.new_money = new_money
        self.discretionary_percent = discretionary_percent
        self.private_percent = private_percent
        self.cash_percent = cash_percent
        self.offshore_percent = offshore_percent
        self.product_restriction = product_restriction or []
        self.discretionary_acceptance = 1.0 if discretionary_acceptance is None else float(discretionary_acceptance)

        # transaction sequence for TRANSACTION_NO tagging
        self.transaction_seq = 0

        self.recommendations = pd.DataFrame(
            columns=[
                "TRANSACTION_NO",           # <--- NEW
                "PORT_ID",
                "PRODUCT_ID",
                "SRC_SHARECODES",
                "DESK",
                "PORT_TYPE",
                "CURRENCY",
                "VALUE",
                "WEIGHT",
                "FLAG",
                "EXPECTED_WEIGHT",
                "ACTION",
                "AMOUNT",
            ]
        )

        # Optionally refresh local cache from DWH at initialization
        # We only need raw tables for caching; merges with product mapping are done at use time.
        if str(os.getenv("LOAD_DATA_FROM_DWH", "")).strip().lower() in {"1", "true", "yes"}:
            try:
                df_es = read_sql("select * from user.kwm.personalized_advisory_es_sell_list")
                write_parquet(df_es, ES_SELL_LIST_FILE)
            except Exception as e:
                # Non-fatal: proceed without blocking init
                print(f"[Rebalancer] Warning: failed to refresh ES sell list cache: {e}")
            try:
                df_rec = read_sql("select * from user.kwm.personalized_advisory_recommendation_rank")
                write_parquet(df_rec, RECO_RANK_FILE)
            except Exception as e:
                print(f"[Rebalancer] Warning: failed to refresh recommendation cache: {e}")

    # ---------------------------
    # Helpers (no leading underscores as requested)
    # ---------------------------
    def next_transaction_no(self) -> int:
        self.transaction_seq += 1
        return self.transaction_seq

    def tag_transaction(self, df: pd.DataFrame, txn_no: int) -> pd.DataFrame:
        """Ensure df has TRANSACTION_NO column stamped with txn_no, aligned to recommendations schema."""
        if df is None or df.empty:
            return pd.DataFrame(columns=self.recommendations.columns)
        out = df.copy()
        out["TRANSACTION_NO"] = txn_no
        # Reorder/ensure columns
        cols = self.recommendations.columns
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[cols]

    def map_cash_proxy(self, ports: Portfolios, df_products: pd.DataFrame) -> pd.DataFrame:
        if ports.product_mapping is None:
            raise RuntimeError("Product mapping must be loaded before mapping cash proxy.")
        df_currency = (
            df_products.groupby(["PORT_ID", "CURRENCY"], dropna=False, as_index=False)["AMOUNT"].sum()
        )
        cash_map = ports.product_mapping[ports.product_mapping["SYMBOL"] == "CASH PROXY"][
            ["PRODUCT_ID", "SRC_SHARECODES", "DESK", "PORT_TYPE", "CURRENCY"]
        ]
        df_cash_proxy = df_currency.merge(cash_map, on="CURRENCY", how="left")
        df_cash_proxy["AMOUNT"] = -1.0 * df_cash_proxy["AMOUNT"]
        df_cash_proxy["VALUE"] = np.nan
        df_cash_proxy["WEIGHT"] = np.nan
        df_cash_proxy["FLAG"] = "cash_proxy_funding"
        df_cash_proxy["EXPECTED_WEIGHT"] = np.nan
        df_cash_proxy["ACTION"] = "funding"
        cols = [
            "PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY",
            "VALUE","WEIGHT","FLAG","EXPECTED_WEIGHT","ACTION","AMOUNT"
        ]
        df_cash_proxy = df_cash_proxy.loc[:, ~df_cash_proxy.columns.duplicated()]
        return df_cash_proxy[cols]

    def update_portfolio(self, ports: Portfolios, trade_rows: pd.DataFrame) -> None:
        """
        Apply the proposed trades to the working copy of the portfolio:
        - Combine existing holdings with trade AMOUNT (as VALUE delta)
        - Remove zero positions
        - Recompute derived fields in Portfolios
        """
        trade_as_positions = (
            trade_rows[["PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","AMOUNT"]]
            .rename(columns={"AMOUNT": "VALUE"})
            .copy()
        )
        base_positions = ports.df_out[["PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","VALUE"]].copy()
        df_products = pd.concat([base_positions, trade_as_positions], ignore_index=True, sort=False)
        df_products = (
            df_products.groupby(
                ["PORT_ID", "PRODUCT_ID", "SRC_SHARECODES", "DESK", "PORT_TYPE"],
                dropna=False,
                as_index=False,
            )["VALUE"]
            .sum()
        )
        product_mapping = ports.product_mapping
        df_out = df_products.merge(
            product_mapping[product_mapping["DESK"] != "SIDEB"],
            on=["PRODUCT_ID", "SRC_SHARECODES", "DESK", "PORT_TYPE"],
            how="left",
        )
        df_out = df_out[df_out["VALUE"] != 0]
        ports.set_portfolio(df_out, ports.df_style, ports.port_ids, ports.port_id_mapping)

    # ----- Mandate helpers -----
    def load_mandate_candidates(self) -> pd.DataFrame:
        # Universe with mandate AA weights per symbol/currency
        sql = """select * from user.kwm.personalized_advisory_asset_allocation_weight"""
        df = read_sql(sql)
        for c in ["AA_CASH","AA_FI","AA_LE","AA_GE","AA_ALT"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

    def select_best_mandate_for_currency(self, ports, ppm, mandate_df: pd.DataFrame, currency: str) -> pd.Series | None:
        """
        Choose one same-currency mandate whose AA vector best fits portfolio underweights.
        Fit = min L2 distance between normalized 'need' (model-current, clipped at 0) and mandate AA.
        """
        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)[0]
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on="PORT_ID", how="left")

        aa_cols = ["AA_CASH","AA_FI","AA_LE","AA_GE","AA_ALT"]
        need = np.array([max(float(diffs[f"{c}_MODEL"].iloc[0] - diffs[c].iloc[0]), 0.0) for c in aa_cols], dtype=float)
        if need.sum() <= 1e-12:
            return None
        need = need / need.sum()

        cands = mandate_df[mandate_df["CURRENCY"] == currency].copy()
        if cands.empty:
            return None

        def norm_row(row):
            v = np.array([row[c] for c in aa_cols], dtype=float)
            s = v.sum()
            return v / s if s > 0 else np.zeros_like(v)

        mat = np.vstack([norm_row(r) for _, r in cands.iterrows()])
        dists = np.linalg.norm(mat - need.reshape(1, -1), axis=1)
        min_dist = dists.min()
        winners = cands.iloc[np.where(np.isclose(dists, min_dist, atol=1e-12))[0]]
        return winners.sample(n=1, random_state=np.random.randint(0, 1_000_000)).iloc[0]

    def get_cash_buckets(self, ports) -> pd.DataFrame:
        cash = ports.df_out[ports.df_out["SYMBOL"] == "CASH PROXY"]
        if cash.empty:
            return cash
        out = (cash.groupby(["PORT_ID","CURRENCY"], as_index=False)
                    .agg(WEIGHT=("WEIGHT","sum"), VALUE=("VALUE","sum")))
        return out.sort_values("WEIGHT", ascending=False)

    def get_mandate_weight(self, ports) -> float:
        """
        Current total Mandate weight in the portfolio.
        Prefer PRODUCT_TYPE_DESC=='Mandate' if available, otherwise use SYMBOL join to mandates.
        """
        if "PRODUCT_TYPE_DESC" in ports.df_out.columns:
            return float(ports.df_out.loc[ports.df_out["PRODUCT_TYPE_DESC"] == "Mandate", "WEIGHT"].sum() or 0.0)
        mandates = getattr(self, "_cached_mandates", None)
        if mandates is None:
            mandates = self.load_mandate_candidates()
            self._cached_mandates = mandates
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
        sub["EXPECTED_WEIGHT"] = 0.2
        return sub

    def check_bulk_risk(self, ports, ppm, hs) -> pd.DataFrame:
        _, comp = ports.get_portfolio_health_score(ppm, hs, cal_comp=True)
        col = "IS_BULK_RISK"
        sub = comp[comp[col]].copy() if col in comp.columns else comp.iloc[0:0].copy()
        sub["FLAG"] = "bulk_risk"
        sub["EXPECTED_WEIGHT"] = 0.2
        return sub

    def build_sell_recommendations(self, ports, ppm, hs) -> pd.DataFrame:
        parts = [
            self.check_not_monitored(ports, ppm, hs),
            self.check_issuer_risk(ports, ppm, hs),
            self.check_bulk_risk(ports, ppm, hs),
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

        cash_proxy = self.map_cash_proxy(ports, rec[rec["AMOUNT"] != 0])

        # Tag single transaction number for this sell batch
        txn_no = self.next_transaction_no()
        rec_txn = self.tag_transaction(rec, txn_no)
        cash_proxy["PORT_ID"] = rec_txn["PORT_ID"].iloc[0] if not rec_txn.empty else None
        cash_proxy_txn = self.tag_transaction(cash_proxy, txn_no)

        trade = pd.concat([rec_txn, cash_proxy_txn], ignore_index=True)
        self.update_portfolio(ports, trade)
        return trade

    # ---------------------------
    # Buy phase
    # ---------------------------
    @staticmethod
    def load_es_sell_list() -> pd.DataFrame:
        if str(os.getenv("LOAD_DATA_FROM_DWH", "")).strip().lower() in {"1", "true", "yes"}:
            df = read_sql("select * from user.kwm.personalized_advisory_es_sell_list")
            try:
                write_parquet(df, ES_SELL_LIST_FILE)
            except Exception as e:
                print(f"[Rebalancer] Warning: failed to persist ES sell list: {e}")
            return df
        # load cached
        return read_parquet(ES_SELL_LIST_FILE)

    @staticmethod
    def load_product_recommendation(ports) -> pd.DataFrame:
        if str(os.getenv("LOAD_DATA_FROM_DWH", "")).strip().lower() in {"1", "true", "yes"}:
            df = read_sql("select * from user.kwm.personalized_advisory_recommendation_rank")
            try:
                write_parquet(df, RECO_RANK_FILE)
            except Exception as e:
                print(f"[Rebalancer] Warning: failed to persist product recommendations: {e}")
        else:
            df = read_parquet(RECO_RANK_FILE)
        df = df.merge(
            ports.product_mapping,
            on=["SYMBOL", "PRODUCT_TYPE_DESC", "ASSET_CLASS_NAME", "DESK"],
            how="left",
        )
        return df

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
        Multi-currency buy loop with Mandate cap:
        - Find most underweight asset class.
        - For each cash currency bucket (largest first), try mandate-first (same currency).
        - If no mandate is feasible (or cap reached), fallback to ranked products filtered to the same currency.
        - Step size = min(10%, available cash in that currency, absolute underweight); single-line cap at 20%.
        - Mandate purchases are additionally capped by `self.discretionary_acceptance`.
        """
        product_rank = self.load_product_recommendation(ports)
        mandates = getattr(self, "_cached_mandates", None)
        if mandates is None:
            mandates = self.load_mandate_candidates()
            self._cached_mandates = mandates

        asset_map = {
            "AA_CASH_DIFF": "Cash and Cash Equivalent",
            "AA_FI_DIFF": "Fixed Income",
            "AA_LE_DIFF": "Local Equity",
            "AA_GE_DIFF": "Global Equity",
            "AA_ALT_DIFF": "Alternative",
        }

        buys = []
        total_value = float(ports.df_out["VALUE"].sum())

        for _ in range(50):  # safety loop
            # 1) Most underweight asset class
            diffs = self.get_port_model_allocation_diff(ports, ppm)
            under_cols = list(asset_map.keys())
            min_col = diffs[under_cols].idxmin(axis=1).iloc[0]
            min_val = float(diffs[min_col].iloc[0])     # negative if underweight
            min_asset = asset_map[min_col]

            if min_val > -0.10:
                break  # keep your original 10% underweight gate

            # 2) Enumerate cash buckets by currency (largest first)
            cash_buckets = self.get_cash_buckets(ports)
            total_cash_w = float(cash_buckets["WEIGHT"].sum()) if not cash_buckets.empty else 0.0
            if total_cash_w < 0.10 or cash_buckets.empty:
                break  # keep your original 10% cash gate

            made_trade = False

            for _, cb in cash_buckets.iterrows():
                fund_ccy = cb["CURRENCY"]
                cash_w_ccy = float(cb["WEIGHT"] or 0.0)
                if cash_w_ccy <= 1e-6:
                    continue

                # step = min(10%, available cash in that currency, absolute underweight)
                step_w = float(min(0.10, cash_w_ccy, abs(min_val)))
                if step_w <= 1e-8:
                    continue
                step_amount = step_w * total_value

                # ---- Cap remaining Mandate headroom (portfolio-level) ----
                mandate_w_now = self.get_mandate_weight(ports)
                mandate_headroom = max(0.0, self.discretionary_acceptance - mandate_w_now)

                # 3) Try mandate-first in this currency (respect headroom + single-line cap)
                if mandate_headroom > 1e-8:
                    chosen_md = self.select_best_mandate_for_currency(ports, ppm, mandates, fund_ccy)
                else:
                    chosen_md = None  # cap reached, skip mandates

                if chosen_md is not None:
                    sym = chosen_md["SYMBOL"]
                    existing = ports.df_out[ports.df_out["SYMBOL"] == sym][["WEIGHT","VALUE"]]
                    cur_weight = float(existing["WEIGHT"].sum()) if not existing.empty else 0.0
                    # single-line 20% cap + mandate portfolio cap
                    add_w = min(step_w, max(0.0, 0.20 - cur_weight), mandate_headroom)
                    if add_w > 1e-8:
                        amt = add_w * total_value
                        chosen = pd.DataFrame([{
                            "PORT_ID":        ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0],
                            "PRODUCT_ID":     chosen_md["PRODUCT_ID"],
                            "SRC_SHARECODES": chosen_md["SRC_SHARECODES"],
                            "DESK":           chosen_md["DESK"],
                            "PORT_TYPE":      chosen_md["PORT_TYPE"],
                            "CURRENCY":       chosen_md["CURRENCY"],  # == fund_ccy
                            "VALUE":          0.0,
                            "WEIGHT":         cur_weight,
                            "FLAG":           "mandate_buy",
                            "EXPECTED_WEIGHT":cur_weight + add_w,
                            "ACTION":         "buy",
                            "AMOUNT":         amt,
                        }])
                        cash_proxy = self.map_cash_proxy(ports, chosen)
                        trade = pd.concat([chosen, cash_proxy], ignore_index=True)

                        # assign transaction number to this buy
                        txn_no = self.next_transaction_no()
                        trade = self.tag_transaction(trade, txn_no)
                        buys.append(trade)
                        self.update_portfolio(ports, trade)
                        made_trade = True
                        break  # move to next iteration (recompute underweights and buckets)

                # 4) Fallback to ranked product in the same currency
                if not made_trade:
                    df_out = ports.df_out[["PORT_ID", "SYMBOL", "CURRENCY", "VALUE", "WEIGHT"]]
                    candidates = (
                        product_rank[
                            (product_rank["ASSET_CLASS_NAME"] == min_asset) &
                            (product_rank["CURRENCY"] == fund_ccy)
                        ]
                        .merge(df_out, on="SYMBOL", how="left", suffixes=("", "_PORT"))
                        .fillna({"VALUE": 0.0, "WEIGHT": 0.0})
                    )
                    if candidates.empty:
                        continue

                    # If mandate cap reached, exclude mandate products from fallback
                    if mandate_headroom <= 1e-8 and "PRODUCT_TYPE_DESC" in candidates.columns:
                        candidates = candidates[candidates["PRODUCT_TYPE_DESC"] != "Mandate"]
                        if candidates.empty:
                            continue

                    candidates["PORT_ID"] = (
                        ports.port_ids.iloc[0] if hasattr(ports.port_ids, "iloc") else ports.port_ids[0]
                    )
                    candidates["FLAG"] = "ranked_buy"
                    candidates["EXPECTED_WEIGHT"] = candidates["WEIGHT"] + step_w
                    candidates["ACTION"] = "buy"
                    candidates["AMOUNT"] = step_amount
                    candidates = candidates[candidates["EXPECTED_WEIGHT"] < 0.20]
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
                    cash_proxy = self.map_cash_proxy(ports, chosen)
                    trade = pd.concat([chosen, cash_proxy], ignore_index=True)

                    # assign transaction number to this buy
                    txn_no = self.next_transaction_no()
                    trade = self.tag_transaction(trade, txn_no)
                    buys.append(trade)
                    self.update_portfolio(ports, trade)
                    made_trade = True
                    break

            if not made_trade:
                break

        if not buys:
            return pd.DataFrame(columns=self.recommendations.columns)

        out = pd.concat(buys, ignore_index=True)
        return out

    # ---------------------------
    # Cash overweight -> Cash Proxy (same currency)
    # ---------------------------
    def move_cash_overweight_to_proxy(self, ports, ppm) -> pd.DataFrame:
        """
        If portfolio Cash (PRODUCT_TYPE_DESC == 'Cash') is overweight vs model AA_CASH_MODEL,
        sell the overweight proportionally across Cash lines and buy Cash Proxy in the same currency.

        Returns the trade dataframe (cash sells + cash-proxy buys). Updates portfolio in-place.
        """
        # Need PRODUCT_TYPE_DESC to identify actual cash lines
        if "PRODUCT_TYPE_DESC" not in ports.df_out.columns:
            return pd.DataFrame(columns=self.recommendations.columns)

        cash_rows = ports.df_out[ports.df_out["PRODUCT_TYPE_DESC"] == "Cash"].copy()
        if cash_rows.empty:
            return pd.DataFrame(columns=self.recommendations.columns)

        port_alloc = ports.get_portfolio_asset_allocation_lookthrough(ppm)[0]
        model_alloc = ports.get_model_asset_allocation_lookthrough(ppm)
        diffs = port_alloc.merge(model_alloc, on="PORT_ID", how="left")

        w_cash_now = float(cash_rows["WEIGHT"].sum() or 0.0)
        w_cash_model = float(diffs["AA_CASH_MODEL"].iloc[0])
        w_over = max(0.0, w_cash_now - w_cash_model)
        if w_over <= 1e-8:
            return pd.DataFrame(columns=self.recommendations.columns)

        total_value = float(ports.df_out["VALUE"].sum())
        if w_cash_now <= 1e-12:
            return pd.DataFrame(columns=self.recommendations.columns)

        # Pro-rata sells from each Cash line
        sells = []
        for _, r in cash_rows.iterrows():
            share = float(r["WEIGHT"]) / w_cash_now  # proportion within Cash bucket
            move_w = w_over * share
            amt = - move_w * total_value  # negative to reduce position
            if abs(amt) <= 1e-6:
                continue

            sells.append({
                "PORT_ID":        r["PORT_ID"],
                "PRODUCT_ID":     r["PRODUCT_ID"],
                "SRC_SHARECODES": r["SRC_SHARECODES"],
                "DESK":           r["DESK"],
                "PORT_TYPE":      r["PORT_TYPE"],
                "CURRENCY":       r.get("CURRENCY", np.nan),
                "VALUE":          r.get("VALUE", np.nan),
                "WEIGHT":         r.get("WEIGHT", np.nan),
                "FLAG":           "cash_overweight",
                "EXPECTED_WEIGHT": (r.get("WEIGHT", np.nan) - move_w) if pd.notna(r.get("WEIGHT", np.nan)) else np.nan,
                "ACTION":         "sell",
                "AMOUNT":         amt,
            })

        if not sells:
            return pd.DataFrame(columns=self.recommendations.columns)

        df_sells = pd.DataFrame(sells)

        # Aggregate per currency to buy Cash Proxy in the same currency
        per_ccy_buy = (
            df_sells.groupby(["PORT_ID", "CURRENCY"], dropna=False, as_index=False)["AMOUNT"]
            .sum()
            .assign(AMOUNT=lambda d: -d["AMOUNT"])  # sells negative -> buys positive
        )

        cash_map = ports.product_mapping[ports.product_mapping["SYMBOL"] == "CASH PROXY"][
            ["PRODUCT_ID", "SRC_SHARECODES", "DESK", "PORT_TYPE", "CURRENCY"]
        ]
        buys = per_ccy_buy.merge(cash_map, on="CURRENCY", how="left")
        buys = buys.dropna(subset=["PRODUCT_ID"])  # skip currencies without mapping

        if buys.empty:
            # Apply only the sells to reduce overweight
            txn_no = self.next_transaction_no()
            trade_only_sells = self.tag_transaction(df_sells, txn_no)
            self.update_portfolio(ports, trade_only_sells)
            return trade_only_sells

        buys = buys.assign(
            VALUE=np.nan,
            WEIGHT=np.nan,
            FLAG="cash_overweight_to_proxy",
            EXPECTED_WEIGHT=np.nan,
            ACTION="buy",
        )
        buys = buys[
            [
                "PORT_ID","PRODUCT_ID","SRC_SHARECODES","DESK","PORT_TYPE","CURRENCY",
                "VALUE","WEIGHT","FLAG","EXPECTED_WEIGHT","ACTION","AMOUNT"
            ]
        ]

        # tag the combined trade with a single transaction number
        txn_no = self.next_transaction_no()
        df_sells_txn = self.tag_transaction(df_sells, txn_no)
        buys_txn = self.tag_transaction(buys, txn_no)
        trade = pd.concat([df_sells_txn, buys_txn], ignore_index=True)
        self.update_portfolio(ports, trade)
        return trade

    # ---------------------------
    # Orchestrator
    # ---------------------------
    def rebalance(self, ports, ppm, hs) -> pd.DataFrame:
        # Phase 1: Sells (risk, non-covered, bulk)
        sells = self.build_sell_recommendations(ports, ppm, hs)
        if not sells.empty:
            self.recommendations = pd.concat([self.recommendations, sells], ignore_index=True)

        # Phase 1.5: Normalize cash — move any Cash overweight to Cash Proxy (same currency)
        cash_shift = self.move_cash_overweight_to_proxy(ports, ppm)
        if not cash_shift.empty:
            self.recommendations = pd.concat([self.recommendations, cash_shift], ignore_index=True)

        # Phase 2: Buys (mandate-first with cap; fallback ranked products by same currency)
        buys = self.build_buy_recommendations(ports, ppm)
        if not buys.empty:
            self.recommendations = pd.concat([self.recommendations, buys], ignore_index=True)

        # You commented out collapsing duplicates; leaving as-is to preserve TRANSACTION_NO per leg.
        return self.recommendations
