from app.utils.utils import get_latest_eom, read_sql, write_parquet

# Export tables from DWH to local data/ parquet files
ES_SELL_SQL = "select * from user.kwm.personalized_advisory_es_sell_list"
RECO_SQL = "select * from user.kwm.personalized_advisory_recommendation_rank"
UNDERLYING_SQL = (
    """
    SELECT * FROM edp.kkps_vw.v_npii_prod_undlycomp_map
    WHERE DATA_DT = (SELECT max(DATA_DT) FROM edp.kkps_vw.v_npii_prod_undlycomp_map)
    """
)

ES_SELL_FILE = "es_sell_list.parquet"
RECO_FILE = "product_recommendation_rank.parquet"
UNDERLYING_FILE = "underlying_mapping.parquet"


def export_product_mapping(as_of_date: str) -> int:
    dim_keys = ['PRODUCT_ID', 'SRC_SHARECODES', 'DESK', 'PORT_TYPE']
    dim_cols = dim_keys + [
        'PRODUCT_DISPLAY_NAME', 'CURRENCY', 'PRODUCT_TYPE_DESC', 'ASSET_CLASS_NAME',
        'SYMBOL', 'PP_ASSET_SUB_CLASS', 'IS_RISKY_ASSET', 'COVERAGE_PRDTYPE', 'IS_COVERAGE', 'EXPECTED_RETURN'
    ]
    query = (
        f"SELECT distinct {', '.join(dim_cols)} FROM ("
        f"select * from user.kwm.health_score_dim_product_info_range('{as_of_date}','{as_of_date}') "
        f"union all select * from user.kwm.personalized_advisory_cash_proxy "
        f"union all select * except (AA_CASH, AA_FI, AA_LE, AA_GE, AA_ALT) from user.kwm.personalized_advisory_asset_allocation_weight)"
    )
    try:
        df_map = read_sql(query)
        path = write_parquet(df_map, f"product_mapping_{as_of_date}.parquet")
        print(f"Wrote product mapping ({as_of_date}) to {path}")
        return 0
    except Exception as e:
        print(f"Failed to export product mapping {as_of_date}: {e}")
        return 1


def main() -> int:
    # Prompt for AS_OF_DATE via stdin with default to latest EOM to avoid env dependency
    default_eom = get_latest_eom()
    try:
        user_in = input(f"AS_OF_DATE (YYYY-MM-DD) [default {default_eom}]: ").strip()
        as_of_date = user_in or default_eom
    except EOFError:
        # Non-interactive mode: fall back to default
        as_of_date = default_eom

    try:
        df_es = read_sql(ES_SELL_SQL)
        p1 = write_parquet(df_es, ES_SELL_FILE)
        print(f"Wrote ES sell list to {p1}")
    except Exception as e:
        print(f"Failed to export ES sell list: {e}")
        return 1

    try:
        df_rec = read_sql(RECO_SQL)
        p2 = write_parquet(df_rec, RECO_FILE)
        print(f"Wrote recommendation rank to {p2}")
    except Exception as e:
        print(f"Failed to export recommendation rank: {e}")
        return 1

    try:
        df_under = read_sql(UNDERLYING_SQL)
        p3 = write_parquet(df_under, UNDERLYING_FILE)
        print(f"Wrote underlying mapping to {p3}")
    except Exception as e:
        print(f"Failed to export underlying mapping: {e}")
        return 1

    # Export product mapping with the chosen as_of_date
    rc = export_product_mapping(as_of_date)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
