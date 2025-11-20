"""
Client listing and search service.

Provides functionality to search and list clients based on customer_id or client name.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from app.models import ClientListItem, ClientListResponse

logger = logging.getLogger(__name__)


def get_accessible_customer_ids(
    sales_id: str, sales_customer_mapping_df: pd.DataFrame | None = None
) -> set[int]:
    """Get all customer IDs that a sales_id can access from in-memory sales-customer mapping.

    This includes direct customers and team customers for team leads/managers.

    Args:
        sales_id: The sales ID to get accessible customers for
        sales_customer_mapping_df: Optional DataFrame containing sales-customer mappings.
                                 If not provided, will attempt to use app.state from FastAPI.

    Returns:
        Set of customer IDs that the sales_id can access
    """
    try:
        # Use provided DataFrame or try to get from app.state
        if sales_customer_mapping_df is None:
            try:
                # Import here to avoid circular imports
                from app.data_store import get_sales_customer_mapping

                sales_customer_mapping_df = get_sales_customer_mapping()
            except (ImportError, RuntimeError) as e:
                logger.warning(
                    f"Could not get sales-customer mapping from data store: {e}. "
                    "Falling back to empty access set."
                )
                return set()

        # Filter DataFrame for the given sales_id
        if sales_customer_mapping_df.empty:
            logger.warning("Sales-customer mapping DataFrame is empty")
            return set()

        # Convert sales_id to int for matching (CSV data is likely numeric)
        try:
            sales_id_int = int(sales_id)
        except ValueError:
            logger.warning(f"Invalid sales_id format: {sales_id}")
            return set()

        # Filter mappings for this sales_id
        filtered_mappings = sales_customer_mapping_df[
            sales_customer_mapping_df["SALES_ID"] == sales_id_int
        ]

        # Extract customer IDs
        customer_ids = set(filtered_mappings["CUSTOMER_ID"].tolist())
        logger.debug(
            f"Found {len(customer_ids)} accessible customers for sales_id={sales_id}"
        )

        return customer_ids

    except Exception as e:
        logger.error(f"Error fetching accessible customer_ids from in-memory data: {e}")
        # Return empty set to fallback to direct filtering
        return set()


def list_clients(
    df_style: pd.DataFrame,
    port_id_mapping: pd.DataFrame,
    acct_cust_mapping: pd.DataFrame,
    customer_id: int | None = None,
    query: str | None = None,
    sales_id: str | None = None,
    user_role: str | None = None,
    sales_customer_mapping_df: pd.DataFrame | None = None,
) -> ClientListResponse:
    """Search and list clients based on customer_id, client name (Thai), or sales_id.

    Args:
        df_style: DataFrame containing client investment styles and metadata
        port_id_mapping: DataFrame mapping port_id to customer_id
        acct_cust_mapping: DataFrame mapping account/sub-account to customer_id
        customer_id: Partial customer ID to search for (optional)
        query: Partial client name in Thai to search for (optional)
        sales_id: Sales ID to filter clients (optional, for access control)
        user_role: User role (system_admin, app_admin, user) - admins bypass sales_id filter
        sales_customer_mapping_df: DataFrame containing sales-customer mappings for access control

    Returns:
        ClientListResponse with up to 10 matching clients

    Raises:
        ValueError: If df_style or port_id_mapping is empty or None
    """
    # Validate inputs
    if df_style is None or df_style.empty:
        logger.warning("df_style is empty or not loaded")
        return ClientListResponse(clients=[], total=0)

    if port_id_mapping is None or port_id_mapping.empty:
        logger.warning("port_id_mapping is empty or not loaded")
        return ClientListResponse(clients=[], total=0)

    # Merge df_style with port_id_mapping to get customer_id
    df_merged = df_style.merge(port_id_mapping, on="port_id", how="left")

    # Filter based on search criteria (but skip sales_id for now - will handle later with Supabase)
    mask = None

    # Apply customer_id filter
    if customer_id is not None:
        # Partial customer_id match (convert to string for partial matching)
        customer_id_str = str(customer_id)
        customer_mask = (
            df_merged["customer_id"].astype(str).str.contains(customer_id_str, na=False)
        )
        mask = customer_mask if mask is None else (mask & customer_mask)

    # Apply query filter (customer_id, Thai name, and English name)
    if query is not None and query.strip():
        # Search in customer_id, Thai name, and English name (OR condition for all three)
        search_mask = None

        # Search in customer_id (convert to string for partial matching)
        customer_id_mask = (
            df_merged["customer_id"]
            .astype(str)
            .str.contains(query, na=False, case=False)
        )
        search_mask = customer_id_mask

        # Search in Thai name
        if "client_full_name_th" in df_merged.columns:
            name_mask = (
                df_merged["client_full_name_th"]
                .astype(str)
                .str.contains(query, na=False, case=False)
            )
            search_mask = search_mask | name_mask

        # Search in English name
        if "client_first_name_en" in df_merged.columns:
            en_mask = (
                df_merged["client_first_name_en"]
                .astype(str)
                .str.contains(query, na=False, case=False)
            )
            search_mask = search_mask | en_mask

        # Search with account no
        if "sub_account_no" in acct_cust_mapping.columns:
            acct_cust_mask = (
                acct_cust_mapping["sub_account_no"]
                .astype(str)
                .str.contains(query, na=False, case=False)
            )
            found_acct = acct_cust_mapping[acct_cust_mask]

            acct_mask = (
                df_merged["customer_id"].astype(str).isin(found_acct["customer_id"])
            )

            search_mask = search_mask | acct_mask

        # Combine with existing mask (AND condition with other filters)
        mask = search_mask if mask is None else (mask & search_mask)
        logger.debug("Applied query search for: %s", query)

    # Apply search filters (customer_id and query) but skip sales_id for now
    if mask is not None:
        logger.debug("Search mask length: %s", len(mask))
        df_search_filtered = df_merged[mask]
    else:
        df_search_filtered = df_merged
        logger.debug("No search criteria provided, using all clients")

    # Now apply access control using in-memory sales-customer mapping
    is_admin = user_role in ["system_admin", "app_admin"]
    if is_admin:
        logger.debug("Admin user (role=%s) - bypassing access control", user_role)
        df_final_filtered = df_search_filtered
    elif sales_id is not None and sales_id.strip():
        # Get accessibule cstomer IDs from in-memory sales-customer mapping
        accessible_customer_ids = get_accessible_customer_ids(
            sales_id, sales_customer_mapping_df
        )

        if accessible_customer_ids:
            # Filter search results to only include accessible customers
            df_final_filtered = df_search_filtered[
                df_search_filtered["customer_id"].isin(accessible_customer_ids)
            ]
            logger.debug(
                "Filtered %d search results to %d accessible customers for sales_id=%s",
                len(df_search_filtered),
                len(df_final_filtered),
                sales_id,
            )
        else:
            # No accessible customers found for this sales_id
            logger.warning(
                "No accessible customers found for sales_id=%s",
                sales_id,
            )
            df_final_filtered = df_search_filtered.head(0)  # Return empty results
    else:
        # No sales_id provided and not admin - return empty for security
        logger.warning(
            "No sales_id provided for non-admin user, returning empty results"
        )
        df_final_filtered = df_search_filtered.head(0)

    # Remove duplicates by customer_id and take most recent as_of_date
    if "as_of_date" in df_final_filtered.columns:
        df_final_filtered = df_final_filtered.sort_values("as_of_date", ascending=False)

    df_unique = df_final_filtered.drop_duplicates(subset=["customer_id"], keep="first")

    # Limit to 10 results
    df_result = df_unique.head(10)

    # Helper to normalize DataFrame values to Optional[str]
    def _to_optional_str(value):
        if pd.isna(value):
            return None
        # Ensure non-string values (e.g. ints) are converted to strings for Pydantic
        return str(value) if not isinstance(value, str) else value

    # Convert to list of dicts and build response with all new fields
    records = df_result.to_dict(orient="records")

    clients: List[ClientListItem] = [
        ClientListItem(
            customer_id=int(r["customer_id"]),
            client_full_name_th=_to_optional_str(r.get("client_full_name_th")),
            client_first_name_en=_to_optional_str(r.get("client_first_name_en")),
            port_investment_style=_to_optional_str(r.get("port_investment_style")),
            client_tier=_to_optional_str(r.get("client_tier")),
            business_unit=_to_optional_str(r.get("business_unit")),
            client_segment_by_inv_aum=_to_optional_str(
                r.get("client_segment_by_inv_aum")
            ),
            client_sub_segment_by_inv_aum=_to_optional_str(
                r.get("client_sub_segment_by_inv_aum")
            ),
            sales_id=_to_optional_str(r.get("sales_id")),
            ui_client=_to_optional_str(r.get("ui_client")),
            sales_first_name_en=_to_optional_str(r.get("sales_first_name_en")),
            sales_team=_to_optional_str(r.get("sales_team")),
        )
        for r in records
    ]

    return ClientListResponse(clients=clients, total=len(clients))
