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


def list_clients(
    df_style: pd.DataFrame,
    port_id_mapping: pd.DataFrame,
    customer_id: int | None = None,
    query: str | None = None,
    sales_id: str | None = None,
) -> ClientListResponse:
    """Search and list clients based on customer_id, client name (Thai), or sales_id.

    Args:
        df_style: DataFrame containing client investment styles and metadata
        port_id_mapping: DataFrame mapping port_id to customer_id
        customer_id: Partial customer ID to search for (optional)
        query: Partial client name in Thai to search for (optional)
        sales_id: Sales ID to filter clients (optional, for access control)

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

    # Filter based on search criteria
    mask = None

    # Filter by sales_id first (access control)
    if sales_id is not None and sales_id.strip():
        if "sales_id" in df_merged.columns:
            mask = df_merged["sales_id"].astype(str) == sales_id
            logger.debug("Filtering by sales_id=%s", sales_id)
        else:
            logger.warning("sales_id column not found in df_style, skipping sales_id filter")

    if customer_id is not None:
        # Partial customer_id match (convert to string for partial matching)
        customer_id_str = str(customer_id)
        customer_mask = df_merged["customer_id"].astype(str).str.contains(customer_id_str, na=False)
        mask = customer_mask if mask is None else (mask & customer_mask)

    if query is not None and query.strip():
        # Partial client name match (case-insensitive) - search in both Thai and English names
        name_mask = None
        if "client_full_name_th" in df_merged.columns:
            name_mask = df_merged["client_full_name_th"].astype(str).str.contains(query, na=False, case=False)
        if "client_first_name_en" in df_merged.columns:
            en_mask = df_merged["client_first_name_en"].astype(str).str.contains(query, na=False, case=False)
            name_mask = en_mask if name_mask is None else (name_mask | en_mask)

        if name_mask is not None:
            mask = name_mask if mask is None else (mask & name_mask)
        else:
            logger.debug("client name columns not found in df_style, skipping name search")

    # Apply filter or return all if no criteria
    if mask is not None:
        df_filtered = df_merged[mask]
    else:
        df_filtered = df_merged

    # Remove duplicates by customer_id and take most recent as_of_date
    if "as_of_date" in df_filtered.columns:
        df_filtered = df_filtered.sort_values("as_of_date", ascending=False)

    df_unique = df_filtered.drop_duplicates(subset=["customer_id"], keep="first")

    # Limit to 10 results
    df_result = df_unique.head(10)

    # Build response with all new fields
    clients: List[ClientListItem] = []
    for _, row in df_result.iterrows():
        client_item = ClientListItem(
            customer_id=int(row["customer_id"]),
            client_full_name_th=row.get("client_full_name_th"),
            client_first_name_en=row.get("client_first_name_en"),
            port_investment_style=row.get("port_investment_style"),
            client_tier=row.get("client_tier"),
            business_unit=row.get("business_unit"),
            client_segment_by_inv_aum=row.get("client_segment_by_inv_aum"),
            client_sub_segment_by_inv_aum=row.get("client_sub_segment_by_inv_aum"),
            sales_id=row.get("sales_id"),
            ui_client=row.get("ui_client"),
            sales_first_name_en=row.get("sales_first_name_en"),
            sales_team=row.get("sales_team"),
        )
        clients.append(client_item)

    return ClientListResponse(clients=clients, total=len(clients))
