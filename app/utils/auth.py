"""
JWT authentication and authorization utilities for Supabase tokens.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

import jwt
from fastapi import HTTPException, Request
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

# Cache for JWKS client
_jwks_client: Optional[PyJWKClient] = None


def get_jwks_client() -> PyJWKClient:
    """Get or create JWKS client for Supabase token verification."""
    global _jwks_client

    if _jwks_client is None:
        supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        if not supabase_url:
            raise ValueError("NEXT_PUBLIC_SUPABASE_URL environment variable not set")

        # Remove trailing slash if present
        supabase_url = supabase_url.rstrip("/")
        jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"

        try:
            _jwks_client = PyJWKClient(jwks_url)
        except Exception as e:
            logger.error(f"Failed to initialize JWKS client: {e}")
            raise

    return _jwks_client


def verify_supabase_token(token: str) -> dict:
    """
    Verify Supabase JWT token and return decoded claims.

    Args:
        token: JWT token from Authorization header

    Returns:
        Decoded token claims including sub (user_id), role, etc.

    Raises:
        HTTPException: If token is invalid or verification fails
    """
    try:
        # First, decode without verification to check the algorithm
        unverified_header = jwt.get_unverified_header(token)
        token_alg = unverified_header.get("alg")
        logger.info(f"Token algorithm: {token_alg}")

        supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL").rstrip("/")

        # Try verification based on the token's algorithm
        if token_alg in ["RS256", "ES256"]:
            # Use JWKS for asymmetric algorithms (RS256, ES256)
            try:
                jwks_client = get_jwks_client()
                signing_key = jwks_client.get_signing_key_from_jwt(token)

                decoded = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=[token_alg],
                    audience="authenticated",
                    issuer=f"{supabase_url}/auth/v1",
                    options={
                        "verify_signature": True,
                        "verify_aud": True,
                        "verify_iss": True,
                        "verify_exp": True,
                    },
                )
            except Exception as e:
                logger.error(f"{token_alg} verification failed: {e}")
                raise HTTPException(
                    status_code=401, detail="Token verification failed"
                ) from e

        elif token_alg == "HS256":
            # Use JWT secret for HS256 tokens
            jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
            if not jwt_secret:
                # Try to get it from service role key (fallback)
                service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                if service_role_key and service_role_key.startswith("sb_secret_"):
                    jwt_secret = service_role_key
                else:
                    raise ValueError("SUPABASE_JWT_SECRET environment variable not set")

            decoded = jwt.decode(
                token,
                jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=f"{supabase_url}/auth/v1",
                options={
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_iss": True,
                    "verify_exp": True,
                },
            )

        else:
            # Unknown algorithm - reject token
            logger.error(f"Unsupported JWT algorithm: {token_alg}")
            raise HTTPException(status_code=401, detail="Unsupported token algorithm")

        return decoded

    except jwt.ExpiredSignatureError as e:
        logger.warning("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired") from e
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token") from e
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed") from e


def get_sales_id_from_user(user_id: str) -> Optional[str]:
    """
    Map authenticated user ID to sales_id using user_profile table.

    Args:
        user_id: Supabase user ID (from 'sub' claim)

    Returns:
        sales_id if found, None otherwise
    """
    try:
        from supabase import Client, create_client

        supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not configured")
            return None

        supabase: Client = create_client(supabase_url, supabase_key)

        # Query user_profile table to get sales_id
        try:
            response = (
                supabase.table("user_profile")
                .select("sales_id")
                .eq("id", user_id)
                .execute()
            )

            if response.data and len(response.data) > 0:
                sales_id = response.data[0].get("sales_id")
                if sales_id:
                    return str(sales_id)

        except Exception:
            # User likely doesn't exist in user_profile table
            pass

        # For development, return a default sales_id
        return "84"

    except Exception as e:
        logger.error(f"Error fetching sales_id for user {user_id}: {e}")
        return None


def get_user_role_from_token(decoded_token: dict) -> str:
    """
    Extract user role from user_profile table.

    Args:
        decoded_token: Decoded JWT claims

    Returns:
        User role (system_admin, app_admin, or user)
    """
    try:
        from supabase import Client, create_client

        supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not configured for role lookup")
            return "user"

        supabase: Client = create_client(supabase_url, supabase_key)
        user_id = decoded_token.get("sub")

        if not user_id:
            logger.warning("No user_id found in token")
            return "user"

        # Query user_profile table to get user_role
        try:
            response = (
                supabase.table("user_profile")
                .select("user_role")
                .eq("id", user_id)
                .execute()
            )

            if response.data and len(response.data) > 0:
                user_role = response.data[0].get("user_role")
                if user_role:
                    return str(user_role)

        except Exception:
            # User likely doesn't exist in user_profile table
            pass

        return "user"

    except Exception as e:
        logger.error(f"Error fetching user_role: {e}")
        return "user"


async def get_current_user(request: Request) -> tuple[str, str, str]:
    """
    Extract and verify authentication from request.

    Args:
        request: FastAPI request object

    Returns:
        Tuple of (user_id, sales_id, user_role)

    Raises:
        HTTPException: If authentication fails
    """
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )

    token = auth_header.replace("Bearer ", "")

    # Verify token
    decoded_token = verify_supabase_token(token)

    # Extract user ID
    user_id = decoded_token.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token: missing user ID")

    # Get sales_id from user mapping
    sales_id = get_sales_id_from_user(user_id)
    if not sales_id:
        raise HTTPException(
            status_code=403, detail="User not authorized: no sales_id mapping found"
        )

    # Get user role
    user_role = get_user_role_from_token(decoded_token)

    return user_id, sales_id, user_role


async def get_system_or_current_user(request: Request) -> tuple[str, str, str]:
    service_token = request.headers.get("X-Service-Token")
    expected_token = os.getenv("INTERNAL_SERVICE_TOKEN")

    if (
        service_token
        and expected_token
        and secrets.compare_digest(service_token, expected_token)
    ):
        user_id = os.getenv("INTERNAL_SERVICE_USER_ID", "system_mcp")
        sales_id = os.getenv("INTERNAL_SERVICE_SALES_ID", "84")
        user_role = "system_admin"
        return user_id, sales_id, user_role

    return await get_current_user(request)
