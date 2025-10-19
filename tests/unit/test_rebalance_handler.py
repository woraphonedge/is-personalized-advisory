"""Unit tests for rebalance_handler module."""

import pandas as pd
import pytest

from app.utils.health_service import STYLE_MAP
from app.utils.rebalance_handler import _build_df_style


class TestBuildDfStyle:
    """Test suite for _build_df_style function."""

    def test_valid_string_style(self):
        """Test with valid string style."""
        df = _build_df_style(14055, "High Risk")

        assert len(df) == 1
        assert df.iloc[0]["customer_id"] == 14055
        assert df.iloc[0]["port_investment_style"] == "High Risk"
        assert df.iloc[0]["portpop_style"] == "High Risk"

    def test_valid_dict_style(self):
        """Test with valid dict containing client_style."""
        df = _build_df_style(14055, {"client_style": "Conservative"})

        assert len(df) == 1
        assert df.iloc[0]["customer_id"] == 14055
        assert df.iloc[0]["port_investment_style"] == "Conservative"
        assert df.iloc[0]["portpop_style"] == "Conservative"

    def test_dict_with_legacy_keys(self):
        """Test dict with legacy INVESTMENT_STYLE keys."""
        # Test INVESTMENT_STYLE key
        df = _build_df_style(14055, {"INVESTMENT_STYLE": "Moderate Low Risk"})
        assert df.iloc[0]["port_investment_style"] == "Moderate Low Risk"

        # Test INVESTMENT_STYLE_AUMX key
        df = _build_df_style(14055, {"INVESTMENT_STYLE_AUMX": "Aggressive Growth"})
        assert df.iloc[0]["port_investment_style"] == "Aggressive Growth"

    def test_none_style_defaults_to_high_risk(self):
        """Test that None style defaults to High Risk."""
        df = _build_df_style(14055, None)

        assert df.iloc[0]["port_investment_style"] == "High Risk"
        assert df.iloc[0]["portpop_style"] == "High Risk"

    def test_empty_string_defaults_to_high_risk(self):
        """Test that empty/whitespace string defaults to High Risk."""
        df = _build_df_style(14055, "   ")

        assert df.iloc[0]["port_investment_style"] == "High Risk"
        assert df.iloc[0]["portpop_style"] == "High Risk"

    def test_unknown_style_maps_to_high_risk(self):
        """Test that unknown style maps to High Risk with warning."""
        df = _build_df_style(14055, "Unknown Style")

        assert df.iloc[0]["port_investment_style"] == "Unknown Style"
        assert df.iloc[0]["portpop_style"] == "High Risk"  # Mapped to default

    def test_list_with_dict_style(self):
        """Test with list containing dict."""
        df = _build_df_style(14055, [{"client_style": "Aggressive Growth"}])

        assert df.iloc[0]["port_investment_style"] == "Aggressive Growth"
        assert df.iloc[0]["portpop_style"] == "Aggressive"

    def test_list_with_string_style(self):
        """Test with list containing string."""
        df = _build_df_style(14055, ["Moderate Low Risk"])

        assert df.iloc[0]["port_investment_style"] == "Moderate Low Risk"
        assert df.iloc[0]["portpop_style"] == "Medium to Moderate Low Risk"

    def test_empty_list_defaults_to_high_risk(self):
        """Test that empty list defaults to High Risk."""
        df = _build_df_style(14055, [])

        assert df.iloc[0]["port_investment_style"] == "High Risk"
        assert df.iloc[0]["portpop_style"] == "High Risk"

    def test_invalid_customer_id_zero(self):
        """Test that customer_id=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid customer_id: 0"):
            _build_df_style(0, "High Risk")

    def test_invalid_customer_id_negative(self):
        """Test that negative customer_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid customer_id: -1"):
            _build_df_style(-1, "High Risk")

    def test_invalid_customer_id_type(self):
        """Test that non-integer customer_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid customer_id: invalid"):
            _build_df_style("invalid", "High Risk")  # type: ignore

    def test_invalid_customer_id_float(self):
        """Test that float customer_id raises ValueError."""
        with pytest.raises(ValueError, match="Invalid customer_id"):
            _build_df_style(14055.5, "High Risk")  # type: ignore

    @pytest.mark.parametrize(
        "style,expected_portpop",
        [
            ("Bulletproof", "Conservative"),
            ("Conservative", "Conservative"),
            ("Moderate Low Risk", "Medium to Moderate Low Risk"),
            ("Moderate High Risk", "Medium to Moderate High Risk"),
            ("High Risk", "High Risk"),
            ("Aggressive Growth", "Aggressive"),
            ("Unwavering", "Aggressive"),
        ],
    )
    def test_all_valid_styles_from_style_map(self, style, expected_portpop):
        """Test all valid styles from STYLE_MAP."""
        df = _build_df_style(14055, style)

        assert df.iloc[0]["port_investment_style"] == style
        assert df.iloc[0]["portpop_style"] == expected_portpop

    def test_returns_dataframe(self):
        """Test that function returns a pandas DataFrame."""
        result = _build_df_style(14055, "High Risk")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == [
            "customer_id",
            "port_investment_style",
            "portpop_style",
        ]

    def test_dataframe_column_types(self):
        """Test that DataFrame has correct column types."""
        df = _build_df_style(14055, "High Risk")

        assert df["customer_id"].dtype == "int64"
        assert df["port_investment_style"].dtype == "object"  # string
        assert df["portpop_style"].dtype == "object"  # string

    def test_style_map_consistency(self):
        """Test that all STYLE_MAP keys work correctly."""
        for style_key in STYLE_MAP.keys():
            df = _build_df_style(14055, style_key)
            assert df.iloc[0]["portpop_style"] == STYLE_MAP[style_key]

    def test_dict_key_priority(self):
        """Test that client_style key has priority over other keys."""
        df = _build_df_style(
            14055,
            {
                "client_style": "High Risk",
                "style": "Conservative",
                "INVESTMENT_STYLE": "Moderate Low Risk",
            },
        )

        # Should use client_style (first in priority)
        assert df.iloc[0]["port_investment_style"] == "High Risk"

    def test_large_customer_id(self):
        """Test with very large customer_id."""
        df = _build_df_style(999999999, "High Risk")

        assert df.iloc[0]["customer_id"] == 999999999

    def test_whitespace_in_style_string(self):
        """Test that whitespace is properly handled."""
        df = _build_df_style(14055, "  High Risk  ")

        assert df.iloc[0]["port_investment_style"] == "High Risk"
