"""Tests for cost estimation module."""
import pytest
from unittest.mock import patch, MagicMock
from src.tools.cost import estimate_tokens, estimate_cost, check_limits, MODEL_PRICING, fetch_real_balance


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string_returns_zero(self):
        assert estimate_tokens("", "deepseek-v4-flash") == 0

    def test_english_text(self):
        tokens = estimate_tokens("Hello, world!", "deepseek-v4-flash")
        assert tokens > 0
        assert tokens < 10  # short text

    def test_chinese_text(self):
        tokens = estimate_tokens("你好世界", "deepseek-v4-flash")
        assert tokens > 0

    def test_mixed_text(self):
        tokens = estimate_tokens(
            "Hello 你好, this is a mixed text 混合文本 test.", "deepseek-v4-flash"
        )
        assert tokens > 0

    def test_unknown_model_fallback(self):
        """Unknown model should fall back to default tokenizer."""
        tokens = estimate_tokens("test text", "nonexistent-model")
        assert tokens > 0

    def test_long_text_scales(self):
        short = estimate_tokens("a", "deepseek-v4-flash")
        long = estimate_tokens("a" * 1000, "deepseek-v4-flash")
        assert long > short


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_deepseek_cost(self):
        """DeepSeek pricing: input 1 CNY/M tokens, output 2 CNY/M tokens."""
        cost = estimate_cost(
            tokens_in=1000, tokens_out=500, model="deepseek-v4-flash"
        )
        expected = (1000 / 1_000_000) * 1.0 + (500 / 1_000_000) * 2.0
        assert cost == pytest.approx(expected)

    def test_zero_tokens_zero_cost(self):
        assert estimate_cost(0, 0, "deepseek-v4-flash") == 0.0

    def test_default_model_pricing(self):
        """Models not in pricing table should use default pricing."""
        cost = estimate_cost(1000000, 1000000, "unknown-model")
        # Default: 1 CNY/M input, 2 CNY/M output
        assert cost == pytest.approx(3.0)  # 1 + 2 = 3 CNY

    def test_pricing_table_has_expected_models(self):
        """Pricing table should include common models."""
        assert "deepseek-v4-flash" in MODEL_PRICING
        assert "gpt-4o" in MODEL_PRICING
        assert "qwen-turbo" in MODEL_PRICING


class TestCheckLimits:
    """Tests for cost/token limit checking."""

    def test_within_limits(self):
        blocked, reason = check_limits(
            current_tokens_in=1000,
            current_cost_yuan=0.01,
            limits={
                "max_tokens_per_request": 50000,
                "max_cost_per_request_yuan": 0.5,
            },
        )
        assert blocked is False
        assert reason == ""

    def test_token_limit_exceeded(self):
        blocked, reason = check_limits(
            current_tokens_in=60000,
            current_cost_yuan=0.01,
            limits={
                "max_tokens_per_request": 50000,
                "max_cost_per_request_yuan": 0.5,
            },
        )
        assert blocked is True
        assert "token" in reason.lower()

    def test_cost_limit_exceeded(self):
        blocked, reason = check_limits(
            current_tokens_in=1000,
            current_cost_yuan=0.60,
            limits={
                "max_tokens_per_request": 50000,
                "max_cost_per_request_yuan": 0.5,
            },
        )
        assert blocked is True
        assert "费用" in reason or "cost" in reason.lower()

    def test_both_limits_exceeded(self):
        blocked, reason = check_limits(
            current_tokens_in=60000,
            current_cost_yuan=0.60,
            limits={
                "max_tokens_per_request": 50000,
                "max_cost_per_request_yuan": 0.5,
            },
        )
        assert blocked is True

    def test_at_boundary_not_exceeded(self):
        blocked, reason = check_limits(
            current_tokens_in=50000,
            current_cost_yuan=0.50,
            limits={
                "max_tokens_per_request": 50000,
                "max_cost_per_request_yuan": 0.5,
            },
        )
        assert blocked is False


class TestFetchRealBalance:
    """Tests for DeepSeek real balance fetching."""

    @patch("src.tools.cost.httpx.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"is_available": True, "balance": "8.56", "currency": "CNY"}
        mock_get.return_value = mock_resp

        result = fetch_real_balance("sk-real-key")
        assert result == pytest.approx(8.56)
        mock_get.assert_called_once()

    @patch("src.tools.cost.httpx.get")
    def test_api_error_returns_none(self, mock_get):
        mock_get.side_effect = Exception("connection refused")
        assert fetch_real_balance("sk-real-key") is None

    def test_fake_key_returns_none(self):
        assert fetch_real_balance("sk-fake-test") is None

    def test_empty_key_returns_none(self):
        assert fetch_real_balance("") is None
