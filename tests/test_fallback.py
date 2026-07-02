"""Tests for src.fallback — local model degradation."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from src.fallback.provider import FallbackChatModel
from src.observability.run_log import RunLog, StepLog


class TestFallbackChatModel:
    """Fallback model construction and state tracking."""

    def test_default_attributes(self):
        llm = FallbackChatModel(api_key="sk-test")
        assert llm.primary_model == "deepseek-v4-flash"
        assert llm.fallback_model == "qwen2.5:1.5b"
        assert llm.timeout == 10
        assert llm.used_fallback is False

    def test_identifying_params(self):
        llm = FallbackChatModel(api_key="sk-test")
        params = llm._identifying_params
        assert params["primary_model"] == "deepseek-v4-flash"
        assert params["fallback_model"] == "qwen2.5:1.5b"

    def test_llm_type(self):
        llm = FallbackChatModel(api_key="sk-test")
        assert llm._llm_type == "fallback-chat-model"

    @patch("src.fallback.provider.ChatOpenAI._generate")
    def test_uses_primary_on_success(self, mock_generate):
        mock_result = MagicMock()
        mock_generate.return_value = mock_result

        llm = FallbackChatModel(api_key="sk-test")
        result = llm._generate([HumanMessage("hello")])

        assert result is mock_result
        assert llm.used_fallback is False
        assert mock_generate.call_count == 1

    @patch("src.fallback.provider.ChatOpenAI._generate")
    def test_falls_back_when_primary_fails(self, mock_generate):
        # First call fails, second succeeds
        mock_result = MagicMock()
        mock_generate.side_effect = [Exception("API down"), mock_result]

        llm = FallbackChatModel(api_key="sk-test")
        result = llm._generate([HumanMessage("hello")])

        assert result is mock_result
        assert llm.used_fallback is True
        assert mock_generate.call_count == 2

    @patch("src.fallback.provider.ChatOpenAI._generate")
    def test_raises_when_both_fail(self, mock_generate):
        mock_generate.side_effect = Exception("Both failed")

        llm = FallbackChatModel(api_key="sk-test")
        with pytest.raises(Exception, match="Both failed"):
            llm._generate([HumanMessage("hello")])
        assert llm.used_fallback is False  # never reached success


class TestRunLogFallback:
    """RunLog fallback field."""

    def test_runlog_has_fallback_default(self):
        log = RunLog(run_id="r1", user_query="q", mode="quick", model="test")
        assert log.fallback is False

    def test_runlog_fallback_in_jsonl_roundtrip(self, tmp_path):
        log = RunLog(run_id="r1", user_query="q", mode="quick", model="test", fallback=True)
        log.steps.append(
            StepLog(step_id=1, name="step1", input_preview="in", output_preview="out", duration_ms=10)
        )

        # Write
        p = tmp_path / "test.jsonl"
        p.write_text(log.to_jsonl(), encoding="utf-8")

        # Read
        loaded = RunLog.from_jsonl(p)
        assert loaded.fallback is True
        assert loaded.run_id == "r1"

    def test_runlog_jsonl_backward_compat(self, tmp_path):
        """Old JSONL without fallback field should load with fallback=False."""
        old = (
            '{"type": "run", "run_id": "r1", "user_query": "q", "mode": "quick", '
            '"model": "test", "total_tokens_in": 0, "total_tokens_out": 0, '
            '"total_cost_yuan": 0.0, "result_path": null}\n'
        )
        p = tmp_path / "old.jsonl"
        p.write_text(old, encoding="utf-8")
        loaded = RunLog.from_jsonl(p)
        assert loaded.fallback is False
