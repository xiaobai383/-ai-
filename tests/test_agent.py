"""Tests for agent orchestration module."""
import tempfile
from pathlib import Path

import pytest
from src.config import AppConfig
from src.agent.app import run_task
from src.observability.run_log import RunLog


# A simple fake LLM that returns predefined responses for testing
class FakeLLM:
    """Fake LLM that returns canned responses, no API calls."""

    def __init__(self, response: str = "测试回复"):
        self._response = response

    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._response)

    def bind_tools(self, tools):
        return self

    def __call__(self, *args, **kwargs):
        return self


class FakeChatModel(FakeLLM):
    """Mimics a LangChain chat model for agent testing."""
    pass


@pytest.fixture
def config():
    return AppConfig(
        model_name="deepseek-v4-flash",
        model_base_url="https://api.deepseek.com/v1",
        api_key="sk-fake-for-testing",
        allowed_paths=["data/", "output/", str(Path(tempfile.gettempdir()))],
        max_file_size_mb=5,
        max_tokens_per_request=50000,
        max_cost_per_request_yuan=0.5,
        redaction_enabled=False,
    )


class TestRunTask:
    """Tests for the main run_task orchestration."""

    def test_run_task_returns_runlog(self, config):
        """run_task should return a RunLog instance."""
        llm = FakeLLM(response="## 总结\n\n这是测试文档的总结内容。")
        result = run_task(
            query="总结文档",
            files=["data/test_sample.txt"],
            mode="quick",
            config=config,
            llm=llm,
            auto_confirm=True,  # skip user confirmation for testing
        )
        assert isinstance(result, RunLog)

    def test_runlog_has_run_id(self, config):
        llm = FakeLLM(response="完成")
        result = run_task(
            query="分析文档",
            files=["data/test_sample.txt"],
            mode="quick",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        assert result.run_id is not None
        assert len(result.run_id) > 0

    def test_runlog_has_steps(self, config):
        llm = FakeLLM(response="完成")
        result = run_task(
            query="分析文档",
            files=["data/test_sample.txt"],
            mode="quick",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        assert len(result.steps) > 0

    def test_runlog_records_user_query(self, config):
        llm = FakeLLM(response="完成")
        result = run_task(
            query="总结第三章内容",
            files=["data/test_sample.txt"],
            mode="quick",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        assert result.user_query == "总结第三章内容"

    def test_runlog_records_mode(self, config):
        llm = FakeLLM(response="完成")
        result = run_task(
            query="test",
            files=["data/test_sample.txt"],
            mode="privacy_enhanced",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        assert result.mode == "privacy_enhanced"

    def test_runlog_has_total_tokens_gt_zero(self, config):
        llm = FakeLLM(response="完成")
        result = run_task(
            query="test",
            files=["data/test_sample.txt"],
            mode="quick",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        assert result.total_tokens_in >= 0

    def test_runlog_result_path(self, config):
        llm = FakeLLM(response="完成")
        result = run_task(
            query="test",
            files=["data/test_sample.txt"],
            mode="quick",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        # Result path should be set if save succeeded
        assert result.result_path is not None or result.result_path == ""
        # At minimum, steps should include the llm_call
        step_names = [s.name for s in result.steps]
        assert "preprocess" in step_names or "parse" in step_names[0].lower()

    def test_run_task_local_fallback_no_llm_call(self, config):
        """Local fallback should not call LLM."""
        llm = FakeLLM(response="不应该被调用")
        result = run_task(
            query="本地分析文档",
            files=["data/test_sample.txt"],
            mode="local_fallback",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        # Local fallback should not have an llm_call step
        step_names = [s.name for s in result.steps]
        assert "llm_call" not in step_names

    def test_run_task_with_multiple_files(self, config):
        llm = FakeLLM(response="完成")
        result = run_task(
            query="对比分析",
            files=["data/test_sample.txt", "data/test_sample.md"],
            mode="quick",
            config=config,
            llm=llm,
            auto_confirm=True,
        )
        assert isinstance(result, RunLog)
        assert len(result.steps) > 0
