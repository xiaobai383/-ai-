"""Tests for src.replay — RunLog loading and filtering."""
import tempfile
from pathlib import Path

import pytest

from src.observability.run_log import RunLog, StepLog
from src.replay.loader import RunLogList, RunLogLoader, RunLogSummary


class TestRunLogLoader:
    """Loader scanning and filtering."""

    @pytest.fixture
    def logs_dir(self):
        """Create a temp dir with sample RunLog JSONL files."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)

            # Success run
            log1 = RunLog(
                run_id="run-001",
                user_query="总结文档要点",
                mode="privacy_enhanced",
                model="deepseek-v4-flash",
                result_path="output/run-001.md",
            )
            log1.steps.append(
                StepLog(step_id=1, name="preprocess", input_preview="f1.txt", output_preview="1 doc", duration_ms=100, status="success")
            )
            (d / "run-001.jsonl").write_text(log1.to_jsonl(), encoding="utf-8")

            # Failed run
            log2 = RunLog(
                run_id="run-002",
                user_query="风险分析",
                mode="quick",
                model="deepseek-v4-flash",
                result_path=None,
            )
            log2.steps.append(
                StepLog(step_id=1, name="preprocess", input_preview="f2.txt", output_preview="error", duration_ms=100, status="failed")
            )
            (d / "run-002.jsonl").write_text(log2.to_jsonl(), encoding="utf-8")

            # Fallback run
            log3 = RunLog(
                run_id="run-003",
                user_query="提取待办",
                mode="privacy_enhanced",
                model="qwen2.5:1.5b",
                result_path="output/run-003.md",
                fallback=True,
            )
            log3.steps.append(
                StepLog(step_id=1, name="llm_call", input_preview="q", output_preview="ok", duration_ms=500, status="success")
            )
            (d / "run-003.jsonl").write_text(log3.to_jsonl(), encoding="utf-8")

            yield d

    def test_list_all_returns_all(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all()
        assert result.total == 3
        assert len(result.items) == 3

    def test_mode_filter(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all(mode_filter="quick")
        assert result.total == 1
        assert result.items[0].run_id == "run-002"

    def test_status_filter_success(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all(status_filter="success")
        assert result.total == 2
        ids = {r.run_id for r in result.items}
        assert "run-001" in ids
        assert "run-003" in ids

    def test_status_filter_failed(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all(status_filter="failed")
        assert result.total == 1
        assert result.items[0].run_id == "run-002"

    def test_search(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all(search="风险")
        assert result.total == 1
        assert result.items[0].run_id == "run-002"

    def test_search_case_insensitive(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all(search="总结")
        assert result.total == 1

    def test_limit_and_offset(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all(limit=2, offset=0)
        assert len(result.items) == 2
        assert result.total == 3

        result2 = loader.list_all(limit=2, offset=2)
        assert len(result2.items) == 1

    def test_load_by_id(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        log = loader.load_by_id("run-001")
        assert log is not None
        assert log.run_id == "run-001"
        assert log.user_query == "总结文档要点"

    def test_load_by_id_not_found(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        assert loader.load_by_id("nonexistent") is None

    def test_count(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        assert loader.count() == 3

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = RunLogLoader(tmp)
            result = loader.list_all()
            assert result.total == 0
            assert loader.count() == 0

    def test_fallback_field(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all()
        fallback_ids = {r.run_id for r in result.items if r.fallback}
        assert "run-003" in fallback_ids

    def test_summary_has_step_count(self, logs_dir):
        loader = RunLogLoader(str(logs_dir))
        result = loader.list_all()
        for item in result.items:
            assert item.step_count > 0


class TestRunLogSummary:
    """Dataclass defaults."""

    def test_defaults(self):
        s = RunLogSummary(
            run_id="r1", user_query="q", mode="m", model="m",
            result_path=None, total_tokens_in=0, total_tokens_out=0,
            total_cost_yuan=0.0, step_count=0, fallback=False,
        )
        assert s.status == "unknown"

    def test_runlog_list_defaults(self):
        rl = RunLogList()
        assert rl.items == []
        assert rl.total == 0
