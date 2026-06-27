"""Tests for RunLog observability module."""
import json
import tempfile
from pathlib import Path

import pytest
from src.observability.run_log import RunLog, StepLog


class TestStepLog:
    """Tests for StepLog dataclass."""

    def test_creates_step_with_required_fields(self):
        step = StepLog(
            step_id=1,
            name="parse_file",
            input_preview="read data/test.txt",
            output_preview="parsed 3 paragraphs",
            duration_ms=150,
        )
        assert step.step_id == 1
        assert step.name == "parse_file"
        assert step.input_preview == "read data/test.txt"
        assert step.output_preview == "parsed 3 paragraphs"
        assert step.duration_ms == 150
        assert step.status == "success"  # default

    def test_default_token_and_cost_zero(self):
        step = StepLog(
            step_id=1,
            name="test",
            input_preview="in",
            output_preview="out",
            duration_ms=10,
        )
        assert step.tokens_in == 0
        assert step.tokens_out == 0
        assert step.cost_yuan == 0.0

    def test_failed_status(self):
        step = StepLog(
            step_id=2,
            name="call_llm",
            input_preview="prompt",
            output_preview="error",
            duration_ms=500,
            status="failed",
        )
        assert step.status == "failed"


class TestRunLog:
    """Tests for RunLog dataclass."""

    def test_creates_runlog_with_steps(self):
        steps = [
            StepLog(
                step_id=1,
                name="parse",
                input_preview="file1.txt",
                output_preview="ok",
                duration_ms=10,
                tokens_in=0,
                tokens_out=0,
            ),
            StepLog(
                step_id=2,
                name="call_llm",
                input_preview="summarize...",
                output_preview="summary text",
                duration_ms=2000,
                tokens_in=500,
                tokens_out=200,
                cost_yuan=0.0015,
            ),
        ]
        run = RunLog(
            run_id="run-001",
            user_query="总结这篇文档",
            mode="privacy_enhanced",
            model="deepseek-v4-flash",
            steps=steps,
        )
        assert run.run_id == "run-001"
        assert run.user_query == "总结这篇文档"
        assert run.mode == "privacy_enhanced"
        assert run.model == "deepseek-v4-flash"
        assert len(run.steps) == 2

    def test_aggregates_total_tokens(self):
        steps = [
            StepLog(
                step_id=1,
                name="s1",
                input_preview="a",
                output_preview="b",
                duration_ms=1,
                tokens_in=100,
                tokens_out=50,
                cost_yuan=0.01,
            ),
            StepLog(
                step_id=2,
                name="s2",
                input_preview="a",
                output_preview="b",
                duration_ms=1,
                tokens_in=200,
                tokens_out=80,
                cost_yuan=0.02,
            ),
        ]
        run = RunLog(
            run_id="r1",
            user_query="q",
            mode="quick",
            model="m",
            steps=steps,
        )
        assert run.total_tokens_in == 300
        assert run.total_tokens_out == 130

    def test_aggregates_total_cost(self):
        steps = [
            StepLog(
                step_id=1,
                name="s1",
                input_preview="a",
                output_preview="b",
                duration_ms=1,
                cost_yuan=0.15,
            ),
            StepLog(
                step_id=2,
                name="s2",
                input_preview="a",
                output_preview="b",
                duration_ms=1,
                cost_yuan=0.25,
            ),
        ]
        run = RunLog(
            run_id="r1",
            user_query="q",
            mode="quick",
            model="m",
            steps=steps,
        )
        assert run.total_cost_yuan == 0.40

    def test_result_path_defaults_to_none(self):
        run = RunLog(run_id="r1", user_query="q", mode="quick", model="m")
        assert run.result_path is None

    def test_empty_steps_default(self):
        run = RunLog(run_id="r1", user_query="q", mode="quick", model="m")
        assert run.steps == []
        assert run.total_tokens_in == 0
        assert run.total_tokens_out == 0
        assert run.total_cost_yuan == 0.0


class TestJsonlSerialization:
    """Tests for JSONL read/write."""

    def test_to_jsonl_produces_valid_line(self):
        step = StepLog(
            step_id=1,
            name="test",
            input_preview="in",
            output_preview="out",
            duration_ms=123,
            tokens_in=10,
            tokens_out=5,
            cost_yuan=0.001,
        )
        line = step.to_jsonl()
        parsed = json.loads(line)
        assert parsed["step_id"] == 1
        assert parsed["name"] == "test"
        assert parsed["tokens_in"] == 10
        assert parsed["cost_yuan"] == 0.001

    def test_runlog_to_jsonl_writes_file(self):
        steps = [
            StepLog(
                step_id=1,
                name="s1",
                input_preview="in",
                output_preview="out",
                duration_ms=1,
                tokens_in=10,
                tokens_out=5,
                cost_yuan=0.001,
            )
        ]
        run = RunLog(
            run_id="run-001",
            user_query="test query",
            mode="quick",
            model="deepseek-v4-flash",
            steps=steps,
            result_path="output/result.md",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(run.to_jsonl())

        path = Path(f.name)
        content = path.read_text(encoding="utf-8")
        lines = [l for l in content.strip().split("\n") if l]

        assert len(lines) == 2  # header line + 1 step line
        header = json.loads(lines[0])
        assert header["type"] == "run"
        assert header["run_id"] == "run-001"
        assert header["user_query"] == "test query"

        step_line = json.loads(lines[1])
        assert step_line["type"] == "step"
        assert step_line["name"] == "s1"

        path.unlink()

    def test_runlog_from_jsonl_roundtrip(self):
        original = RunLog(
            run_id="run-roundtrip",
            user_query="往返测试",
            mode="privacy_enhanced",
            model="deepseek-v4-flash",
            steps=[
                StepLog(
                    step_id=1,
                    name="parse",
                    input_preview="in",
                    output_preview="out",
                    duration_ms=42,
                    tokens_in=100,
                    tokens_out=50,
                    cost_yuan=0.0003,
                ),
                StepLog(
                    step_id=2,
                    name="llm_call",
                    input_preview="prompt",
                    output_preview="response",
                    duration_ms=1500,
                    tokens_in=200,
                    tokens_out=80,
                    cost_yuan=0.00056,
                ),
            ],
            result_path="output/test.md",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(original.to_jsonl())

        path = Path(f.name)
        restored = RunLog.from_jsonl(path)

        assert restored.run_id == original.run_id
        assert restored.user_query == original.user_query
        assert restored.mode == original.mode
        assert restored.model == original.model
        assert len(restored.steps) == 2
        assert restored.steps[0].name == "parse"
        assert restored.steps[0].tokens_in == 100
        assert restored.steps[1].name == "llm_call"
        assert restored.total_tokens_in == 300
        assert restored.total_cost_yuan == pytest.approx(0.00086)
        assert restored.result_path == "output/test.md"

        path.unlink()
