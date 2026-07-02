"""Tests for src.dashboard — aggregator and charts."""
import tempfile
from pathlib import Path

import pytest

from src.dashboard.aggregator import DailyStats, DashboardAggregator, RunStats
from src.dashboard.charts import (
    cost_trend_chart,
    mode_pie_chart,
    model_pie_chart,
    token_bar_chart,
)
from src.observability.run_log import RunLog, StepLog


class TestDashboardAggregator:
    """Stats aggregation from RunLog JSONL files."""

    @pytest.fixture
    def logs_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)

            log1 = RunLog(run_id="r1", user_query="q1", mode="quick", model="m1", result_path="out/r1.md")
            log1.steps.append(StepLog(step_id=1, name="s1", input_preview="in", output_preview="out", duration_ms=100, tokens_in=100, tokens_out=50, cost_yuan=0.01))
            (d / "r1.jsonl").write_text(log1.to_jsonl(), encoding="utf-8")

            log2 = RunLog(run_id="r2", user_query="q2", mode="privacy_enhanced", model="m1", result_path=None)
            log2.steps.append(StepLog(step_id=1, name="s1", input_preview="in", output_preview="err", duration_ms=200, status="failed"))
            (d / "r2.jsonl").write_text(log2.to_jsonl(), encoding="utf-8")

            log3 = RunLog(run_id="r3", user_query="q3", mode="privacy_enhanced", model="m2", result_path="out/r3.md", fallback=True)
            log3.steps.append(StepLog(step_id=1, name="s1", input_preview="in", output_preview="out", duration_ms=50, tokens_in=200, tokens_out=80, cost_yuan=0.02))
            (d / "r3.jsonl").write_text(log3.to_jsonl(), encoding="utf-8")

            yield d

    def test_aggregate_all(self, logs_dir):
        agg = DashboardAggregator(str(logs_dir))
        stats = agg.aggregate(days=0)
        assert stats.total_tasks == 3
        assert stats.successful_tasks == 2
        assert stats.failed_tasks == 1
        assert stats.fallback_count == 1

    def test_aggregate_tokens(self, logs_dir):
        agg = DashboardAggregator(str(logs_dir))
        stats = agg.aggregate(days=0)
        assert stats.total_tokens_in == 300  # 100 + 0 + 200
        assert stats.total_tokens_out == 130  # 50 + 0 + 80
        assert stats.total_cost_yuan == 0.03  # 0.01 + 0 + 0.02

    def test_mode_distribution(self, logs_dir):
        agg = DashboardAggregator(str(logs_dir))
        stats = agg.aggregate(days=0)
        assert stats.mode_distribution["quick"] == 1
        assert stats.mode_distribution["privacy_enhanced"] == 2

    def test_model_distribution(self, logs_dir):
        agg = DashboardAggregator(str(logs_dir))
        stats = agg.aggregate(days=0)
        assert stats.model_distribution["m1"] == 2
        assert stats.model_distribution["m2"] == 1

    def test_recent_tasks(self, logs_dir):
        agg = DashboardAggregator(str(logs_dir))
        stats = agg.aggregate(days=0)
        assert len(stats.recent_tasks) <= 10
        assert len(stats.recent_tasks) >= 3

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            agg = DashboardAggregator(tmp)
            stats = agg.aggregate(days=0)
            assert stats.total_tasks == 0
            assert len(stats.daily) == 0
            assert len(stats.recent_tasks) == 0

    def test_daily_stats_dataclass(self):
        ds = DailyStats(date="2025-01-15", task_count=5, tokens_in=1000, tokens_out=500, cost_yuan=0.05)
        assert ds.date == "2025-01-15"
        assert ds.task_count == 5

    def test_runstats_defaults(self):
        rs = RunStats()
        assert rs.total_tasks == 0
        assert rs.mode_distribution == {}
        assert rs.model_distribution == {}


class TestCharts:
    """Chart factory functions."""

    def test_cost_trend_with_data(self):
        data = [
            DailyStats(date="2025-01-01", cost_yuan=0.01),
            DailyStats(date="2025-01-02", cost_yuan=0.03),
        ]
        fig = cost_trend_chart(data)
        assert fig is not None
        assert len(fig.data) > 0

    def test_cost_trend_empty(self):
        fig = cost_trend_chart([])
        assert fig is not None

    def test_mode_pie_with_data(self):
        fig = mode_pie_chart({"quick": 5, "privacy_enhanced": 3})
        assert fig is not None
        assert len(fig.data) > 0

    def test_mode_pie_empty(self):
        fig = mode_pie_chart({})
        assert fig is not None

    def test_model_pie_with_data(self):
        fig = model_pie_chart({"m1": 10})
        assert fig is not None

    def test_model_pie_empty(self):
        fig = model_pie_chart({})
        assert fig is not None

    def test_token_bar_with_data(self):
        data = [
            DailyStats(date="2025-01-01", tokens_in=100, tokens_out=50),
            DailyStats(date="2025-01-02", tokens_in=200, tokens_out=80),
        ]
        fig = token_bar_chart(data)
        assert fig is not None
        assert len(fig.data) > 0

    def test_token_bar_empty(self):
        fig = token_bar_chart([])
        assert fig is not None
