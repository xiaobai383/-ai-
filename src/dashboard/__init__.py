"""仪表盘模块 — 汇总统计与交互式 Plotly 图表。"""
from src.dashboard.aggregator import DailyStats, DashboardAggregator, RunStats
from src.dashboard.charts import (
    cost_trend_chart,
    mode_pie_chart,
    model_pie_chart,
)

__all__ = [
    "DashboardAggregator",
    "RunStats",
    "DailyStats",
    "cost_trend_chart",
    "mode_pie_chart",
    "model_pie_chart",
]
