"""Dashboard module — aggregate stats and interactive Plotly charts."""
from src.dashboard.aggregator import DailyStats, DashboardAggregator, RunStats
from src.dashboard.charts import (
    cost_trend_chart,
    mode_pie_chart,
    model_pie_chart,
    token_bar_chart,
)

__all__ = [
    "DashboardAggregator",
    "RunStats",
    "DailyStats",
    "cost_trend_chart",
    "mode_pie_chart",
    "model_pie_chart",
    "token_bar_chart",
]
