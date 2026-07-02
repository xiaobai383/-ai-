"""Plotly chart factory — interactive dashboard visualizations.

ponytail: direct Plotly calls, no chart config DSL. Upgrade path: add chart
theme config and export-to-static option.
"""
from typing import Dict

import plotly.express as px
import plotly.graph_objects as go


def cost_trend_chart(daily_data: list) -> go.Figure:
    """Line chart: daily cost (yuan) over time.

    Args:
        daily_data: List of DailyStats or dicts with 'date' and 'cost_yuan'.

    Returns:
        Plotly Figure.
    """
    if not daily_data:
        return _empty_chart("暂无数据", "请先执行一些任务")

    dates = [d.date if hasattr(d, 'date') else d.get('date', '') for d in daily_data]
    costs = [d.cost_yuan if hasattr(d, 'cost_yuan') else d.get('cost_yuan', 0) for d in daily_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=costs,
        mode="lines+markers",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=6),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
        name="费用 (¥)",
    ))
    fig.update_layout(
        title="费用趋势（近7天）",
        xaxis_title="日期",
        yaxis_title="费用 (¥)",
        margin=dict(l=40, r=20, t=40, b=40),
        height=250,
    )
    return fig


def mode_pie_chart(mode_dist: Dict[str, int]) -> go.Figure:
    """Pie chart: task mode distribution.

    Args:
        mode_dist: Dict mapping mode name → count.

    Returns:
        Plotly Figure.
    """
    if not mode_dist:
        return _empty_chart("暂无数据", "")

    labels_map = {
        "quick": "快速模式",
        "privacy_enhanced": "隐私增强",
        "manual_confirm": "手动确认",
        "local_fallback": "本地兜底",
    }

    labels = [labels_map.get(k, k) for k in mode_dist.keys()]
    values = list(mode_dist.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.4,
        marker=dict(colors=["#6366f1", "#f59e0b", "#10b981", "#ef4444"]),
    )])
    fig.update_layout(
        title="模式使用分布",
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
    )
    return fig


def model_pie_chart(model_dist: Dict[str, int]) -> go.Figure:
    """Pie chart: model usage distribution.

    Args:
        model_dist: Dict mapping model name → count.

    Returns:
        Plotly Figure.
    """
    if not model_dist:
        return _empty_chart("暂无数据", "")

    fig = go.Figure(data=[go.Pie(
        labels=list(model_dist.keys()),
        values=list(model_dist.values()),
        hole=0.4,
    )])
    fig.update_layout(
        title="模型使用分布",
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
    )
    return fig


def token_bar_chart(daily_data: list) -> go.Figure:
    """Stacked bar chart: daily token input/output.

    Args:
        daily_data: List of DailyStats or dicts.

    Returns:
        Plotly Figure.
    """
    if not daily_data:
        return _empty_chart("暂无数据", "")

    dates = [d.date if hasattr(d, 'date') else d.get('date', '') for d in daily_data]
    tokens_in = [d.tokens_in if hasattr(d, 'tokens_in') else d.get('tokens_in', 0) for d in daily_data]
    tokens_out = [d.tokens_out if hasattr(d, 'tokens_out') else d.get('tokens_out', 0) for d in daily_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=tokens_in, name="输入 Token", marker_color="#6366f1"))
    fig.add_trace(go.Bar(x=dates, y=tokens_out, name="输出 Token", marker_color="#a5b4fc"))
    fig.update_layout(
        title="Token 消耗趋势",
        xaxis_title="日期",
        yaxis_title="Token 数量",
        barmode="stack",
        margin=dict(l=40, r=20, t=40, b=40),
        height=250,
    )
    return fig


def _empty_chart(title: str, subtitle: str = "") -> go.Figure:
    """Return a placeholder chart when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="#9ca3af"),
    )
    if subtitle:
        fig.add_annotation(
            text=subtitle,
            x=0.5, y=0.35,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="#d1d5db"),
        )
    fig.update_layout(
        height=200,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig
