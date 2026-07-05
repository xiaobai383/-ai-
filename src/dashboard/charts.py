"""Plotly 图表工厂 — 交互式仪表盘可视化。

ponytail：直接调用 Plotly，无图表配置 DSL。升级路径：添加图表主题配置
和静态导出选项。
"""
from typing import Dict

import plotly.express as px
import plotly.graph_objects as go


def cost_trend_chart(daily_data: list) -> go.Figure:
    """折线图：每日费用（元）随时间变化。

    参数：
        daily_data：DailyStats 列表或包含 'date' 和 'cost_yuan' 的字典列表。

    返回：
        Plotly Figure。
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
    """饼图：任务模式分布。

    参数：
        mode_dist：模式名称 → 数量的字典。

    返回：
        Plotly Figure。
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
    """饼图：模型使用分布。

    参数：
        model_dist：模型名称 → 数量的字典。

    返回：
        Plotly Figure。
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


def _empty_chart(title: str, subtitle: str = "") -> go.Figure:
    """当没有数据时返回占位图表。"""
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
