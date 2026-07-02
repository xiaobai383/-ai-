"""仪表盘数据汇总器 — 从 RunLog JSONL 文件计算统计数据。

ponytail：每次请求对所有日志文件进行 O(n) 扫描。升级路径：当日志数量超过
约 5,000 时，在保存时缓存统计数据（增量更新）。
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from src.observability.run_log import RunLog

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """每日汇总统计。"""
    date: str                     # YYYY-MM-DD
    task_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_yuan: float = 0.0
    duration_ms: int = 0


@dataclass
class RunStats:
    """所有 RunLog 的汇总统计。"""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_yuan: float = 0.0
    total_duration_ms: int = 0
    fallback_count: int = 0

    # 分布
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    model_distribution: Dict[str, int] = field(default_factory=dict)

    # 时间序列
    daily: List[DailyStats] = field(default_factory=list)

    # 最近任务
    recent_tasks: List[dict] = field(default_factory=list)


class DashboardAggregator:
    """汇总 RunLog 统计数据，支持可选时间范围过滤。

    用法：
        agg = DashboardAggregator("data/logs")
        stats = agg.aggregate(days=7)  # 最近 7 天
    """

    def __init__(self, logs_dir: str = "data/logs"):
        self._logs_dir = Path(logs_dir)

    def aggregate(self, days: int = 7) -> RunStats:
        """从所有日志计算统计数据，可选过滤最近 N 天。

        参数：
            days：只包含最近 N 天的日志。0 = 全部时间。

        返回：
            RunStats 汇总。
        """
        cutoff = None
        if days > 0:
            cutoff = datetime.now() - timedelta(days=days)

        files = sorted(self._logs_dir.glob("*.jsonl")) if self._logs_dir.exists() else []
        stats = RunStats()
        daily_map: Dict[str, DailyStats] = {}

        for fp in files:
            # 从文件修改时间确定文件日期
            try:
                mtime = datetime.fromtimestamp(fp.stat().st_mtime)
            except OSError:
                mtime = datetime.now()

            if cutoff and mtime < cutoff:
                continue

            date_key = mtime.strftime("%Y-%m-%d")

            try:
                run_log = RunLog.from_jsonl(fp)
            except Exception:
                continue

            is_success = run_log.result_path is not None and all(
                s.status == "success" for s in run_log.steps
            )

            stats.total_tasks += 1
            if is_success:
                stats.successful_tasks += 1
            else:
                stats.failed_tasks += 1

            stats.total_tokens_in += run_log.total_tokens_in
            stats.total_tokens_out += run_log.total_tokens_out
            stats.total_cost_yuan += run_log.total_cost_yuan

            step_duration = sum(s.duration_ms for s in run_log.steps)
            stats.total_duration_ms += step_duration

            if run_log.fallback:
                stats.fallback_count += 1

            # 模式分布
            mode = run_log.mode or "unknown"
            stats.mode_distribution[mode] = stats.mode_distribution.get(mode, 0) + 1

            # 模型分布
            model = run_log.model or "unknown"
            stats.model_distribution[model] = stats.model_distribution.get(model, 0) + 1

            # 每日汇总
            if date_key not in daily_map:
                daily_map[date_key] = DailyStats(date=date_key)
            ds = daily_map[date_key]
            ds.task_count += 1
            ds.tokens_in += run_log.total_tokens_in
            ds.tokens_out += run_log.total_tokens_out
            ds.cost_yuan += run_log.total_cost_yuan
            ds.duration_ms += step_duration

        # 按时间顺序排列每日统计数据
        stats.daily = sorted(daily_map.values(), key=lambda d: d.date)

        # 最近任务（最近 10 个）
        recent = []
        for fp in reversed(files[-20:]):  # 检查最近 20 个，保留 10 个
            try:
                run_log = RunLog.from_jsonl(fp)
            except Exception:
                continue
            recent.append({
                "run_id": run_log.run_id,
                "user_query": run_log.user_query[:60],
                "mode": run_log.mode,
                "model": run_log.model,
                "cost_yuan": round(run_log.total_cost_yuan, 4),
                "fallback": run_log.fallback,
                "status": "success" if run_log.result_path else "failed",
            })
            if len(recent) >= 10:
                break
        stats.recent_tasks = recent

        return stats
