"""RunLog 加载器 — 扫描、加载和过滤执行日志。

ponytail：对 JSONL 文件进行线性扫描，无索引。升级路径：当日志数量超过
约 10,000 个文件时，使用 SQLite 索引加速过滤。
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.observability.run_log import RunLog

logger = logging.getLogger(__name__)


@dataclass
class RunLogSummary:
    """轻量级摘要，用于列表展示，不含完整步骤数据。"""
    run_id: str
    user_query: str
    mode: str
    model: str
    result_path: Optional[str]
    total_tokens_in: int
    total_tokens_out: int
    total_cost_yuan: float
    step_count: int
    fallback: bool
    status: str = "unknown"  # "success" | "failed" | "unknown"


@dataclass
class RunLogList:
    """分页的 RunLog 摘要列表。"""
    items: List[RunLogSummary] = field(default_factory=list)
    total: int = 0


class RunLogLoader:
    """从 data/logs 目录加载和过滤 RunLog。

    用法：
        loader = RunLogLoader("data/logs")
        summaries = loader.list_all()
        full_log = loader.load_by_id("run-abc123")
    """

    def __init__(self, logs_dir: str = "data/logs"):
        self._logs_dir = Path(logs_dir)

    def _scan_files(self) -> List[Path]:
        """返回日志目录中排序后的 JSONL 文件列表。"""
        if not self._logs_dir.exists():
            return []
        return sorted(self._logs_dir.glob("*.jsonl"))

    def list_all(
        self,
        mode_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> RunLogList:
        """列出 RunLog 摘要，支持可选过滤。

        参数：
            mode_filter：按模式过滤（例如 'privacy_enhanced'）。
            status_filter：按状态过滤（'success' | 'failed'）。
            search：在 user_query 中不区分大小写的子串匹配。
            limit：最多返回条数。
            offset：跳过前 N 条。

        返回：
            包含摘要的 RunLogList。
        """
        files = self._scan_files()
        all_summaries: List[RunLogSummary] = []

        for fp in files:
            try:
                run_log = RunLog.from_jsonl(fp)
            except Exception:
                continue

            # 确定状态
            status = "success"
            if run_log.result_path is None:
                status = "failed"
            elif any(s.status == "failed" for s in run_log.steps):
                status = "failed"

            # 应用过滤器
            if mode_filter and run_log.mode != mode_filter:
                continue
            if status_filter and status != status_filter:
                continue
            if search and search.lower() not in run_log.user_query.lower():
                continue

            all_summaries.append(RunLogSummary(
                run_id=run_log.run_id,
                user_query=run_log.user_query,
                mode=run_log.mode,
                model=run_log.model,
                result_path=run_log.result_path,
                total_tokens_in=run_log.total_tokens_in,
                total_tokens_out=run_log.total_tokens_out,
                total_cost_yuan=run_log.total_cost_yuan,
                step_count=len(run_log.steps),
                fallback=run_log.fallback,
                status=status,
            ))

        total = len(all_summaries)
        page = all_summaries[offset:offset + limit]

        return RunLogList(items=page, total=total)

    def load_by_id(self, run_id: str) -> Optional[RunLog]:
        """通过 run_id 加载完整的 RunLog。"""
        for fp in self._scan_files():
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                if run_id in first_line:
                    return RunLog.from_jsonl(fp)
            except Exception:
                continue
        return None

    def count(self) -> int:
        """返回日志文件的总数。"""
        return len(self._scan_files())
