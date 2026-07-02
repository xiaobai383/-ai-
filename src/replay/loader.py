"""RunLog loader — scan, load, and filter execution logs.

ponytail: linear scan of JSONL files, no index. Upgrade path: SQLite-backed
index for fast filtering when log count exceeds ~10,000 files.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.observability.run_log import RunLog

logger = logging.getLogger(__name__)


@dataclass
class RunLogSummary:
    """Lightweight summary for listing, without full step data."""
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
    """Paginated list of RunLog summaries."""
    items: List[RunLogSummary] = field(default_factory=list)
    total: int = 0


class RunLogLoader:
    """Load and filter RunLogs from the data/logs directory.

    Usage:
        loader = RunLogLoader("data/logs")
        summaries = loader.list_all()
        full_log = loader.load_by_id("run-abc123")
    """

    def __init__(self, logs_dir: str = "data/logs"):
        self._logs_dir = Path(logs_dir)

    def _scan_files(self) -> List[Path]:
        """Return sorted list of JSONL files in the logs directory."""
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
        """List RunLog summaries with optional filters.

        Args:
            mode_filter: Filter by mode (e.g. 'privacy_enhanced').
            status_filter: Filter by status ('success' | 'failed').
            search: Case-insensitive substring match on user_query.
            limit: Max items to return.
            offset: Skip first N items.

        Returns:
            RunLogList with summaries.
        """
        files = self._scan_files()
        all_summaries: List[RunLogSummary] = []

        for fp in files:
            try:
                run_log = RunLog.from_jsonl(fp)
            except Exception:
                continue

            # Determine status
            status = "success"
            if run_log.result_path is None:
                status = "failed"
            elif any(s.status == "failed" for s in run_log.steps):
                status = "failed"

            # Apply filters
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
        """Load a full RunLog by its run_id."""
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
        """Return total number of log files."""
        return len(self._scan_files())
