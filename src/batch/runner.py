"""批量处理执行器 — 串行执行多个 run_task 并聚合结果。"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from src.agent.app import run_task
from src.config import AppConfig
from src.observability.run_log import RunLog

logger = logging.getLogger(__name__)


@dataclass
class BatchReport:
    """一次批量处理的聚合报告。"""

    batch_id: str
    total_files: int = 0
    completed: int = 0
    failed: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cost_yuan: float = 0.0
    total_duration_ms: int = 0
    runs: List[RunLog] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.completed / self.total_files


class BatchRunner:
    """批量处理执行器。

    接受多个文件路径，串行执行 run_task，聚合所有结果。

    使用方式:
        runner = BatchRunner(config)
        report = runner.run(query="总结文档", files=["a.txt", "b.txt"])
    """

    def __init__(
        self,
        config: AppConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """初始化执行器。

        Args:
            config: AppConfig 实例。
            progress_callback: 可选进度回调 (current, total, current_file)。
        """
        self.config = config
        self.progress_callback = progress_callback

    def run(
        self,
        query: str,
        files: List[str],
        mode: str = "privacy_enhanced",
        llm: Any = None,
        auto_confirm: bool = True,
    ) -> BatchReport:
        """串行执行批量文件处理。

        Args:
            query: 用户提示词。
            files: 要处理的文件路径列表。
            mode: 处理模式。
            llm: 可重用的 LLM 实例。
            auto_confirm: 跳过用户确认。

        Returns:
            聚合的 BatchReport。
        """
        import uuid

        batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        report = BatchReport(batch_id=batch_id, total_files=len(files))

        t0 = time.time()

        for i, file_path in enumerate(files):
            if self.progress_callback:
                self.progress_callback(i + 1, len(files), file_path)

            logger.info("批量处理 [%d/%d]: %s", i + 1, len(files), file_path)

            try:
                run_log = run_task(
                    query=query,
                    files=[file_path],
                    mode=mode,
                    config=self.config,
                    llm=llm,
                    auto_confirm=auto_confirm,
                    suppress_notification=True,
                )

                report.runs.append(run_log)
                if run_log.result_path:
                    report.completed += 1
                    report.total_tokens_in += run_log.total_tokens_in
                    report.total_tokens_out += run_log.total_tokens_out
                    report.total_cost_yuan += run_log.total_cost_yuan
                else:
                    report.failed += 1
                    report.errors.append(f"{file_path}: 未生成结果文件")

            except Exception as e:
                report.failed += 1
                report.errors.append(f"{file_path}: {str(e)}")
                logger.error("批量处理文件失败 %s: %s", file_path, e)

        report.total_duration_ms = int((time.time() - t0) * 1000)

        logger.info(
            "批量处理完成: %d/%d 成功, tokens=%d, cost=¥%.6f",
            report.completed,
            report.total_files,
            report.total_tokens_in + report.total_tokens_out,
            report.total_cost_yuan,
        )

        # v0.3: 批量完成后发一条汇总通知
        _send_batch_notification(report, self.config)

        return report


def _send_batch_notification(report: BatchReport, config) -> None:
    """v0.3: 批量完成后发送汇总通知。"""
    if not config.notifications_enabled:
        return
    try:
        from src.notify.dispatch import notify

        title = "📦 批量处理完成"
        message = (
            f"{report.batch_id}\n"
            f"成功 {report.completed}/{report.total_files}，失败 {report.failed}\n"
            f"Token: {report.total_tokens_in}↗ / {report.total_tokens_out}↘\n"
            f"费用: ¥{report.total_cost_yuan:.6f}"
        )
        notify(
            title=title,
            message=message,
            engine=config.notifications_engine,
            log_file=config.notifications_log_file,
        )
    except Exception:
        pass
