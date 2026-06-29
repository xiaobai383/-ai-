"""基于 APScheduler 的定时任务调度引擎。"""
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.config import AppConfig
from src.scheduler.jobs import JobStore, ScheduledJob

logger = logging.getLogger(__name__)


class SchedulerEngine:
    """定时任务调度引擎。

    封装 APScheduler 的 BackgroundScheduler，管理定时任务的
    注册、暂停、恢复、删除。任务定义通过 JobStore 持久化。

    使用方式:
        engine = SchedulerEngine(config, executor_callback)
        engine.start()
        ...
        engine.stop()
    """

    def __init__(
        self,
        config: AppConfig,
        executor_callback: Optional[Callable[[ScheduledJob], None]] = None,
    ):
        """初始化引擎。

        Args:
            config: AppConfig 实例。
            executor_callback: 任务到期时调用的回调，传入任务定义。
        """
        self.config = config
        self.executor_callback = executor_callback
        self._scheduler = None
        self._running = False
        self._job_store = JobStore(config.scheduler_jobs_dir)
        self._aps_jobs: Dict[str, Any] = {}  # job_id → APScheduler job

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def job_store(self) -> JobStore:
        return self._job_store

    def start(self) -> None:
        """启动调度引擎并从存储加载所有已启用的任务。"""
        if self._running:
            logger.warning("SchedulerEngine 已在运行")
            return

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            self._scheduler = BackgroundScheduler(daemon=True)
            self._running = True

            # 加载并注册所有已启用的任务
            for job in self._job_store.list_all():
                if job.enabled:
                    self._register_job(job)

            self._scheduler.start()
            logger.info("SchedulerEngine 已启动，已加载 %d 个任务", len(self._aps_jobs))

        except ImportError:
            logger.error("APScheduler 未安装。请运行: pip install apscheduler>=3.10.0")
            self._running = False

    def stop(self) -> None:
        """停止调度引擎。"""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
        self._running = False
        self._aps_jobs.clear()
        logger.info("SchedulerEngine 已停止")

    def add_job(self, job: ScheduledJob) -> str:
        """添加并注册新任务。

        Args:
            job: ScheduledJob 定义。

        Returns:
            job_id。
        """
        # 持久化
        self._job_store.save(job)

        # 如果已启用且引擎在运行中，注册到 APScheduler
        if job.enabled and self._running and self._scheduler:
            self._register_job(job)

        logger.info("已添加定时任务: %s (%s)", job.name, job.job_id)
        return job.job_id

    def remove_job(self, job_id: str) -> bool:
        """删除任务。

        Args:
            job_id: 要删除的任务 ID。

        Returns:
            成功返回 True。
        """
        # 从 APScheduler 移除
        if job_id in self._aps_jobs:
            try:
                self._aps_jobs[job_id].remove()
            except Exception:
                pass
            del self._aps_jobs[job_id]

        # 从存储删除
        return self._job_store.delete(job_id)

    def pause_job(self, job_id: str) -> bool:
        """暂停任务。"""
        job_def = self._job_store.get(job_id)
        if job_def is None:
            return False
        job_def.enabled = False
        self._job_store.save(job_def)

        if job_id in self._aps_jobs:
            try:
                self._aps_jobs[job_id].pause()
            except Exception:
                pass

        return True

    def resume_job(self, job_id: str) -> bool:
        """恢复任务。"""
        job_def = self._job_store.get(job_id)
        if job_def is None:
            return False
        job_def.enabled = True
        self._job_store.save(job_def)

        # 重新注册到 APScheduler
        if self._running and self._scheduler:
            if job_id in self._aps_jobs:
                try:
                    self._aps_jobs[job_id].resume()
                except Exception:
                    self._register_job(job_def)

        return True

    def list_jobs(self) -> List[Dict[str, Any]]:
        """列出所有任务及其状态。"""
        jobs = self._job_store.list_all()
        result = []
        for j in jobs:
            entry = j.to_dict()
            entry["is_active"] = j.job_id in self._aps_jobs
            result.append(entry)
        return result

    # ── 内部 ──

    def _register_job(self, job: ScheduledJob) -> None:
        """向 APScheduler 注册单个任务。"""
        if not self._scheduler:
            return

        try:
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            if job.trigger_type == "cron":
                trigger = CronTrigger(**job.trigger_kwargs)
            elif job.trigger_type == "interval":
                trigger = IntervalTrigger(**job.trigger_kwargs)
            else:
                logger.warning("未知触发器类型: %s, 跳过任务 %s", job.trigger_type, job.job_id)
                return

            aps_job = self._scheduler.add_job(
                self._execute_job,
                trigger=trigger,
                args=[job],
                id=job.job_id,
                name=job.name,
                replace_existing=True,
            )
            self._aps_jobs[job.job_id] = aps_job
        except Exception as e:
            logger.error("注册任务 %s 失败: %s", job.job_id, e)

    def _execute_job(self, job: ScheduledJob) -> None:
        """执行定时任务（由 APScheduler 调用）。"""
        # 更新最后运行时间
        job.last_run = datetime.now().isoformat()
        job.last_status = "running"
        self._job_store.save(job)

        try:
            if self.executor_callback:
                self.executor_callback(job)

            job.last_status = "success"
        except Exception as e:
            logger.error("定时任务 %s 执行失败: %s", job.job_id, e)
            job.last_status = f"failed: {e}"

        self._job_store.save(job)
