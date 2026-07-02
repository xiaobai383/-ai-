"""定时任务定义与持久化。"""
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── 预设调度模板 ──

PRESET_SCHEDULES: Dict[str, Dict[str, Any]] = {
    "every_hour": {
        "label": "每小时",
        "trigger": "interval",
        "hours": 1,
    },
    "every_3_hours": {
        "label": "每3小时",
        "trigger": "interval",
        "hours": 3,
    },
    "daily_9am": {
        "label": "每天 9:00",
        "trigger": "cron",
        "hour": 9,
        "minute": 0,
    },
    "daily_6pm": {
        "label": "每天 18:00",
        "trigger": "cron",
        "hour": 18,
        "minute": 0,
    },
    "weekday_morning": {
        "label": "工作日 8:00",
        "trigger": "cron",
        "day_of_week": "mon-fri",
        "hour": 8,
        "minute": 0,
    },
    "weekly_monday": {
        "label": "每周一 9:00",
        "trigger": "cron",
        "day_of_week": "mon",
        "hour": 9,
        "minute": 0,
    },
}


@dataclass
class ScheduledJob:
    """一个定时任务的完整定义。"""

    job_id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""

    # 调度: 预设名称或原始 cron/interval 参数
    schedule_preset: Optional[str] = None   # 预设名，如 "daily_9am"
    trigger_type: str = "cron"              # cron 或 interval
    trigger_kwargs: Dict[str, Any] = field(default_factory=dict)

    # 执行内容
    query: str = ""                         # 用户提示词
    mode: str = "privacy_enhanced"
    file_paths: List[str] = field(default_factory=list)
    workflow_template: Optional[str] = None
    output_format: str = "markdown"

    # 状态
    enabled: bool = True
    last_run: Optional[str] = None
    last_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledJob":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class JobStore:
    """管理定时任务的持久化存储。

    每个任务保存为独立的 JSON 文件，存放在调度任务目录中。
    """

    def __init__(self, jobs_dir: str | Path | None = None):
        if jobs_dir is None:
            jobs_dir = Path("data/scheduled_jobs")
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def list_all(self) -> List[ScheduledJob]:
        """列出所有已保存的任务。"""
        jobs = []
        for f in sorted(self.jobs_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                jobs.append(ScheduledJob.from_dict(data))
            except Exception:
                continue
        return jobs

    def get(self, job_id: str) -> Optional[ScheduledJob]:
        """按 ID 获取任务。"""
        path = self._job_path(job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ScheduledJob.from_dict(data)
        except Exception:
            return None

    def save(self, job: ScheduledJob) -> None:
        """保存（创建或更新）任务。"""
        path = self._job_path(job.job_id)
        path.write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def delete(self, job_id: str) -> bool:
        """删除任务。成功返回 True。"""
        path = self._job_path(job_id)
        if path.exists():
            path.unlink()
            return True
        return False

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> ScheduledJob:
        """从预设模板创建 ScheduledJob。

        Args:
            preset_name: PRESET_SCHEDULES 中的键之一。
            **overrides: 覆盖默认字段（如 query, file_paths）。

        Returns:
            已配置的 ScheduledJob。

        Raises:
            ValueError: 如果预设名称无效。
        """
        preset = PRESET_SCHEDULES.get(preset_name)
        if not preset:
            raise ValueError(
                f"无效预设 '{preset_name}'。可用: {list(PRESET_SCHEDULES)}"
            )

        trigger_type = preset["trigger"]
        trigger_kwargs = {k: v for k, v in preset.items() if k not in ("label", "trigger")}

        job = ScheduledJob(
            name=preset["label"],
            schedule_preset=preset_name,
            trigger_type=trigger_type,
            trigger_kwargs=trigger_kwargs,
        )

        # 应用覆盖
        for k, v in overrides.items():
            if hasattr(job, k):
                setattr(job, k, v)

        return job
