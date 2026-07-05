"""测试定时任务模块。"""
import json
import tempfile
from pathlib import Path

import pytest

from src.config import AppConfig
from src.scheduler.jobs import JobStore, ScheduledJob, PRESET_SCHEDULES


class TestScheduledJob:
    """ScheduledJob dataclass 测试。"""

    def test_defaults(self):
        """默认字段正确。"""
        job = ScheduledJob()
        assert job.job_id.startswith("job-")
        assert job.enabled is True
        assert job.trigger_type == "cron"

    def test_to_dict_and_from_dict_roundtrip(self):
        """序列化/反序列化往返。"""
        job = ScheduledJob(
            name="测试任务",
            query="总结文档",
            trigger_type="cron",
            trigger_kwargs={"hour": 9, "minute": 0},
        )
        data = job.to_dict()
        restored = ScheduledJob.from_dict(data)
        assert restored.name == "测试任务"
        assert restored.query == "总结文档"
        assert restored.trigger_kwargs == {"hour": 9, "minute": 0}

    def test_from_dict_ignores_unknown_keys(self):
        """忽略未知键。"""
        data = {"name": "test", "unknown_field": 123}
        job = ScheduledJob.from_dict(data)
        assert job.name == "test"
        assert not hasattr(job, "unknown_field")


class TestPresetSchedules:
    """预设调度测试。"""

    def test_all_presets_have_trigger(self):
        """所有预设都有 trigger 字段。"""
        for name, preset in PRESET_SCHEDULES.items():
            assert "trigger" in preset, f"预设 {name} 缺少 trigger"
            assert preset["trigger"] in ("cron", "interval"), f"预设 {name} trigger 类型无效"


class TestJobStore:
    """JobStore 持久化测试。"""

    @pytest.fixture
    def store(self, tmp_path):
        """创建临时 JobStore。"""
        return JobStore(str(tmp_path / "jobs"))

    def test_list_empty(self, store):
        """空 store 返回空列表。"""
        assert store.list_all() == []

    def test_save_and_get(self, store):
        """保存后能获取。"""
        job = ScheduledJob(name="test", query="hello")
        store.save(job)
        restored = store.get(job.job_id)
        assert restored is not None
        assert restored.name == "test"

    def test_save_and_list(self, store):
        """保存后出现在列表中。"""
        job = ScheduledJob(name="job1")
        store.save(job)
        jobs = store.list_all()
        assert len(jobs) == 1
        assert jobs[0].name == "job1"

    def test_delete(self, store):
        """删除后 get 返回 None。"""
        job = ScheduledJob(name="to_delete")
        store.save(job)
        assert store.delete(job.job_id) is True
        assert store.get(job.job_id) is None

    def test_delete_nonexistent(self, store):
        """删除不存在的任务返回 False。"""
        assert store.delete("nonexistent") is False

    def test_jobs_dir_created(self, tmp_path):
        """store 自动创建目录。"""
        jobs_dir = tmp_path / "auto_created" / "jobs"
        store = JobStore(str(jobs_dir))
        assert jobs_dir.exists()
