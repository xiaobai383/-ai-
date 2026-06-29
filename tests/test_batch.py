"""测试批量处理模块。"""
import tempfile
from pathlib import Path

import pytest

from src.config import AppConfig
from src.batch.runner import BatchRunner, BatchReport


@pytest.fixture
def batch_config(tmp_path):
    """创建带临时目录的测试配置。"""
    config = AppConfig()
    config.allowed_paths = [str(tmp_path), "data/", "output/"]
    config.max_file_size_mb = 100
    return config


@pytest.fixture
def sample_files(batch_config, tmp_path):
    """创建测试用文件。"""
    files = []
    for i in range(3):
        f = tmp_path / f"test_{i}.txt"
        f.write_text(f"这是测试文件 {i} 的内容。\n包含第二行。", encoding="utf-8")
        files.append(str(f))
    return files


class TestBatchReport:
    """BatchReport 测试。"""

    def test_success_rate(self):
        """成功率计算正确。"""
        report = BatchReport(batch_id="test", total_files=4, completed=3, failed=1)
        assert report.success_rate == 0.75

    def test_success_rate_zero_files(self):
        """零文件时成功率为 0。"""
        report = BatchReport(batch_id="test", total_files=0)
        assert report.success_rate == 0.0

    def test_fields_default(self):
        """默认字段。"""
        report = BatchReport(batch_id="b1")
        assert report.total_files == 0
        assert report.completed == 0
        assert report.failed == 0
        assert report.total_cost_yuan == 0.0


class TestBatchRunner:
    """BatchRunner 功能测试。"""

    def test_runner_creation(self, batch_config):
        """创建 BatchRunner。"""
        runner = BatchRunner(batch_config)
        assert runner.config is batch_config

    def test_run_with_all_valid_files(self, batch_config, sample_files):
        """全部文件有效时批量处理成功。"""
        runner = BatchRunner(batch_config)
        report = runner.run(
            query="总结文档",
            files=sample_files,
            mode="quick",
            auto_confirm=True,
        )
        assert report.total_files == 3
        assert report.completed == 3
        assert report.failed == 0
        assert report.total_duration_ms >= 0
        assert len(report.runs) == 3
        assert len(report.errors) == 0

    def test_run_with_mixed_files(self, batch_config, sample_files, tmp_path):
        """混合有效和无效文件。"""
        # 添加一个不存在的文件
        all_files = sample_files + [str(tmp_path / "nonexistent.txt")]
        runner = BatchRunner(batch_config)
        report = runner.run(
            query="测试",
            files=all_files,
            mode="quick",
            auto_confirm=True,
        )
        assert report.total_files == 4
        assert report.completed == 3
        assert report.failed == 1
        assert len(report.errors) == 1

    def test_run_empty_files(self, batch_config):
        """空文件列表。"""
        runner = BatchRunner(batch_config)
        report = runner.run(query="测试", files=[], auto_confirm=True)
        assert report.total_files == 0
        assert report.completed == 0

    def test_progress_callback(self, batch_config, sample_files):
        """进度回调被调用。"""
        calls = []

        def progress_cb(current, total, current_file):
            calls.append((current, total, current_file))

        runner = BatchRunner(batch_config, progress_callback=progress_cb)
        runner.run(query="测试", files=sample_files, auto_confirm=True)
        assert len(calls) == 3
        assert calls[0] == (1, 3, sample_files[0])
        assert calls[2] == (3, 3, sample_files[2])
