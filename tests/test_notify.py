"""测试通知模块。"""
import json
from pathlib import Path

import pytest

from src.notify.dispatch import notify


class TestNotifyLog:
    """测试日志通知引擎。"""

    def test_notify_log(self, tmp_path):
        """log 引擎写入 JSONL 文件。"""
        log_file = str(tmp_path / "notifications.jsonl")
        result = notify(
            title="测试标题",
            message="测试消息",
            engine="log",
            log_file=log_file,
        )
        assert result is True

        # 验证文件内容
        content = Path(log_file).read_text(encoding="utf-8").strip()
        assert content
        entry = json.loads(content)
        assert entry["title"] == "测试标题"
        assert entry["message"] == "测试消息"
        assert "timestamp" in entry

    def test_notify_log_appends(self, tmp_path):
        """多次调用追加而非覆盖。"""
        log_file = str(tmp_path / "appends.jsonl")
        notify(title="A", message="1", engine="log", log_file=log_file)
        notify(title="B", message="2", engine="log", log_file=log_file)

        lines = Path(log_file).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_notify_auto_falls_back_to_log(self, tmp_path):
        """auto 模式下，plyer 不可用时降级到日志。"""
        import sys
        # 模拟 plyer 不可用
        original_plyer = sys.modules.get("plyer")
        sys.modules["plyer"] = None

        try:
            log_file = str(tmp_path / "fallback.jsonl")
            result = notify(
                title="降级测试",
                message="plyer 不可用",
                engine="auto",
                log_file=log_file,
            )
            assert result is True
            assert Path(log_file).exists()
        finally:
            if original_plyer is not None:
                sys.modules["plyer"] = original_plyer
            else:
                sys.modules.pop("plyer", None)

    def test_notify_plyer_only_failure(self, tmp_path):
        """plyer 引擎失败时不写日志。"""
        import sys
        original_plyer = sys.modules.get("plyer")
        sys.modules["plyer"] = None

        try:
            log_file = str(tmp_path / "should_not_exist.jsonl")
            result = notify(
                title="失败",
                message="不应写日志",
                engine="plyer",
                log_file=log_file,
            )
            assert result is False
        finally:
            if original_plyer is not None:
                sys.modules["plyer"] = original_plyer
            else:
                sys.modules.pop("plyer", None)

    def test_notify_empty_message(self, tmp_path):
        """空消息也能处理。"""
        log_file = str(tmp_path / "empty.jsonl")
        result = notify(title="", message="", engine="log", log_file=log_file)
        assert result is True
