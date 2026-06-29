"""测试文件夹监听模块。"""
import tempfile
import time
from pathlib import Path

import pytest

from src.config import AppConfig
from src.monitor.watcher import FolderWatcher


@pytest.fixture
def watch_config(tmp_path):
    """创建带临时目录的测试配置。"""
    config = AppConfig()
    config.watch_dirs = [str(tmp_path / "watch")]
    config.watch_patterns = ["*.txt", "*.md"]
    config.watch_trigger_mode = "instant"
    return config


class TestFolderWatcher:
    """FolderWatcher 功能测试。"""

    def test_initial_state(self, watch_config):
        """初始状态：未运行。"""
        watcher = FolderWatcher(watch_config)
        assert not watcher.is_running

    def test_match_pattern(self, watch_config):
        """模式匹配测试。"""
        watcher = FolderWatcher(watch_config)
        assert watcher._matches_pattern("doc.txt")
        assert watcher._matches_pattern("readme.md")
        assert not watcher._matches_pattern("image.png")
        assert not watcher._matches_pattern("data.json")

    def test_match_all_when_no_patterns(self, watch_config):
        """无模式时匹配所有文件。"""
        watch_config.watch_patterns = []
        watcher = FolderWatcher(watch_config)
        assert watcher._matches_pattern("anything.bin")

    def test_watch_dir_created(self, watch_config):
        """启动时自动创建监控目录。"""
        watch_dir = Path(watch_config.watch_dirs[0])
        # 确保目录不存在
        if watch_dir.exists():
            import shutil
            shutil.rmtree(watch_dir)

        watcher = FolderWatcher(watch_config)
        watcher.start()
        assert watch_dir.exists()
        watcher.stop()

    def test_instant_callback(self, watch_config):
        """instant 模式下立即触发回调。"""
        called_files = []

        def callback(files):
            called_files.extend(files)

        watcher = FolderWatcher(watch_config, callback=callback)
        # 直接模拟事件
        from src.monitor.watcher import WatchEvent
        watcher._on_event(WatchEvent("created", "/tmp/test.txt"))
        assert "/tmp/test.txt" in called_files

    def test_batch_callback(self, watch_config):
        """batch 模式下，事件加入队列，flush 后才触���回调。"""
        watch_config.watch_trigger_mode = "batch"
        called_files = []

        def callback(files):
            called_files.extend(files)

        watcher = FolderWatcher(watch_config, callback=callback)

        from src.monitor.watcher import WatchEvent
        watcher._on_event(WatchEvent("created", "/tmp/a.txt"))
        watcher._on_event(WatchEvent("created", "/tmp/b.txt"))

        # callback 尚未触发
        assert len(called_files) == 0

        # flush 触发
        watcher._flush_batch()
        assert "/tmp/a.txt" in called_files
        assert "/tmp/b.txt" in called_files

    def test_stop_cleans_up(self, watch_config):
        """停止后 is_running 为 False。"""
        watcher = FolderWatcher(watch_config)
        watcher._running = True
        watcher.stop()
        assert not watcher.is_running


class TestStartStopWatcher:
    """start_watcher / stop_watcher 全局函数测试。"""

    def test_start_watcher(self, watch_config, tmp_path):
        """start_watcher 创建并启动监听器。"""
        import src.monitor.watcher as watcher_mod

        watch_config.watch_dirs = [str(tmp_path / "w")]
        watcher_mod.start_watcher(watch_config, lambda f: None)
        assert watcher_mod._active_watcher is not None
        assert watcher_mod._active_watcher.is_running
        watcher_mod.stop_watcher()
        assert watcher_mod._active_watcher is None
