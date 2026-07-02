"""文件夹监听器 — 监控目录变更并自动触发 run_task。"""
import fnmatch
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
from queue import Queue

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class WatchEvent:
    """一次文件变更事件。"""
    event_type: str   # 创建、修改
    path: str
    timestamp: float = field(default_factory=time.time)


class FolderWatcher:
    """基于 watchdog 的文件夹监听器。

    支持两种触发模式：
    - instant: 每次文件变更立即触发处理
    - batch: 在时间窗口内累积文件，窗口到期后批量触发

    使用方式:
        watcher = FolderWatcher(config, callback)
        watcher.start()
        ...
        watcher.stop()
    """

    def __init__(
        self,
        config: AppConfig,
        callback: Optional[Callable[[List[str]], None]] = None,
    ):
        """初始化监听器。

        Args:
            config: 包含监听设置的 AppConfig。
            callback: 当触发时调用的回调，接收文件路径列表。
        """
        self.config = config
        self.callback = callback
        self._observer = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._batch_queue: Queue = Queue()
        self._batch_timer: Optional[threading.Timer] = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """启动文件夹监听。

        为每个配置的目录启动 observer。
        如果 trigger_mode 是 batch，同时启动批量累积定时器。
        """
        if self._running:
            logger.warning("FolderWatcher 已在运行")
            return

        self._running = True

        # 确保目录存在
        for watch_dir in self.config.watch_dirs:
            Path(watch_dir).mkdir(parents=True, exist_ok=True)

        # 启动 watchdog observer 线程
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="folder-watcher"
        )
        self._thread.start()

        # 批量模式：启动定时 flush
        if self.config.watch_trigger_mode == "batch":
            self._start_batch_timer()

        logger.info(
            "FolderWatcher 已启动: dirs=%s, mode=%s",
            self.config.watch_dirs,
            self.config.watch_trigger_mode,
        )

    def stop(self) -> None:
        """停止文件夹监听。"""
        self._running = False

        if self._batch_timer:
            self._batch_timer.cancel()
            self._batch_timer = None

        # 批量模式：flush 剩余文件
        if self.config.watch_trigger_mode == "batch":
            self._flush_batch()

        logger.info("FolderWatcher 已停止")

    def trigger_now(self) -> None:
        """手动立即触发处理（flush 当前批次）。"""
        self._flush_batch()

    # ── 内部逻辑 ──

    def _watch_loop(self) -> None:
        """主监听循环 — 轮询检查目录中的新文件。

        使用轮询方式而非 watchdog 库，避免额外依赖问题。
        每 2 秒扫描配置的目录，比较文件修改时间来检测变更。
        """
        # 记录已知文件及其修改时间
        known_files: Dict[str, float] = {}

        while self._running:
            for watch_dir in self.config.watch_dirs:
                dir_path = Path(watch_dir)
                if not dir_path.exists():
                    continue

                for file_path in dir_path.iterdir():
                    if not file_path.is_file():
                        continue
                    if not self._matches_pattern(file_path.name):
                        continue

                    path_str = str(file_path)
                    mtime = file_path.stat().st_mtime

                    prev_mtime = known_files.get(path_str)
                    if prev_mtime is None:
                        # 新文件
                        known_files[path_str] = mtime
                        self._on_event(WatchEvent("created", path_str))
                    elif mtime > prev_mtime:
                        # 已修改
                        known_files[path_str] = mtime
                        self._on_event(WatchEvent("modified", path_str))

            time.sleep(2)

    def _matches_pattern(self, filename: str) -> bool:
        """检查文件名是否匹配配置的模式。"""
        if not self.config.watch_patterns:
            return True
        for pattern in self.config.watch_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def _on_event(self, event: WatchEvent) -> None:
        """处理单个文件事件。"""
        logger.debug("检测到文件事件: %s %s", event.event_type, event.path)

        if self.config.watch_trigger_mode == "instant":
            if self.callback:
                self.callback([event.path])
        else:
            # batch 模式：加入队列
            self._batch_queue.put(event)

    def _start_batch_timer(self) -> None:
        """启动批量累积的定时器。"""
        window = self.config.watch_batch_window_seconds
        if window <= 0:
            window = 60

        self._batch_timer = threading.Timer(window, self._on_batch_tick)
        self._batch_timer.daemon = True
        self._batch_timer.start()

    def _on_batch_tick(self) -> None:
        """批量定时器到期 — flush 并重新调度。"""
        self._flush_batch()
        if self._running:
            self._start_batch_timer()

    def _flush_batch(self) -> None:
        """取出队列中所有待处理的文件并触发回调。"""
        paths: List[str] = []
        while not self._batch_queue.empty():
            try:
                event: WatchEvent = self._batch_queue.get_nowait()
                if event.path not in paths:
                    paths.append(event.path)
            except Exception:
                break

        if paths and self.callback:
            logger.info("批量触发: %d 个文件", len(paths))
            self.callback(paths)


# ── 便捷函数 ──

_active_watcher: Optional[FolderWatcher] = None


def start_watcher(config: AppConfig, callback: Callable[[List[str]], None]) -> FolderWatcher:
    """创建并启动全局监听器。

    Args:
        config: AppConfig。
        callback: 文件变更回调。

    Returns:
        正在运行的 FolderWatcher。
    """
    global _active_watcher
    if _active_watcher is not None and _active_watcher.is_running:
        _active_watcher.stop()
    _active_watcher = FolderWatcher(config, callback)
    _active_watcher.start()
    return _active_watcher


def stop_watcher() -> None:
    """停止全局监听器。"""
    global _active_watcher
    if _active_watcher is not None:
        _active_watcher.stop()
        _active_watcher = None
