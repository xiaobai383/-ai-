"""通知分发 — plyer 桌面通知 + JSONL 日志兜底。"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def notify(
    title: str,
    message: str,
    engine: str = "auto",
    log_file: Optional[str] = None,
) -> bool:
    """发送通知。

    策略：
    - engine="plyer": 仅使用 plyer 桌面通知
    - engine="log": 仅写入 JSONL 日志
    - engine="auto": 优先 plyer，失败则降级到日志

    Args:
        title: 通知标题。
        message: 通知正文。
        engine: 通知引擎选择。
        log_file: JSONL 日志文件路径（若写日志）。

    Returns:
        如果至少一种方式成功则返回 True。
    """
    success = False

    if engine in ("auto", "plyer"):
        try:
            _notify_plyer(title, message)
            success = True
            logger.debug("plyer 通知已发送: %s", title)
        except Exception as e:
            logger.warning("plyer 通知失败: %s", e)

    if engine == "log" or (engine == "auto" and not success):
        try:
            _notify_log(title, message, log_file)
            success = True
        except Exception as e:
            logger.warning("日志通知写入失败: %s", e)

    return success


def _notify_plyer(title: str, message: str) -> None:
    """通过 plyer 发送桌面通知。"""
    from plyer import notification
    notification.notify(
        title=title,
        message=message,
        app_name="AI 工作流助手",
        timeout=5,
    )


def _notify_log(title: str, message: str, log_file: Optional[str] = None) -> None:
    """追加通知到 JSONL 日志文件。"""
    if log_file is None:
        log_file = "data/logs/notifications.jsonl"

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "title": title,
        "message": message,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
