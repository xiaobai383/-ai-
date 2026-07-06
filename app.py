"""Hush 入口。

用法：
    python app.py              — 启动 FastAPI 服务（默认）
    python app.py --watch      — 启动 API 并启用文件夹监听
    python app.py --scheduler  — 启动 API 并启用定时任务
    python app.py --watch --scheduler  — API + 监听 + 定时
    python app.py --test       — 运行所有测试
"""
import logging
import sys

from src.config import AppConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    if "--test" in sys.argv:
        import pytest
        sys.exit(pytest.main(["tests/", "-v"]))

    config = AppConfig.from_yaml_and_env()

    # v0.3 CLI 参数 — 覆盖配置文件中的默认值
    if "--watch" in sys.argv:
        config.watch_enabled = True
    if "--scheduler" in sys.argv:
        config.scheduler_enabled = True

    from src.api.server import launch_api
    launch_api(config)


if __name__ == "__main__":
    main()
