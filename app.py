"""个人 AI 工作流助手入口。

用法：
    python app.py              — 启动 Gradio UI
    python app.py --api        — 启动 FastAPI 服务
    python app.py --watch      — 启动 Gradio UI 并启用文件夹监听
    python app.py --scheduler  — 启动 Gradio UI 并启用定时任务
    python app.py --api --watch --scheduler  — FastAPI + 监听 + 定时
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

    if "--api" in sys.argv:
        from src.api.server import launch_api
        launch_api(config)
    else:
        from src.ui.gradio_app import launch_ui
        launch_ui(config)


if __name__ == "__main__":
    main()
