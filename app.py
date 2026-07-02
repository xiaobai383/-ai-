"""Entry point for the Personal AI Workflow Assistant.

Usage:
    python app.py              — launch the Gradio UI
    python app.py --api        — launch the FastAPI server
    python app.py --watch      — launch Gradio UI with folder watcher enabled
    python app.py --scheduler  — launch Gradio UI with scheduler enabled
    python app.py --api --watch --scheduler  — FastAPI + watch + scheduler
    python app.py --test       — run all tests
"""
import sys

from src.config import AppConfig


def main():
    if "--test" in sys.argv:
        import pytest
        sys.exit(pytest.main(["tests/", "-v"]))

    config = AppConfig.from_yaml_and_env()

    # v0.3 CLI flags — override config file defaults
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
