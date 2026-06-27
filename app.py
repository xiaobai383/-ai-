"""Entry point for the Personal AI Workflow Assistant.

Usage:
    python app.py          — launch the Gradio UI
    python app.py --api    — launch the FastAPI server
    python app.py --test   — run all tests
"""
import sys

from src.config import AppConfig


def main():
    if "--test" in sys.argv:
        import pytest
        sys.exit(pytest.main(["tests/", "-v"]))
    elif "--api" in sys.argv:
        from src.api.server import launch_api
        config = AppConfig.from_yaml_and_env()
        launch_api(config)
    else:
        from src.ui.gradio_app import launch_ui
        config = AppConfig.from_yaml_and_env()
        launch_ui(config)


if __name__ == "__main__":
    main()
