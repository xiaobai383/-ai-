"""Entry point for the Personal AI Workflow Assistant.

Usage:
    python app.py          — launch the Gradio UI
    python app.py --test   — run all tests
"""
import sys

from src.config import AppConfig
from src.ui.gradio_app import launch_ui


def main():
    if "--test" in sys.argv:
        import pytest
        sys.exit(pytest.main(["tests/", "-v"]))
    else:
        config = AppConfig.from_yaml_and_env()
        launch_ui(config)


if __name__ == "__main__":
    main()
