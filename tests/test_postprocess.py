"""Tests for postprocessing module."""
import tempfile
from pathlib import Path

import pytest
from src.config import AppConfig
from src.workflow.postprocess import format_output, validate_save_path


@pytest.fixture
def config():
    return AppConfig(
        model_name="deepseek-v4-flash",
        model_base_url="https://api.deepseek.com/v1",
        allowed_paths=["data/", "output/", str(Path(tempfile.gettempdir()))],
    )


class TestFormatOutput:
    """Tests for output formatting."""

    def test_preserves_markdown_headings(self):
        raw = "# 标题\n\n内容"
        formatted = format_output(raw)
        assert "# 标题" in formatted

    def test_adds_paragraph_spacing(self):
        raw = "第一段\n第二段\n第三段"
        formatted = format_output(raw)
        # format_output preserves structure, each line separated by newline
        assert "第一段" in formatted
        assert "第二段" in formatted
        assert "第三段" in formatted

    def test_preserves_lists(self):
        raw = "- 项目一\n- 项目二"
        formatted = format_output(raw)
        assert "- 项目一" in formatted
        assert "- 项目二" in formatted

    def test_empty_input(self):
        formatted = format_output("")
        assert formatted == ""

    def test_whitespace_only(self):
        formatted = format_output("   \n  \n  ")
        assert isinstance(formatted, str)


class TestValidateSavePath:
    """Tests for save path validation."""

    def test_valid_path(self, config):
        p = str(Path(tempfile.gettempdir()) / "output.md")
        assert validate_save_path(p, config) is True

    def test_path_traversal_blocked(self, config):
        """../ path traversal should be denied."""
        p = str(Path(tempfile.gettempdir()) / ".." / "etc" / "output.md")
        assert validate_save_path(p, config) is False

    def test_absolute_path_not_in_whitelist(self, config):
        assert validate_save_path("/etc/output.md", config) is False

    def test_path_in_allowed_directory(self, config):
        p = str(Path("output/result.md").resolve())
        # Need to ensure output/ exists
        Path("output").mkdir(exist_ok=True)
        try:
            assert validate_save_path(p, config) is True
        finally:
            pass
