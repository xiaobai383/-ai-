"""Tests for post-processing — formatting output in various formats."""
import json
import re

from src.workflow.postprocess import (
    format_output,
    validate_save_path,
)


def test_format_output_markdown():
    """Test Markdown formatting."""
    raw = "## Title\n\nParagraph 1\n\nParagraph 2"
    result = format_output(raw, fmt="markdown")

    assert "## Title" in result
    assert "Paragraph 1" in result
    assert "Paragraph 2" in result
    assert "\n\n" in result  # Paragraphs separated by blank line


def test_format_output_markdown_collapse_newlines():
    """Test that 3+ newlines collapse to 2."""
    raw = "Line 1\n\n\n\nLine 2"
    result = format_output(raw, fmt="markdown")

    assert "\n\n\n" not in result


def test_format_output_markdown_with_metadata():
    """Test Markdown with metadata footer."""
    raw = "Content here"
    metadata = {"timestamp": "2026-01-01", "model": "deepseek-v4-flash"}
    result = format_output(raw, fmt="markdown", metadata=metadata)

    assert "生成时间" in result
    assert "deepseek-v4-flash" in result


def test_format_output_plain():
    """Test plain text formatting."""
    raw = "## Header\n\n**Bold** text with [link](http://example.com)"
    result = format_output(raw, fmt="plain")

    assert "##" not in result
    assert "**" not in result
    assert "link" in result
    assert "http://example.com" not in result


def test_format_output_json():
    """Test JSON formatting."""
    raw = "## Section 1\n\nContent 1\n\n## Section 2\n\nContent 2"
    result = format_output(raw, fmt="json")

    data = json.loads(result)
    assert "sections" in data
    assert "metadata" in data
    assert len(data["sections"]) >= 2


def test_format_output_html():
    """Test HTML formatting."""
    raw = "## Header\n\n**Bold** text"
    result = format_output(raw, fmt="html")

    assert "<h2>" in result
    assert "<strong>" in result
    assert "Header" in result


def test_format_output_html_with_metadata():
    """Test HTML with metadata footer."""
    raw = "Content"
    metadata = {"timestamp": "2026-01-01", "model": "gpt-4o"}
    result = format_output(raw, fmt="html", metadata=metadata)

    assert "<footer>" in result
    assert "gpt-4o" in result


def test_format_output_empty():
    """Test empty input."""
    assert format_output("", fmt="markdown") == ""
    assert format_output(None, fmt="markdown") == ""


def test_format_output_unknown_format():
    """Test unknown format defaults to markdown."""
    raw = "## Title\n\nContent"
    result = format_output(raw, fmt="unknown")

    assert "## Title" in result


def test_validate_save_path_allowed():
    """Test valid save path validation."""
    class MockConfig:
        allowed_paths = ["data/", "output/"]

    config = MockConfig()
    assert validate_save_path("data/test.md", config) is True
    assert validate_save_path("output/result.md", config) is True


def test_validate_save_path_blocked():
    """Test blocked save path validation."""
    class MockConfig:
        allowed_paths = ["data/", "output/"]

    config = MockConfig()
    assert validate_save_path("/etc/passwd", config) is False
    assert validate_save_path("secrets/key.pem", config) is False
