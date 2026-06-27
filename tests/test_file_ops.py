"""Tests for file operations module."""
import os
import tempfile
from pathlib import Path

import pytest
from src.config import AppConfig
from src.tools.file_ops import (
    ParsedDocument,
    chunk_text,
    parse_file,
    read_file,
    save_file,
)


@pytest.fixture
def config():
    """Create a test config with data/ and output/ in allowed paths."""
    return AppConfig(
        model_name="deepseek-v4-flash",
        model_base_url="https://api.deepseek.com/v1",
        allowed_paths=["data/", "output/", str(Path(tempfile.gettempdir()))],
        blocked_patterns=[".env", "*.key", "*.pem"],
        max_file_size_mb=5,
    )


class TestReadFileSecurity:
    """Tests for read_file security boundaries."""

    def test_read_allowed_path(self, config):
        content = read_file("data/test_sample.txt", config)
        assert "测试文本文件" in content

    def test_blocked_blacklist_file(self, config):
        """Reading a .env file should raise PermissionError."""
        with pytest.raises(PermissionError, match="blocked"):
            read_file("data/.env", config)

    def test_blocked_key_file(self, config):
        """Reading a .key file should raise PermissionError."""
        with pytest.raises(PermissionError, match="blocked"):
            read_file("data/secret.key", config)

    def test_path_not_in_whitelist(self, config):
        """Reading from a path outside allowed dirs should raise."""
        with pytest.raises(PermissionError, match="not in allowed"):
            read_file("/etc/passwd", config)

    def test_file_too_large(self, config):
        """Files exceeding max size should be rejected."""
        # Create a temp file > 5MB
        big_path = Path(tempfile.gettempdir()) / "_big_test_file.txt"
        try:
            # Write a file just over the limit (config has 5MB)
            five_mb_plus = (5 * 1024 * 1024) + 100
            big_path.write_bytes(b"x" * five_mb_plus)
            with pytest.raises(PermissionError, match="exceeds"):
                read_file(str(big_path), config)
        finally:
            big_path.unlink(missing_ok=True)

    def test_nonexistent_file(self, config):
        with pytest.raises(FileNotFoundError):
            read_file("data/nonexistent.txt", config)


class TestSaveFileSecurity:
    """Tests for save_file security boundaries."""

    def test_save_to_allowed_path(self, config):
        out_path = str(Path(tempfile.gettempdir()) / "test_output.txt")
        try:
            result = save_file(out_path, "hello world", config, overwrite=True)
            assert result == out_path
            assert Path(out_path).read_text(encoding="utf-8") == "hello world"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_save_outside_allowed_raises(self, config):
        with pytest.raises(PermissionError, match="not in allowed"):
            save_file("/etc/output.txt", "data", config)

    def test_save_overwrite_protection(self, config):
        out_path = str(Path(tempfile.gettempdir()) / "test_no_overwrite.txt")
        try:
            # First write
            Path(out_path).write_text("original", encoding="utf-8")
            # Second write without overwrite should raise
            with pytest.raises(FileExistsError, match="already exists"):
                save_file(out_path, "new content", config, overwrite=False)
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_save_overwrite_allowed(self, config):
        out_path = str(Path(tempfile.gettempdir()) / "test_overwrite_ok.txt")
        try:
            Path(out_path).write_text("original", encoding="utf-8")
            result = save_file(out_path, "new content", config, overwrite=True)
            assert Path(out_path).read_text(encoding="utf-8") == "new content"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_read_write_roundtrip(self, config):
        """Write content, then read it back — should match."""
        out_path = str(Path(tempfile.gettempdir()) / "roundtrip.txt")
        original = "往返测试 content 内容"
        try:
            save_file(out_path, original, config, overwrite=True)
            # Read needs the path in allowed paths too — tempdir IS in allowed
            restored = read_file(out_path, config)
            assert restored == original
        finally:
            Path(out_path).unlink(missing_ok=True)


class TestParseFile:
    """Tests for file parsing."""

    def test_parse_txt_file(self, config):
        doc = parse_file("data/test_sample.txt", config)
        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == "txt"
        assert len(doc.paragraphs) > 0

    def test_parse_md_file(self, config):
        doc = parse_file("data/test_sample.md", config)
        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == "md"
        assert "第一章" in doc.raw_text

    def test_unsupported_extension(self, config):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_file("data/file.xyz", config)

    def test_parsed_document_has_path(self, config):
        doc = parse_file("data/test_sample.txt", config)
        assert doc.path.endswith("test_sample.txt")

    def test_parsed_txt_title_from_filename(self, config):
        doc = parse_file("data/test_sample.txt", config)
        assert doc.title is not None


class TestChunkText:
    """Tests for text chunking."""

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_paragraph_boundary_preserved(self):
        text = "段落一。\n\n段落二。\n\n段落三。"
        chunks = chunk_text(text, max_tokens=10)
        # Each paragraph should be in a chunk, not split mid-paragraph
        for chunk in chunks:
            # No chunk should have a trailing incomplete paragraph
            assert not chunk.endswith("段")

    def test_max_tokens_enforced(self):
        """Chunks should not exceed max_tokens significantly."""
        long_text = "这是一段很长的文本。" * 100
        chunks = chunk_text(long_text, max_tokens=50)
        from src.tools.cost import estimate_tokens
        for chunk in chunks:
            tokens = estimate_tokens(chunk, "deepseek-v4-flash")
            assert tokens <= 55  # allow small buffer for paragraph boundary

    def test_empty_text(self):
        chunks = chunk_text("", max_tokens=100)
        assert chunks == []

    def test_single_paragraph_exceeds_limit(self):
        """A single huge paragraph should still be chunked (hard split)."""
        huge = "词" * 500
        chunks = chunk_text(huge, max_tokens=10)
        assert len(chunks) > 1
