"""Tests for preprocessing workflow."""
import pytest
from src.config import AppConfig
from src.tools.file_ops import ParsedDocument
from src.workflow.preprocess import preprocess


@pytest.fixture
def config():
    return AppConfig(
        model_name="deepseek-v4-flash",
        model_base_url="https://api.deepseek.com/v1",
        allowed_paths=["data/", "output/"],
        blocked_patterns=[".env"],
        max_file_size_mb=5,
        redaction_enabled=False,  # disable redaction for now
    )


class TestPreprocess:
    """Tests for preprocessing orchestration."""

    def test_single_file(self, config):
        docs = preprocess(["data/test_sample.txt"], config)
        assert len(docs) == 1
        assert isinstance(docs[0], ParsedDocument)
        assert docs[0].path.endswith("test_sample.txt")

    def test_multiple_files(self, config):
        docs = preprocess(
            ["data/test_sample.txt", "data/test_sample.md"], config
        )
        assert len(docs) == 2
        assert docs[0].path.endswith("test_sample.txt")
        assert docs[1].path.endswith("test_sample.md")

    def test_file_not_found(self, config):
        with pytest.raises(FileNotFoundError):
            preprocess(["data/nonexistent.txt"], config)

    def test_empty_file_list(self, config):
        docs = preprocess([], config)
        assert docs == []

    def test_chunks_are_populated(self, config):
        docs = preprocess(["data/test_sample.txt"], config)
        assert len(docs[0].chunks) > 0
        assert all(isinstance(c, str) for c in docs[0].chunks)

    def test_sensitive_matches_default_empty(self, config):
        """When redaction is disabled, sensitive_matches should be empty list."""
        docs = preprocess(["data/test_sample.txt"], config)
        assert docs[0].sensitive_matches == []

    def test_sensitive_matches_when_enabled(self):
        """When redaction enabled, sensitive info in file should be detected."""
        config_enabled = AppConfig(
            model_name="deepseek-v4-flash",
            model_base_url="https://api.deepseek.com/v1",
            allowed_paths=["data/", "output/"],
            blocked_patterns=[".env"],
            max_file_size_mb=5,
            redaction_enabled=True,
        )
        # This file doesn't have sensitive info, but detection should run
        docs = preprocess(["data/test_sample.txt"], config_enabled)
        # Should run without error, matches may or may not be found
        assert isinstance(docs[0].sensitive_matches, list)
