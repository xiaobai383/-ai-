"""Tests for config module."""
import os
import tempfile
import pytest
from src.config import AppConfig


class TestAppConfig:
    """Test AppConfig loading from YAML and env."""

    def test_loads_from_yaml_and_env(self):
        """Config should load model and limits from config.yaml + env vars."""
        config = AppConfig.from_yaml_and_env()

        assert config.model_name == "deepseek-v4-flash"
        assert config.model_base_url == "https://api.deepseek.com/v1"
        assert config.max_file_size_mb == 5
        assert config.max_tokens_per_request == 50000
        assert config.max_cost_per_request_yuan == 0.5

    def test_allowed_paths_loaded(self):
        """Allowed paths should be loaded from config."""
        config = AppConfig.from_yaml_and_env()
        assert "data/" in config.allowed_paths
        assert "output/" in config.allowed_paths

    def test_blocked_patterns_loaded(self):
        """Blocked patterns should be loaded from config."""
        config = AppConfig.from_yaml_and_env()
        assert ".env" in config.blocked_patterns
        assert "*.key" in config.blocked_patterns

    def test_redaction_rules_loaded(self):
        """Redaction rules should be loaded from config."""
        config = AppConfig.from_yaml_and_env()
        assert config.redaction_enabled is True
        assert config.redaction_rules["phone"] is True

    def test_env_var_overrides_model_name(self):
        """MODEL_NAME env var should override config.yaml value."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("model:\n  name: default-model\n  base_url: https://x.com/v1\n")
            yaml_path = f.name

        try:
            os.environ["MODEL_NAME"] = "env-model"
            os.environ["OPENAI_BASE_URL"] = "https://override.com/v1"
            config = AppConfig.from_yaml_and_env(yaml_path=yaml_path)
            assert config.model_name == "env-model"
            assert config.model_base_url == "https://override.com/v1"
        finally:
            os.unlink(yaml_path)
            del os.environ["MODEL_NAME"]
            del os.environ["OPENAI_BASE_URL"]

    def test_api_key_from_env(self):
        """API key should come from OPENAI_API_KEY env var."""
        os.environ["OPENAI_API_KEY"] = "sk-test-123"
        try:
            config = AppConfig.from_yaml_and_env()
            assert config.api_key == "sk-test-123"
        finally:
            del os.environ["OPENAI_API_KEY"]

    def test_default_limits_when_missing(self):
        """Missing limits in YAML should fall back to defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("model:\n  name: m\n  base_url: https://x.com/v1\n")
            yaml_path = f.name

        try:
            config = AppConfig.from_yaml_and_env(yaml_path=yaml_path)
            assert config.max_file_size_mb > 0  # has a default
            assert config.max_tokens_per_request > 0
        finally:
            os.unlink(yaml_path)
