"""Application configuration loaded from YAML and environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv


@dataclass
class AppConfig:
    """Central configuration for the AI workflow assistant."""

    model_name: str = "deepseek-v4-flash"
    model_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""

    max_file_size_mb: int = 5
    max_tool_calls_per_request: int = 15
    max_tokens_per_request: int = 50000
    max_cost_per_request_yuan: float = 0.5

    allowed_paths: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)

    redaction_enabled: bool = True
    redaction_rules: Dict[str, bool] = field(default_factory=dict)

    # v0.2 additions
    default_output_format: str = "markdown"
    workflows_dir: str = "workflows"
    preferences_dir: str = "data/preferences"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @classmethod
    def from_yaml_and_env(cls, yaml_path: str | None = None) -> "AppConfig":
        """Load config from YAML file and override with environment variables.

        Args:
            yaml_path: Path to config.yaml. Defaults to 'config.yaml' in cwd.

        Returns:
            AppConfig instance with merged settings.
        """
        load_dotenv()

        if yaml_path is None:
            yaml_path = "config.yaml"

        config = cls()

        if Path(yaml_path).exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            model_cfg = raw.get("model", {})
            config.model_name = model_cfg.get("name", config.model_name)
            config.model_base_url = model_cfg.get("base_url", config.model_base_url)

            limits = raw.get("limits", {})
            config.max_file_size_mb = limits.get(
                "max_file_size_mb", config.max_file_size_mb
            )
            config.max_tool_calls_per_request = limits.get(
                "max_tool_calls_per_request", config.max_tool_calls_per_request
            )
            config.max_tokens_per_request = limits.get(
                "max_tokens_per_request", config.max_tokens_per_request
            )
            config.max_cost_per_request_yuan = limits.get(
                "max_cost_per_request_yuan", config.max_cost_per_request_yuan
            )

            paths = raw.get("paths", {})
            config.allowed_paths = list(paths.get("allowed", []))
            config.blocked_patterns = list(paths.get("blocked_patterns", []))

            redaction = raw.get("redaction", {})
            config.redaction_enabled = redaction.get("enabled", True)
            config.redaction_rules = {
                k: v
                for k, v in redaction.get("rules", {}).items()
                if isinstance(v, bool)
            }

            # v0.2 additions
            output_cfg = raw.get("output", {})
            config.default_output_format = output_cfg.get("format", config.default_output_format)

            workflow_cfg = raw.get("workflow", {})
            config.workflows_dir = workflow_cfg.get("templates_dir", config.workflows_dir)

            preferences_cfg = raw.get("preferences", {})
            config.preferences_dir = preferences_cfg.get("dir", config.preferences_dir)

            api_cfg = raw.get("api", {})
            config.api_host = api_cfg.get("host", config.api_host)
            config.api_port = api_cfg.get("port", config.api_port)

        # Environment variable overrides
        config.model_name = os.environ.get("MODEL_NAME", config.model_name)
        config.model_base_url = os.environ.get(
            "OPENAI_BASE_URL", config.model_base_url
        )
        config.api_key = os.environ.get("OPENAI_API_KEY", config.api_key)

        max_size_env = os.environ.get("MAX_FILE_SIZE_MB")
        if max_size_env is not None:
            config.max_file_size_mb = int(max_size_env)

        return config
