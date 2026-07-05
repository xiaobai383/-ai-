"""应用配置 — 从 YAML 和环境变量加载。"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv


@dataclass
class AppConfig:
    """AI 工作流助手的统一配置中心。"""

    model_name: str = "deepseek-v4-flash"
    model_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""

    max_file_size_mb: int = 5
    max_tokens_per_request: int = 50000
    max_cost_per_request_yuan: float = 0.5

    allowed_paths: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)

    redaction_enabled: bool = True
    redaction_rules: Dict[str, bool] = field(default_factory=dict)

    # v0.2 新增
    default_output_format: str = "markdown"
    workflows_dir: str = "workflows"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    gradio_port: int = 7860
    budget_yuan: float = 10.0

    # v0.3 新增 — 监听 / 调度 / 通知
    watch_enabled: bool = False
    watch_dirs: List[str] = field(default_factory=list)
    watch_patterns: List[str] = field(default_factory=list)
    watch_trigger_mode: str = "instant"
    watch_batch_window_seconds: int = 60
    watch_workflow_template: str = "summarize"
    watch_mode: str = "privacy_enhanced"
    scheduler_enabled: bool = False
    scheduler_jobs_dir: str = "data/scheduled_jobs"
    notifications_enabled: bool = True
    notifications_engine: str = "auto"
    notifications_log_file: str = "data/logs/notifications.jsonl"

    # v1.0 新增 — 知识检索
    knowledge_chroma_dir: str = "data/chroma"
    knowledge_embed_model: str = "nomic-embed-text"
    knowledge_embed_base_url: str = "http://localhost:11434"
    knowledge_default_top_k: int = 5

    # v1.1 新增 — 对话记忆 + 检索优化
    conversation_memory_enabled: bool = True
    query_rewrite_enabled: bool = True
    query_rewrite_use_llm: bool = True
    hierarchical_chunk_child_size: int = 4000
    hierarchical_chunk_parent_size: int = 12000

    # v1.0 新增 — 本地模型兜底
    fallback_enabled: bool = True
    fallback_ollama_base_url: str = "http://localhost:11434/v1"
    fallback_ollama_model: str = "qwen2.5:1.5b"
    fallback_timeout_seconds: int = 30

    @classmethod
    def from_yaml_and_env(cls, yaml_path: str | None = None) -> "AppConfig":
        """从 YAML 文件和 .env 环境变量加载配置。

        参数:
            yaml_path: config.yaml 的路径，默认为当前目录下的 'config.yaml'。

        返回:
            合并了所有设置项的 AppConfig 实例。
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

            # v0.2 新增
            output_cfg = raw.get("output", {})
            config.default_output_format = output_cfg.get("format", config.default_output_format)

            workflow_cfg = raw.get("workflow", {})
            config.workflows_dir = workflow_cfg.get("templates_dir", config.workflows_dir)

            api_cfg = raw.get("api", {})
            config.api_host = api_cfg.get("host", config.api_host)
            config.api_port = api_cfg.get("port", config.api_port)
            config.gradio_port = api_cfg.get("gradio_port", config.gradio_port)
            config.budget_yuan = float(api_cfg.get("budget_yuan", config.budget_yuan))

            # v0.3 新增 — 监听 / 调度 / 通知
            watch_cfg = raw.get("watch", {})
            config.watch_enabled = watch_cfg.get("enabled", config.watch_enabled)
            config.watch_dirs = list(watch_cfg.get("dirs", []))
            config.watch_patterns = list(watch_cfg.get("patterns", ["*.txt", "*.md"]))
            config.watch_trigger_mode = watch_cfg.get("trigger_mode", config.watch_trigger_mode)
            config.watch_batch_window_seconds = watch_cfg.get(
                "batch_window_seconds", config.watch_batch_window_seconds
            )
            config.watch_workflow_template = watch_cfg.get(
                "workflow_template", config.watch_workflow_template
            )
            config.watch_mode = watch_cfg.get("mode", config.watch_mode)

            scheduler_cfg = raw.get("scheduler", {})
            config.scheduler_enabled = scheduler_cfg.get("enabled", config.scheduler_enabled)
            config.scheduler_jobs_dir = scheduler_cfg.get("jobs_dir", config.scheduler_jobs_dir)

            notifications_cfg = raw.get("notifications", {})
            config.notifications_enabled = notifications_cfg.get(
                "enabled", config.notifications_enabled
            )
            config.notifications_engine = notifications_cfg.get("engine", config.notifications_engine)
            config.notifications_log_file = notifications_cfg.get("log_file", config.notifications_log_file)

            # v1.0 新增 — 知识检索
            knowledge_cfg = raw.get("knowledge", {})
            config.knowledge_chroma_dir = knowledge_cfg.get("chroma_dir", config.knowledge_chroma_dir)
            config.knowledge_embed_model = knowledge_cfg.get("embed_model", config.knowledge_embed_model)
            config.knowledge_embed_base_url = knowledge_cfg.get("embed_base_url", config.knowledge_embed_base_url)
            config.knowledge_default_top_k = knowledge_cfg.get("default_top_k", config.knowledge_default_top_k)

            # v1.1 新增 — 对话记忆 + 检索优化
            memory_cfg = raw.get("memory", {})
            config.conversation_memory_enabled = memory_cfg.get("conversation_memory_enabled", config.conversation_memory_enabled)
            config.query_rewrite_enabled = memory_cfg.get("query_rewrite_enabled", config.query_rewrite_enabled)
            config.query_rewrite_use_llm = memory_cfg.get("query_rewrite_use_llm", config.query_rewrite_use_llm)
            config.hierarchical_chunk_child_size = memory_cfg.get("chunk_child_size", config.hierarchical_chunk_child_size)
            config.hierarchical_chunk_parent_size = memory_cfg.get("chunk_parent_size", config.hierarchical_chunk_parent_size)

            # v1.0 新增 — 本地模型兜底
            fallback_cfg = raw.get("fallback", {})
            config.fallback_enabled = fallback_cfg.get("enabled", config.fallback_enabled)
            config.fallback_ollama_base_url = fallback_cfg.get("ollama_base_url", config.fallback_ollama_base_url)
            config.fallback_ollama_model = fallback_cfg.get("ollama_model", config.fallback_ollama_model)
            config.fallback_timeout_seconds = fallback_cfg.get("timeout_seconds", config.fallback_timeout_seconds)

        # 环境变量覆盖
        config.model_name = os.environ.get("MODEL_NAME", config.model_name)
        config.model_base_url = os.environ.get(
            "OPENAI_BASE_URL", config.model_base_url
        )
        config.api_key = os.environ.get("OPENAI_API_KEY", config.api_key)

        max_size_env = os.environ.get("MAX_FILE_SIZE_MB")
        if max_size_env is not None:
            config.max_file_size_mb = int(max_size_env)

        return config

    def save_to_yaml(self, yaml_path: str = "config.yaml") -> None:
        """把当前配置写回 YAML 文件（前端改配置同步后端）。

        只写回前端可编辑的字段，保留 YAML 中其他原有配置不动。
         ponytail: 全量重写整个 model/limits/fallback/redaction/api 段，
        其他段（watch/scheduler/notifications/knowledge）保持原样合并。
        升级路径：做增量字段级 diff 更新，避免覆盖手动编辑。
        """
        # 先读现有 YAML，保留不在编辑范围内的段
        existing = {}
        yp = Path(yaml_path)
        if yp.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                existing = {}

        # 更新可编辑段
        existing["model"] = {
            "name": self.model_name,
            "base_url": self.model_base_url,
        }
        existing["limits"] = {
            "max_file_size_mb": self.max_file_size_mb,
            "max_tokens_per_request": self.max_tokens_per_request,
            "max_cost_per_request_yuan": self.max_cost_per_request_yuan,
        }
        existing["api"] = existing.get("api", {})
        existing["api"]["budget_yuan"] = self.budget_yuan
        existing["redaction"] = {
            "enabled": self.redaction_enabled,
            "rules": self.redaction_rules,
        }
        existing["fallback"] = {
            "enabled": self.fallback_enabled,
            "ollama_base_url": self.fallback_ollama_base_url,
            "ollama_model": self.fallback_ollama_model,
            "timeout_seconds": self.fallback_timeout_seconds,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
