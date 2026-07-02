"""兜底模块 — 从云端 LLM 自动降级到本地 Ollama。"""
from src.fallback.provider import FallbackChatModel

__all__ = ["FallbackChatModel"]
