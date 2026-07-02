"""Fallback module — automatic LLM degradation from cloud to local Ollama."""
from src.fallback.provider import FallbackChatModel

__all__ = ["FallbackChatModel"]
