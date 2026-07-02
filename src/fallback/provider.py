"""Fallback LLM provider — cloud-first with automatic Ollama degradation.

ponytail: simple try/except fallback, no retry backoff or circuit breaker.
Upgrade path: add exponential backoff + circuit state tracking for production.
"""
import logging
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


class FallbackChatModel(BaseChatModel):
    """A LangChain-compatible chat model that falls back to Ollama on failure.

    Usage:
        llm = FallbackChatModel(
            primary_model="deepseek-v4-flash",
            primary_base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
            fallback_base_url="http://localhost:11434/v1",
            fallback_model="qwen2.5:1.5b",
            timeout=10,
        )
        response = llm.invoke([HumanMessage("hello")])
    """

    primary_model: str = "deepseek-v4-flash"
    primary_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    fallback_base_url: str = "http://localhost:11434/v1"
    fallback_model: str = "qwen2.5:1.5b"
    timeout: int = 10

    # Runtime state — set after each invoke
    used_fallback: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Try primary API, fall back to Ollama on failure."""
        self.used_fallback = False

        # Try primary
        try:
            primary = ChatOpenAI(
                model=self.primary_model,
                api_key=self.api_key,
                base_url=self.primary_base_url,
                temperature=0.3,
                timeout=self.timeout,
            )
            result = primary._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            return result
        except Exception as e:
            logger.warning("Primary LLM (%s) failed: %s. Falling back to Ollama.", self.primary_model, e)

        # Fallback to Ollama
        try:
            fallback = ChatOpenAI(
                model=self.fallback_model,
                api_key="ollama",  # Ollama doesn't validate
                base_url=self.fallback_base_url,
                temperature=0.3,
                timeout=60,  # local model may be slower
            )
            result = fallback._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self.used_fallback = True
            return result
        except Exception as e:
            logger.error("Fallback LLM (%s) also failed: %s", self.fallback_model, e)
            raise

    @property
    def _llm_type(self) -> str:
        return "fallback-chat-model"

    @property
    def _identifying_params(self) -> dict:
        return {
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
        }
