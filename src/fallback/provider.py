"""兜底 LLM 提供者 — 云端优先，失败时自动降级到 Ollama。

ponytail：简单的 try/except 兜底，无重试退避或熔断器。
升级路径：为生产环境添加指数退避 + 熔断状态跟踪。
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
    """兼容 LangChain 的聊天模型，失败时回退到 Ollama。

    用法：
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

    # 运行时状态 — 每次调用后设置
    used_fallback: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """先尝试主 API，失败则回退到 Ollama。"""
        self.used_fallback = False

        # 尝试主 API
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
            logger.warning("主 LLM (%s) 失败：%s。正在回退到 Ollama。", self.primary_model, e)

        # 回退到 Ollama
        try:
            fallback = ChatOpenAI(
                model=self.fallback_model,
                api_key="ollama",  # Ollama 不验证
                base_url=self.fallback_base_url,
                temperature=0.3,
                timeout=60,  # 本地模型可能较慢
            )
            result = fallback._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self.used_fallback = True
            return result
        except Exception as e:
            logger.error("兜底 LLM (%s) 也失败了：%s", self.fallback_model, e)
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
