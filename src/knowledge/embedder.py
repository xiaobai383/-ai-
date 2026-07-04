"""Ollama embedding 封装 — 通过 OpenAI 兼容 API 调用 nomic-embed-text。

ponytail: 基于 requests 的轻量封装，无异步。升级路径：如需更丰富的错误处理和
流式支持，可改用 ollama Python 包。
"""
import logging
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT = 30  # 秒


class OllamaEmbedder:
    """通过本地 Ollama 实例为文本生成 embedding。

    用法:
        embedder = OllamaEmbedder()
        if embedder.is_available():
            vec = embedder.embed("hello world")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._available: Optional[bool] = None
        self._last_check: float = 0.0
        self._check_interval: float = 30.0  # 两次可用性探测之间的间隔秒数
        self._last_embed_time: float = 0.0

    def is_available(self) -> bool:
        """检查 Ollama 是否可达且模型可拉取/已加载。

        将结果缓存 self._check_interval 秒，避免每次 embed 调用都频繁请求
        Ollama。
        """
        now = time.time()
        if self._available is not None and (now - self._last_check) < self._check_interval:
            return self._available

        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                self._available = False
                self._last_check = now
                return False

            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # 模型名称可能包含标签后缀，如 "nomic-embed-text:latest"
            available = any(self._model in m for m in models)
            self._available = available
            self._last_check = now
            return available
        except requests.RequestException:
            self._available = False
            self._last_check = now
            return False

    def embed(self, text: str) -> Optional[List[float]]:
        """为单个文本生成 embedding。

        失败时返回 None — 调用者必须妥善处理。
        """
        if not self.is_available():
            logger.warning("Ollama not available, skipping embed")
            return None

        # Ollama 不能处理快速连续请求，需要间隔
        now = time.time()
        elapsed = now - self._last_embed_time
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                    timeout=self._timeout,
                )
                self._last_embed_time = time.time()
                resp.raise_for_status()
                data = resp.json()
                return data.get("embedding")
            except requests.RequestException as e:
                if attempt < 2:
                    logger.warning("Ollama embed attempt %d failed: %s, retrying...", attempt + 1, e)
                    time.sleep(1)
                else:
                    logger.warning("Ollama embed failed after 3 attempts: %s", e)
                    return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """逐个为多个文本生成 embedding（Ollama 批量 API 因模型而异）。"""
        return [self.embed(t) for t in texts]
