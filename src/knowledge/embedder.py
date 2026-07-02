"""Ollama embedding wrapper — nomic-embed-text via OpenAI-compatible API.

ponytail: thin wrapper around requests, no async. Upgrade path: use ollama Python
package for richer error handling and streaming if latency becomes an issue.
"""
import logging
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_TIMEOUT = 30  # seconds


class OllamaEmbedder:
    """Embed text via a local Ollama instance.

    Usage:
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
        self._check_interval: float = 30.0  # seconds between availability probes

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is pullable/loaded.

        Caches the result for self._check_interval seconds to avoid
        hammering Ollama on every embed call.
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
            # Model names may include tag suffix like "nomic-embed-text:latest"
            available = any(self._model in m for m in models)
            self._available = available
            self._last_check = now
            return available
        except requests.RequestException:
            self._available = False
            self._last_check = now
            return False

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text.

        Returns None on failure — callers must handle gracefully.
        """
        if not self.is_available():
            logger.warning("Ollama not available, skipping embed")
            return None

        try:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding")
        except requests.RequestException as e:
            logger.warning("Ollama embed failed: %s", e)
            return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts, one at a time (Ollama batch API is model-dependent)."""
        return [self.embed(t) for t in texts]
