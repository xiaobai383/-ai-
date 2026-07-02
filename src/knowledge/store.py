"""ChromaDB storage layer — collection management and persistence.

ponytail: single ChromaDB client, no fancy multi-tenancy; O(1) collection lookup
by name. Upgrade path: add tenant isolation if shared ChromaDB is ever needed.
"""
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# ── Well-known collection names ──
COLLECTION_RUNLOG = "runlog"
COLLECTION_OUTPUTS = "outputs"
COLLECTION_PREFERENCES = "preferences"


class KnowledgeStore:
    """Manages ChromaDB persistence and collection lifecycle.

    Usage:
        store = KnowledgeStore(persist_dir="data/chroma")
        runlog_col = store.get_or_create(COLLECTION_RUNLOG)
        store.add(runlog_col, docs, metadatas, ids, embeddings)
    """

    def __init__(self, persist_dir: str = "data/chroma"):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[chromadb.PersistentClient] = None
        # ponytail: disable default embedding model download — we provide
        # embeddings via Ollama. Without this, ChromaDB downloads 79MB ONNX.
        self._ef = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazy-init the ChromaDB persistent client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def get_or_create(self, collection_name: str) -> chromadb.Collection:
        """Return an existing collection or create it (no default embedding fn)."""
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
        )

    def close(self) -> None:
        """Cleanly shut down the ChromaDB client (releases SQLite locks)."""
        self._client = None

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists."""
        try:
            self.client.delete_collection(name=collection_name)
        except (ValueError, Exception):
            pass  # already gone (chromadb raises NotFoundError or ValueError)

    def list_collections(self) -> List[str]:
        """Return names of all existing collections."""
        cols = self.client.list_collections()
        # chromadb >= 0.5 returns name strings; older versions return objects
        return [c if isinstance(c, str) else c.name for c in cols]

    def count(self, collection_name: str) -> int:
        """Return document count for a collection."""
        col = self.get_or_create(collection_name)
        return col.count()

    def add(
        self,
        collection: chromadb.Collection,
        documents: List[str],
        metadatas: List[dict],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Add documents to a collection (upsert by id)."""
        if not documents:
            return
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def query(
        self,
        collection: chromadb.Collection,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Semantic search in a collection."""
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
