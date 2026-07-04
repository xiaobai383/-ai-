"""ChromaDB 存储层 — 集合管理与持久化。

ponytail: 单一 ChromaDB 客户端，无多租户设计；按名称 O(1) 查找集合。
升级路径：如果将来需要共享 ChromaDB，可增加租户隔离。
"""
import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

# ── 预定义的集合名称 ──
COLLECTION_RUNLOG = "runlog"
COLLECTION_OUTPUTS = "outputs"
COLLECTION_SESSION_FILES = "session_files"  # 会话上传文件的向量检索（RAG 注入）


class KnowledgeStore:
    """管理 ChromaDB 持久化和集合生命周期。

    用法:
        store = KnowledgeStore(persist_dir="data/chroma")
        runlog_col = store.get_or_create(COLLECTION_RUNLOG)
        store.add(runlog_col, docs, metadatas, ids, embeddings)
    """

    def __init__(self, persist_dir: str = "data/chroma"):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[chromadb.PersistentClient] = None
        # ponytail: 禁用默认 embedding 模型下载 — 我们通过 Ollama 提供
        # embedding。否则 ChromaDB 会下载 79MB ONNX 模型。
        self._ef = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """延迟初始化 ChromaDB 持久化客户端。"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def get_or_create(self, collection_name: str) -> chromadb.Collection:
        """返回现有集合或创建新集合（cosine 距离，无默认 embedding 函数）。"""
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def delete_collection(self, collection_name: str) -> None:
        """如果集合存在则将其删除。"""
        try:
            self.client.delete_collection(name=collection_name)
        except (ValueError, Exception):
            pass  # 已被删除（chromadb 抛出 NotFoundError 或 ValueError）

    def list_collections(self) -> List[str]:
        """返回所有现有集合的名称。"""
        cols = self.client.list_collections()
        # chromadb >= 0.5 返回名称字符串；旧版本返回对象
        return [c if isinstance(c, str) else c.name for c in cols]

    def count(self, collection_name: str) -> int:
        """返回集合中的文档数量。"""
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
        """向集合中添加文档（按 id 更新插入）。"""
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
        """在集合中执行语义搜索。"""
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
