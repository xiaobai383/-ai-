"""Semantic search interface — query ChromaDB collections with natural language.

ponytail: simple top-k search across named collections; no re-ranking or hybrid
BM25+vector. Upgrade path: add keyword-filter pre-pass and cross-encoder rerank.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.knowledge.embedder import OllamaEmbedder
from src.knowledge.store import (
    COLLECTION_OUTPUTS,
    COLLECTION_PREFERENCES,
    COLLECTION_RUNLOG,
    KnowledgeStore,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search hit."""
    chunk_id: str
    document: str
    score: float       # 0–1, higher is more relevant
    source: str        # original file path
    doc_type: str      # "runlog" | "output" | "preferences"
    chunk_index: int = 0


@dataclass
class SearchResponse:
    """Full search response."""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_hits: int = 0
    embedding_available: bool = True
    collections_searched: List[str] = field(default_factory=list)


ALL_COLLECTIONS = [COLLECTION_RUNLOG, COLLECTION_OUTPUTS, COLLECTION_PREFERENCES]


class Searcher:
    """Semantic search across indexed knowledge collections.

    Usage:
        searcher = Searcher(store, embedder)
        results = searcher.search("上次风险分析的结果是什么？", top_k=5)
    """

    def __init__(self, store: KnowledgeStore, embedder: OllamaEmbedder):
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        collections: Optional[List[str]] = None,
        doc_type_filter: Optional[str] = None,
    ) -> SearchResponse:
        """Search across specified collections (default: all).

        Args:
            query: Natural language search query.
            top_k: Max results per collection.
            collections: Which collections to search (default: all).
            doc_type_filter: Optional filter on doc_type metadata.

        Returns:
            SearchResponse with ranked results.
        """
        if collections is None:
            collections = ALL_COLLECTIONS

        if not self.embedder.is_available():
            return SearchResponse(
                query=query,
                embedding_available=False,
                collections_searched=collections,
            )

        embedding = self.embedder.embed(query)
        if embedding is None:
            return SearchResponse(
                query=query,
                embedding_available=False,
                collections_searched=collections,
            )

        all_results: List[SearchResult] = []

        where = None
        if doc_type_filter:
            where = {"doc_type": doc_type_filter}

        for col_name in collections:
            try:
                col = self.store.get_or_create(col_name)
                if col.count() == 0:
                    continue

                raw = self.store.query(col, embedding, top_k=top_k, where=where)

                # Flatten ChromaDB response
                doc_lists = raw.get("documents", [[]])
                meta_lists = raw.get("metadatas", [[]])
                dist_lists = raw.get("distances", [[]])
                id_lists = raw.get("ids", [[]])

                for i, doc in enumerate(doc_lists[0] if doc_lists else []):
                    meta = (meta_lists[0][i] if meta_lists and i < len(meta_lists[0]) else {})
                    dist = (dist_lists[0][i] if dist_lists and i < len(dist_lists[0]) else 1.0)
                    cid = (id_lists[0][i] if id_lists and i < len(id_lists[0]) else "")

                    # Convert distance to similarity score (cosine: 0=identical, 2=opposite)
                    score = 1.0 - (dist / 2.0) if isinstance(dist, (int, float)) else 0.0
                    score = max(0.0, min(1.0, score))

                    all_results.append(SearchResult(
                        chunk_id=str(cid),
                        document=doc,
                        score=score,
                        source=str(meta.get("source", "")),
                        doc_type=str(meta.get("doc_type", col_name)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                    ))

            except Exception as e:
                logger.warning("Search failed for collection '%s': %s", col_name, e)

        # Sort by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)
        top = all_results[:top_k]

        return SearchResponse(
            query=query,
            results=top,
            total_hits=len(all_results),
            embedding_available=True,
            collections_searched=collections,
        )
