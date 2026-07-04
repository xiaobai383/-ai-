"""语义搜索接口 — 用自然语言查询 ChromaDB 集合。

ponytail: 跨命名集合的简单 top-k 搜索；无重排序或 BM25+向量混合检索。
升级路径：增加关键词预过滤和交叉编码器重排序。
"""
import logging
import re
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
    """单条搜索命中结果。"""
    chunk_id: str
    document: str
    score: float       # 0–1，越高越相关
    source: str        # 原始文件路径
    doc_type: str      # "runlog" | "output" | "preferences"
    chunk_index: int = 0
    user_query: str = ""
    answer_preview: str = ""


@dataclass
class SearchResponse:
    """完整搜索响应。"""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_hits: int = 0
    embedding_available: bool = True
    collections_searched: List[str] = field(default_factory=list)


ALL_COLLECTIONS = [COLLECTION_RUNLOG, COLLECTION_OUTPUTS, COLLECTION_PREFERENCES]


class Searcher:
    """跨已索引知识集合的语义搜索。

    用法:
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
        """跨指定集合搜索（默认：全部）。

        Args:
            query: 自然语言搜索查询。
            top_k: 每个集合的最大结果数。
            collections: 要搜索的集合（默认：全部）。
            doc_type_filter: 可选的 doc_type 元数据过滤。

        Returns:
            包含排序结果的 SearchResponse。
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

                raw = self.store.query(col, embedding, top_k=top_k * 3, where=where)

                # 展平 ChromaDB 响应
                doc_lists = raw.get("documents", [[]])
                meta_lists = raw.get("metadatas", [[]])
                dist_lists = raw.get("distances", [[]])
                id_lists = raw.get("ids", [[]])

                for i, doc in enumerate(doc_lists[0] if doc_lists else []):
                    meta = (meta_lists[0][i] if meta_lists and i < len(meta_lists[0]) else {})
                    dist = (dist_lists[0][i] if dist_lists and i < len(dist_lists[0]) else 1.0)
                    cid = (id_lists[0][i] if id_lists and i < len(id_lists[0]) else "")

                    # 将距离转换为相似度分数（余弦：0=完全相同，2=完全相反）
                    score = 1.0 - (dist / 2.0) if isinstance(dist, (int, float)) else 0.0
                    score = max(0.0, min(1.0, score))

                    all_results.append(SearchResult(
                        chunk_id=str(cid),
                        document=doc,
                        score=score,
                        source=str(meta.get("source", "")),
                        doc_type=str(meta.get("doc_type", col_name)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        user_query=str(meta.get("user_query", "")),
                        answer_preview=str(meta.get("answer_preview", "")),
                    ))

            except Exception as e:
                logger.warning("Search failed for collection '%s': %s", col_name, e)

        # 关键词加权：文档中包含查询词的结果提升分数
        # 完整查询匹配权重更高，分词匹配权重稍低
        query_lower = query.lower()
        query_terms = [t for t in re.split(r'\s+', query_lower) if len(t) > 1]
        for r in all_results:
            doc_lower = r.document.lower()
            hits = sum(1 for t in query_terms if re.search(re.escape(t), doc_lower))
            # 完整查询出现在文档中，大幅加分
            if re.search(re.escape(query_lower), doc_lower):
                r.score = min(1.0, r.score + 0.3)
            elif hits > 0:
                r.score = min(1.0, r.score + 0.1 * hits)

        # 按分数降序排序，取 top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        top = all_results[:top_k]

        return SearchResponse(
            query=query,
            results=top,
            total_hits=len(all_results),
            embedding_available=True,
            collections_searched=collections,
        )
