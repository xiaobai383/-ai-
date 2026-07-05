"""总结场景：Map-Reduce 全文聚合 + 主题增强检索。"""
import logging
from typing import List

logger = logging.getLogger(__name__)


def summarize_all_chunks(chunks: List[str], llm, max_chunk_chars: int = 3000) -> str:
    """全文总结：优先直接发送，超出上下文窗口时才走 Map-Reduce。"""
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]

    from langchain_core.messages import HumanMessage

    # 拼接全文，检查总长度
    full_text = "\n\n".join(c[:max_chunk_chars] for c in chunks)

    # 大多数模型上下文窗口 ~128k tokens ≈ 400k 字符
    # 保守阈值：200k 字符以内直接发送，超出才 Map-Reduce
    DIRECT_THRESHOLD = 200_000

    if len(full_text) <= DIRECT_THRESHOLD:
        # 文档不长，直接全文发给 LLM 总结，不压缩
        prompt = (
            "请对以下文档进行完整、详细的总结。要求：\n"
            "1. 保留所有重要内容和关键细节\n"
            "2. 按文档的逻辑结构组织，使用标题和子标题\n"
            "3. 数据、结论、待办事项全部保留\n\n"
            f"{full_text}"
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as exc:
            logger.warning("直接总结失败，降级为 Map-Reduce: %s", exc)

    # 超长文档才走 Map-Reduce
    summaries = []
    for i, chunk in enumerate(chunks):
        prompt = (
            "请从以下文本中提取所有关键信息，包括：核心论点、重要细节、数据、"
            "结论、待办事项等。不要遗漏重要内容，保持原文的结构和层次：\n\n"
            f"{chunk[:max_chunk_chars]}"
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            summaries.append(resp.content.strip())
        except Exception as exc:
            logger.warning("Map 阶段第 %d 块失败: %s", i, exc)
            summaries.append(chunk[:500])

    combined = "\n\n".join(f"【第{i+1}部分】{s}" for i, s in enumerate(summaries))
    reduce_prompt = (
        "请将以下各部分的要点整合为一份完整的文档总结。要求：\n"
        "1. 保留所有重要内容，不要省略关键细节\n"
        "2. 按文档的逻辑结构组织，使用标题和子标题\n"
        "3. 如果有数据、结论、待办事项，全部保留\n\n"
        f"{combined}"
    )
    try:
        resp = llm.invoke([HumanMessage(content=reduce_prompt)])
        return resp.content.strip()
    except Exception:
        return combined


def topic_focused_retrieval(
    session_id: str, query: str, config, embedder, store, top_k: int = 5,
) -> List[str]:
    """增强主题检索：over-fetch + MMR 去重 + 优先返回父块。"""
    from src.knowledge.store import COLLECTION_SESSION_FILES

    col = store.get_or_create(COLLECTION_SESSION_FILES)
    if col.count() == 0:
        return []

    emb = embedder.embed(query)
    if emb is None:
        return []

    # over-fetch 3x 后 MMR 筛选
    raw = store.query(col, emb, top_k=top_k * 3, where={"session_id": session_id})
    docs = raw.get("documents", [[]])
    metas = raw.get("metadatas", [[]])
    dists = raw.get("distances", [[]])
    if not docs or not docs[0]:
        return []

    selected = []
    selected_dists = []
    for i, doc in enumerate(docs[0]):
        dist = dists[0][i] if dists and i < len(dists[0]) else 1.0
        # MMR 简化：跳过与已选结果距离过近的（多样性）
        if any(abs(dist - sd) < 0.08 for sd in selected_dists):
            continue
        meta = metas[0][i] if metas and i < len(metas[0]) else {}
        # 优先使用父块内容（上下文更完整）
        parent_content = meta.get("parent_content", "")
        content = parent_content if parent_content else doc
        source = meta.get("source", "未知文件")
        selected.append(f"[{source}]\n{content[:3000]}")
        selected_dists.append(dist)
        if len(selected) >= top_k:
            break

    return selected
