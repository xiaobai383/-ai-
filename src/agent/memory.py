"""分层对话记忆 — 短期窗口 + 长期向量记忆 + Query 改写。"""
import logging
import uuid
from typing import List

from src.knowledge.store import COLLECTION_CONVERSATION_MEMORY, KnowledgeStore

logger = logging.getLogger(__name__)


def rewrite_query(history: List[dict], query: str, llm=None, use_llm: bool = True) -> str:
    """基于短期记忆做指代消解，把模糊提问补全为完整问句。

    Args:
        use_llm: True 用 LLM 改写（准确），False 用规则改写（零成本）。
    """
    if not history:
        return query

    recent = history[-4:]  # 最近 2 轮

    if use_llm and llm:
        return _llm_rewrite(recent, query, llm)
    return _rule_rewrite(recent, query)


def _rule_rewrite(recent: List[dict], query: str) -> str:
    """规则改写：替换常见代词为上轮主题词。"""
    q = query
    # 从上一轮 user 消息提取可能的主题词
    prev = ""
    for msg in reversed(recent):
        if msg["role"] == "user":
            prev = msg["content"][:50]
            break
    if not prev:
        return q
    # 简单替换：它/这个/那个 → 上轮提到的关键词
    pronouns = ["它", "这个", "那个", "上面的", "刚才的"]
    for p in pronouns:
        if p in q and p not in prev:
            q = q.replace(p, f"{p}（指「{prev[:15]}」）")
    return q


def _llm_rewrite(recent: List[dict], query: str, llm) -> str:
    """LLM 改写：准确的指代消解。"""
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    prompt = f"""根据以下对话历史，将用户最新问题中的代词（它、这个、那个、上面的）替换为具体名词，输出改写后的完整问题。如果没有需要改写的代词，原样输出。

对话历史：
{history_text}

用户最新问题：{query}

改写后的问题（仅输出问题本身）："""
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        rewritten = resp.content.strip()
        if not rewritten or len(rewritten) > len(query) * 3:
            return query
        return rewritten
    except Exception:
        return query


def store_conversation_turn(store, embedder, session_id: str,
                            user_msg: str, assistant_msg: str) -> None:
    """将一轮对话摘要存入 conversation_memory 集合。"""
    col = store.get_or_create(COLLECTION_CONVERSATION_MEMORY)
    summary_text = f"用户: {user_msg[:200]}\n助手: {assistant_msg[:200]}"

    emb = embedder.embed(summary_text)
    if emb is None:
        return

    doc_id = f"mem-{session_id}-{uuid.uuid4().hex[:8]}"
    store.add(
        col, [summary_text],
        [{"session_id": session_id, "type": "turn_summary"}],
        [doc_id], [emb],
    )


def recall_long_term_memory(store, embedder, session_id: str,
                            query: str, top_k: int = 3) -> str:
    """从 conversation_memory 召回相关历史记忆。"""
    col = store.get_or_create(COLLECTION_CONVERSATION_MEMORY)
    if col.count() == 0:
        return ""

    emb = embedder.embed(query)
    if emb is None:
        return ""

    raw = store.query(col, emb, top_k=top_k, where={"session_id": session_id})
    docs = raw.get("documents", [[]])
    dists = raw.get("distances", [[]])
    if not docs or not docs[0]:
        return ""

    memories = []
    for i, doc in enumerate(docs[0]):
        dist = dists[0][i] if dists and i < len(dists[0]) else 1.0
        score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
        if score < 0.2:
            continue
        memories.append(doc)

    return "\n---\n".join(memories) if memories else ""
