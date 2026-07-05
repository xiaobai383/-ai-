"""多轮连续对话 —— 拼接历史上下文 + 文件 RAG 注入调用 LLM。

本模块是【替换原有 LLM 调用逻辑的位置】：
    - 原 run_task（src/agent/app.py）：单轮，context = 文档文本，面向文件处理工作流。
    - 本模块 chat 路径：多轮，context = 历史对话 + 文件检索片段 + 当前问题，
      面向豆包式会话场景。

底层 LLM 实例（FallbackChatModel / ChatOpenAI）与 run_task 共用同一套构建逻辑，
区别在于「上下文怎么拼、消息怎么发」。
"""
import logging
from typing import Iterator, List, Optional

from src.agent.prompts import CHAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_context_prompt(
    history: List[dict],
    current_query: str,
    file_context: str = "",
    long_term_memory: str = "",
    max_turns: int = 6,
) -> str:
    """把历史对话 + 文件检索片段 + 当前问题拼接为上下文 prompt。

    滑动窗口：仅保留最近 max_turns 轮对话（一轮 = 一问一答 = 2 条消息），
    控制 prompt token 量，避免多轮对话历史线性膨胀导致成本飙升。
    更早的历史被截断；若发生截断，会在历史段开头注明，让 LLM 知道上下文不完整。

    Args:
        history: 历史轮次列表，每项含 role / content。
        current_query: 当前用户问题。
        file_context: 可选，从 ChromaDB 检索到的文件片段文本。
        long_term_memory: 可选，从 conversation_memory 召回的相关历史记忆。
        max_turns: 保留的最大对话轮数（默认 6 轮 = 12 条消息）。

    Returns:
        拼接后的 prompt 文本。
    """
    lines: List[str] = []

    if history:
        # 滑动窗口：一轮 = user + assistant = 2 条消息
        max_messages = max_turns * 2
        truncated = len(history) > max_messages
        recent = history[-max_messages:] if truncated else history

        lines += ["## 历史对话", ""]
        if truncated:
            lines.append(f"（注：仅保留最近 {max_turns} 轮对话，更早的历史已省略）")
            lines.append("")
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}：{msg['content']}")
        lines.append("")

    if long_term_memory:
        lines += ["## 相关历史记忆", "", long_term_memory, ""]

    if file_context:
        lines += ["## 参考文件（来自上传）", "", file_context, ""]

    lines += ["## 当前问题", "", current_query]
    return "\n".join(lines)


def retrieve_file_context(
    session_id: str,
    query: str,
    config,
    top_k: int = 3,
) -> tuple[str, dict]:
    """从 ChromaDB 检索当前会话上传文件的相关片段（RAG 注入）。

    用 Searcher 在 COLLECTION_SESSION_FILES 集合中检索，where session_id 过滤。
    Ollama 不可用时返回空字符串（降级为纯对话，不阻塞）。

    Args:
        session_id: 当前会话 ID。
        query: 当前用户问题（作为检索 query）。
        config: AppConfig 实例。
        top_k: 检索结果数。

    Returns:
        (file_context, file_redact_map) — 文件片段文本 + 文件级脱敏映射。
    """
    try:
        import json as _json

        from src.knowledge.embedder import OllamaEmbedder
        from src.knowledge.store import COLLECTION_SESSION_FILES, KnowledgeStore

        store = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
        embedder = OllamaEmbedder(
            base_url=config.knowledge_embed_base_url,
            model=config.knowledge_embed_model,
        )
        if not embedder.is_available():
            return "", {}

        col = store.get_or_create(COLLECTION_SESSION_FILES)
        if col.count() == 0:
            return "", {}

        query_emb = embedder.embed(query)
        if query_emb is None:
            return "", {}

        # where session_id 过滤，只检索当前会话的文件
        raw = store.query(col, query_emb, top_k=top_k, where={"session_id": session_id})
        docs = raw.get("documents", [[]])
        if not docs or not docs[0]:
            return "", {}

        metas = raw.get("metadatas", [[]])
        dists = raw.get("distances", [[]])
        fragments: List[str] = []
        file_redact_map: dict = {}
        for i, doc in enumerate(docs[0]):
            dist = dists[0][i] if dists and i < len(dists[0]) else 1.0
            score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
            if score < 0.15:  # 相关性太低跳过
                continue
            meta = metas[0][i] if metas and i < len(metas[0]) else {}
            source = meta.get("source", "未知文件")
            fragments.append(f"[{source} 相关度{int(score*100)}%]\n{doc[:2000]}")
            # 收集文件级 redact_map
            frm = meta.get("redact_map")
            if frm:
                try:
                    file_redact_map.update(_json.loads(frm))
                except Exception:
                    pass

        return ("\n\n".join(fragments) if fragments else ""), file_redact_map
    except Exception as exc:
        logger.warning("文件 RAG 检索失败，降级为纯对话: %s", exc)
        return "", {}


def chat_stream(
    history: List[dict],
    query: str,
    llm,
    file_context: str = "",
    long_term_memory: str = "",
    usage_out: Optional[dict] = None,
) -> Iterator[str]:
    """流式多轮对话生成器，逐块 yield LLM 输出文本。

    【本函数即替换原 run_task 单轮 LLM 调用的多轮对话路径】
    原 run_task 通过 _call_llm 发送 (system_prompt, query+文档context)；
    这里发送 (CHAT_SYSTEM_PROMPT, 历史对话+文件检索+当前问题)，使 LLM 能基于历史
    和上传文件连续回答。

    Args:
        history: 历史消息列表（已按时间正序），每项含 role / content。
        query: 当前用户问题。
        llm: LangChain 兼容聊天模型实例（FallbackChatModel / ChatOpenAI）。
        file_context: 可选，从 ChromaDB 检索到的文件片段文本。
        long_term_memory: 可选，从 conversation_memory 召回的相关历史记忆。
        usage_out: 可选，流式结束后写入 token 用量 {tokens_in, tokens_out, used_fallback}。

    Yields:
        每个流式 token 块的文本。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    context_prompt = build_context_prompt(history, query, file_context, long_term_memory)
    messages = [
        SystemMessage(content=CHAT_SYSTEM_PROMPT),
        HumanMessage(content=context_prompt),
    ]

    if llm is None:
        yield "（未配置 LLM，无法生成回复）"
        return

    def _fill_usage(response_obj, final_text: str = ""):
        """从 LLM 响应/最后 chunk 提取 token usage 写入 usage_out。

        如果 API 未返回 token 用量（流式场景常见），用 tiktoken 从文本估算。
        """
        if usage_out is None:
            return
        try:
            from src.agent.app import _extract_token_usage
            ti, to = _extract_token_usage(response_obj)
        except Exception:
            ti, to = 0, 0

        # API 未返回用量时，从文本估算
        if ti == 0 and to == 0 and final_text:
            try:
                from src.tools.cost import estimate_tokens
                from src.config import AppConfig
                # 估算输入：历史 + 当前问题（粗略）
                ti = estimate_tokens(query, "default")
                # 估算输出：LLM 回复文本
                to = estimate_tokens(final_text, "default")
            except Exception:
                pass

        usage_out["tokens_in"] = ti
        usage_out["tokens_out"] = to
        usage_out["used_fallback"] = getattr(llm, "used_fallback", False)

    # 优先流式
    if hasattr(llm, "stream"):
        try:
            last_chunk = None
            streamed_text = ""
            for chunk in llm.stream(messages):
                last_chunk = chunk
                chunk_text = chunk.content if hasattr(chunk, "content") else str(chunk)
                if chunk_text:
                    streamed_text += chunk_text
                    yield chunk_text
            _fill_usage(last_chunk, streamed_text)
            return
        except Exception as exc:
            logger.warning("对话流式失败，回退 invoke: %s", exc)

    # 非流式兜底
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    _fill_usage(response, text)
    yield text


def smart_retrieve(
    session_id: str, query: str, config, llm,
    history: List[dict], store=None, embedder=None,
) -> tuple[str, str, dict]:
    """意图感知检索：分类 → 路由 → 返回 (file_context, long_term_memory, redact_map)。"""
    from src.agent.intent import classify_intent
    from src.agent.memory import recall_long_term_memory, rewrite_query

    if store is None:
        from src.knowledge.store import KnowledgeStore
        store = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
    if embedder is None:
        from src.knowledge.embedder import OllamaEmbedder
        embedder = OllamaEmbedder(
            base_url=config.knowledge_embed_base_url,
            model=config.knowledge_embed_model,
        )
    if not embedder.is_available():
        return "", "", {}

    # Step 1: Query 改写（指代消解）
    if getattr(config, 'query_rewrite_enabled', True):
        use_llm = getattr(config, 'query_rewrite_use_llm', True)
        rewritten = rewrite_query(history, query, llm, use_llm=use_llm)
    else:
        rewritten = query

    # Step 2: 意图分类（关键词 + LLM 辅助）
    intent = classify_intent(rewritten, llm=llm)

    # Step 3: 路由检索
    file_context = ""
    file_redact_map = {}

    if intent == "qa":
        file_context, file_redact_map = retrieve_file_context(session_id, rewritten, config)
    elif intent == "summarize_full":
        import json as _json
        from src.knowledge.store import COLLECTION_SESSION_FILES
        col = store.get_or_create(COLLECTION_SESSION_FILES)
        raw = col.get(where={"session_id": session_id}, include=["documents", "metadatas"])
        all_docs = raw.get("documents", [])
        all_metas = raw.get("metadatas", [])
        for meta in all_metas:
            frm = meta.get("redact_map")
            if frm:
                try:
                    file_redact_map.update(_json.loads(frm))
                except Exception:
                    pass
        if all_docs:
            from src.agent.summarize import summarize_all_chunks
            file_context = summarize_all_chunks(all_docs, llm)
    elif intent == "summarize_topic":
        from src.agent.summarize import topic_focused_retrieval
        chunks = topic_focused_retrieval(session_id, rewritten, config, embedder, store)
        file_context = "\n\n".join(chunks)

    # Step 4: 长期记忆召回
    long_term = ""
    if getattr(config, 'conversation_memory_enabled', True):
        long_term = recall_long_term_memory(store, embedder, session_id, rewritten)

    return file_context, long_term, file_redact_map
