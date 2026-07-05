"""对话意图识别 — 关键词初筛 + LLM 辅助确认。"""
import logging
from typing import Literal

logger = logging.getLogger(__name__)

Intent = Literal["qa", "summarize_full", "summarize_topic"]

_SUMMARY_FULL_KEYWORDS = ["总结全文", "全文总结", "概括一下", "帮我总结", "总结一下", "做个总结"]
_SUMMARY_TOPIC_KEYWORDS = ["总结", "归纳", "梳理"]


def classify_intent(query: str, llm=None) -> Intent:
    """判断用户意图：qa / summarize_full / summarize_topic。

    两级策略：
    1. 关键词明确命中 → 直接返回（零成本）
    2. 关键词模糊 / 无法确定 → 调用 LLM 分类（~30 tokens）
    """
    q = query.strip()

    # 全文总结：短句 + 总结关键词
    for kw in _SUMMARY_FULL_KEYWORDS:
        if kw in q and len(q) < len(kw) + 15:
            return "summarize_full"

    # 主题总结：含总结词 + 有明确主题
    for kw in _SUMMARY_TOPIC_KEYWORDS:
        if kw in q:
            remaining = q.replace(kw, "").strip()
            if remaining and len(remaining) > 2:
                return "summarize_topic"

    # 关键词无法确定 → LLM 辅助
    if llm:
        return _llm_classify(q, llm)

    return "qa"


def _llm_classify(query: str, llm) -> Intent:
    """用 LLM 做意图分类，仅输出一个词。"""
    from langchain_core.messages import HumanMessage

    prompt = (
        "判断用户意图分类，仅输出一个词：\n"
        "- qa：信息查询、问答\n"
        "- summarize_full：要求总结全部内容\n"
        "- summarize_topic：要求总结特定主题\n\n"
        f"用户问题：{query}\n\n分类："
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        label = resp.content.strip().lower()
        if label in ("qa", "summarize_full", "summarize_topic"):
            return label
    except Exception as exc:
        logger.warning("LLM 意图分类失败，降级为 qa: %s", exc)
    return "qa"
