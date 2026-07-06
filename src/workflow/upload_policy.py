"""上传策略 —— 决定向云端 LLM 发送什么内容，并生成预览。"""
from dataclasses import dataclass, field
from typing import Dict, List

from src.tools.cost import estimate_cost, estimate_tokens
from src.tools.file_ops import ParsedDocument
from src.tools.redaction import redact


@dataclass
class UploadDecision:
    """上传策略决策的结果。"""

    strategy: str  # full, chunks, redacted, blocked（全文、分块、脱敏、阻止）
    selected_chunks: List[str] = field(default_factory=list)
    redact_map: Dict[str, str] = field(default_factory=dict)
    needs_confirmation: bool = False


@dataclass
class UploadPreview:
    """上传前展示给用户的预览信息。"""

    summary: str
    tokens_in_estimate: int
    cost_estimate: float
    model: str
    has_sensitive: bool = False
    sensitive_types: List[str] = field(default_factory=list)


def decide_upload_strategy(
    doc: ParsedDocument, mode: str, config
) -> UploadDecision:
    """根据模式和文档内容确定上传策略。

    模式说明：
        privacy_enhanced — 默认；无敏感信息则发送全文，有则脱敏后发送
        local_fallback — 完全阻止上传，仅本地处理

    参数：
        doc: 已解析的文档。
        mode: 用户选择的模式。
        config: AppConfig 实例。

    返回：
        UploadDecision，包含策略和辅助数据。
    """
    has_sensitive = len(doc.sensitive_matches) > 0

    if mode == "privacy_enhanced":
        if has_sensitive:
            # 发送前对敏感信息进行脱敏
            redacted_text, redact_map = redact(doc.raw_text, doc.sensitive_matches)
            return UploadDecision(
                strategy="redacted",
                selected_chunks=[redacted_text],
                redact_map=redact_map,
            )
        else:
            return UploadDecision(strategy="full")

    elif mode == "local_fallback":
        return UploadDecision(strategy="blocked")

    else:
        # 未知模式 —— 默认脱敏（隐私增强）
        return decide_upload_strategy(doc, "privacy_enhanced", config)


def generate_preview(
    doc: ParsedDocument,
    decision: UploadDecision,
    config,
) -> UploadPreview:
    """生成人类可读的上传预览。

    参数：
        doc: 已解析的文档。
        decision: 上传策略决策。
        config: AppConfig 实例。

    返回：
        UploadPreview，包含摘要、token/费用估算以及敏感信息标志。
    """
    if decision.strategy == "blocked":
        return UploadPreview(
            summary="⚠️ 本地兜底模式：不上传任何内容到云端。",
            tokens_in_estimate=0,
            cost_estimate=0.0,
            model=config.model_name,
        )

    # 确定将要发送的文本内容
    if decision.strategy == "full":
        text_to_send = doc.raw_text
    elif decision.strategy == "redacted":
        text_to_send = (
            decision.selected_chunks[0] if decision.selected_chunks else ""
        )
    elif decision.strategy == "chunks":
        text_to_send = "\n\n".join(decision.selected_chunks)
    else:
        text_to_send = doc.raw_text

    tokens_in = estimate_tokens(text_to_send, config.model_name)
    cost = estimate_cost(tokens_in, 0, config.model_name)

    has_sensitive = len(doc.sensitive_matches) > 0
    sensitive_types = list({m.type for m in doc.sensitive_matches})

    # 构建摘要
    file_info = f"文件: {doc.title} ({doc.file_type})"
    strategy_labels = {
        "full": "全文发送",
        "redacted": "脱敏后发送",
        "chunks": "分块发送（需确认）",
        "blocked": "已阻止上传",
    }
    strategy_label = strategy_labels.get(decision.strategy, decision.strategy)

    summary_lines = [
        file_info,
        f"策略: {strategy_label}",
        f"预计 token: {tokens_in}",
        f"预计费用: ¥{cost:.6f}",
        f"模型: {config.model_name}",
    ]

    if has_sensitive:
        summary_lines.append(f"⚠️ 检测到敏感信息: {', '.join(sensitive_types)}")

    return UploadPreview(
        summary="\n".join(summary_lines),
        tokens_in_estimate=tokens_in,
        cost_estimate=cost,
        model=config.model_name,
        has_sensitive=has_sensitive,
        sensitive_types=sensitive_types,
    )
