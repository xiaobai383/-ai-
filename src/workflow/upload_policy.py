"""Upload policy — decide what to send to the cloud LLM and generate previews."""
from dataclasses import dataclass, field
from typing import Dict, List

from src.tools.cost import estimate_cost, estimate_tokens
from src.tools.file_ops import ParsedDocument
from src.tools.redaction import redact


@dataclass
class UploadDecision:
    """The result of the upload policy decision."""

    strategy: str  # full, chunks, redacted, blocked
    selected_chunks: List[str] = field(default_factory=list)
    redact_map: Dict[str, str] = field(default_factory=dict)
    needs_confirmation: bool = False


@dataclass
class UploadPreview:
    """Preview information shown to the user before upload."""

    summary: str
    tokens_in_estimate: int
    cost_estimate: float
    model: str
    has_sensitive: bool = False
    sensitive_types: List[str] = field(default_factory=list)


def decide_upload_strategy(
    doc: ParsedDocument, mode: str, config
) -> UploadDecision:
    """Determine the upload strategy based on mode and document content.

    Modes:
        quick — always send full text, ignore sensitive info
        privacy_enhanced — full if no sensitive, redacted if sensitive
        manual_confirm — send as chunks, always require user confirmation
        local_fallback — block upload entirely, local-only processing

    Args:
        doc: The parsed document.
        mode: User-selected mode.
        config: AppConfig instance.

    Returns:
        UploadDecision with strategy and supporting data.
    """
    has_sensitive = len(doc.sensitive_matches) > 0

    if mode == "quick":
        return UploadDecision(strategy="full")

    elif mode == "privacy_enhanced":
        if has_sensitive:
            # Redact sensitive info before sending
            redacted_text, redact_map = redact(doc.raw_text, doc.sensitive_matches)
            return UploadDecision(
                strategy="redacted",
                selected_chunks=[redacted_text],
                redact_map=redact_map,
            )
        else:
            return UploadDecision(strategy="full")

    elif mode == "manual_confirm":
        return UploadDecision(
            strategy="chunks",
            selected_chunks=list(doc.chunks),
            needs_confirmation=True,
        )

    elif mode == "local_fallback":
        return UploadDecision(strategy="blocked")

    else:
        # Unknown mode — safest default: manual confirm
        return UploadDecision(
            strategy="chunks",
            selected_chunks=list(doc.chunks),
            needs_confirmation=True,
        )


def generate_preview(
    doc: ParsedDocument,
    decision: UploadDecision,
    config,
) -> UploadPreview:
    """Generate a human-readable preview of what will be uploaded.

    Args:
        doc: The parsed document.
        decision: The upload strategy decision.
        config: AppConfig instance.

    Returns:
        UploadPreview with summary, token/cost estimates, and sensitive info flags.
    """
    if decision.strategy == "blocked":
        return UploadPreview(
            summary="⚠️ 本地兜底模式：不上传任何内容到云端。",
            tokens_in_estimate=0,
            cost_estimate=0.0,
            model=config.model_name,
        )

    # Determine what text will be sent
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

    # Build summary
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
