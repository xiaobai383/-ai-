"""Agent 编排 —— run_task 主入口点。"""
import time
import uuid
from typing import Any, List

from src.agent.prompts import SYSTEM_PROMPT
from src.observability.run_log import RunLog, StepLog
from src.tools.cost import check_limits, estimate_cost, estimate_tokens
from src.tools.file_ops import read_file as _file_read
from src.tools.file_ops import save_file as _file_save
from src.tools.redaction import detect_sensitive, redact
from src.workflow.postprocess import format_output, restore_redactions, validate_save_path
from src.workflow.preprocess import preprocess
from src.workflow.upload_policy import decide_upload_strategy, generate_preview


def run_task(
    query: str,
    files: List[str],
    mode: str,
    config,
    llm: Any = None,
    auto_confirm: bool = False,
    output_format: str = "markdown",
) -> RunLog:
    """执行完整工作流：预处理 → 上传策略 → LLM → 后处理。

    这是所有用户任务的主入口点。

    Args:
        query: 用户的自然语言请求。
        files: 待处理的文件路径列表。
        mode: 模式之一：'quick'、'privacy_enhanced'、'manual_confirm'、'local_fallback'。
        config: AppConfig 实例。
        llm: LangChain 兼容的聊天模型。若为 None，则使用占位符。
        auto_confirm: 跳过用户确认步骤（用于测试/自动化）。

    Returns:
        包含完整执行追踪的 RunLog。
    """
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    run_log = RunLog(
        run_id=run_id,
        user_query=query,
        mode=mode,
        model=config.model_name,
    )
    cumulative_tokens_in = 0
    cumulative_cost = 0.0

    # ── 第 1 步：预处理文件 ──
    t0 = time.time()
    try:
        documents = preprocess(files, config)
        duration_ms = int((time.time() - t0) * 1000)
        input_preview = f"files: {files}"
        output_preview = f"parsed {len(documents)} document(s)"
        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="preprocess",
                input_preview=input_preview,
                output_preview=output_preview,
                duration_ms=duration_ms,
                status="success",
            )
        )
    except Exception as e:
        duration_ms = int((time.time() - t0) * 1000)
        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="preprocess",
                input_preview=f"files: {files}",
                output_preview=str(e),
                duration_ms=duration_ms,
                status="failed",
            )
        )
        run_log.result_path = None
        return run_log

    if not documents:
        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="preprocess",
                input_preview="",
                output_preview="no documents to process",
                duration_ms=0,
                status="failed",
            )
        )
        return run_log

    # ── 第 2 步：上传策略决策 ──
    t0 = time.time()
    combined_text = "\n\n".join(doc.raw_text for doc in documents)
    # 使用第一个文档的敏感匹配做策略决策
    primary_doc = documents[0]
    # 合并所有文档的敏感匹配
    all_matches = []
    for doc in documents:
        all_matches.extend(doc.sensitive_matches)
    primary_doc.sensitive_matches = all_matches
    primary_doc.raw_text = combined_text
    primary_doc.chunks = []
    for doc in documents:
        primary_doc.chunks.extend(doc.chunks)

    decision = decide_upload_strategy(primary_doc, mode, config)
    preview = generate_preview(primary_doc, decision, config)
    duration_ms = int((time.time() - t0) * 1000)

    run_log.steps.append(
        StepLog(
            step_id=len(run_log.steps) + 1,
            name="upload_policy",
            input_preview=f"mode={mode}, files={len(documents)}",
            output_preview=f"strategy={decision.strategy}, tokens_in_estimate={preview.tokens_in_estimate}, needs_confirmation={decision.needs_confirmation}",
            duration_ms=duration_ms,
            tokens_in=preview.tokens_in_estimate,
            cost_yuan=preview.cost_estimate,
            status="success",
        )
    )

    # ── 第 2.5 步：检查限制 ──
    blocked, reason = check_limits(
        current_tokens_in=preview.tokens_in_estimate,
        current_cost_yuan=preview.cost_estimate,
        limits={
            "max_tokens_per_request": config.max_tokens_per_request,
            "max_cost_per_request_yuan": config.max_cost_per_request_yuan,
        },
    )
    if blocked:
        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="limit_check",
                input_preview="",
                output_preview=reason,
                duration_ms=0,
                status="failed",
            )
        )
        run_log.result_path = None
        return run_log

    cumulative_tokens_in += preview.tokens_in_estimate
    cumulative_cost += preview.cost_estimate

    # ── 第 3 步：LLM 调用（或本地兜底） ──
    # 根据上传策略确定实际发送的文本
    if decision.strategy in ("redacted", "chunks") and decision.selected_chunks:
        text_to_send = "\n\n".join(decision.selected_chunks)
    else:
        text_to_send = combined_text

    if decision.strategy == "blocked":
        # 本地兜底：不调用云端
        t0 = time.time()
        result_text = _local_analysis(query, combined_text)
        duration_ms = int((time.time() - t0) * 1000)
        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="local_analysis",
                input_preview=f"query: {query}",
                output_preview=result_text[:200],
                duration_ms=duration_ms,
                status="success",
            )
        )
    else:
        # 云端 LLM 调用
        t0 = time.time()
        try:
            result_text, used_fallback = _call_llm(
                llm=llm,
                system_prompt=SYSTEM_PROMPT,
                user_query=query,
                context=text_to_send,
            )
            if used_fallback:
                run_log.fallback = True
        except Exception as e:
            duration_ms = int((time.time() - t0) * 1000)
            run_log.steps.append(
                StepLog(
                    step_id=len(run_log.steps) + 1,
                    name="llm_call",
                    input_preview=f"query: {query}",
                    output_preview=str(e),
                    duration_ms=duration_ms,
                    status="failed",
                )
            )
            return run_log

        duration_ms = int((time.time() - t0) * 1000)
        # 估算输出 token 数
        tokens_out = estimate_tokens(result_text, config.model_name)
        cost = estimate_cost(cumulative_tokens_in, tokens_out, config.model_name)

        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="llm_call",
                input_preview=f"query: {query}",
                output_preview=result_text[:200],
                duration_ms=duration_ms,
                tokens_in=cumulative_tokens_in,
                tokens_out=tokens_out,
                cost_yuan=cost,
                status="success",
            )
        )
        cumulative_cost += cost

        # 还原脱敏占位符
        if decision.redact_map:
            result_text = restore_redactions(result_text, decision.redact_map)

    # ── 第 4 步：后处理 ──
    t0 = time.time()
    formatted = format_output(result_text, fmt=output_format)
    ext = {"markdown": ".md", "plain": ".txt", "json": ".json", "html": ".html"}.get(output_format, ".md")
    output_path = f"output/{run_id}{ext}"
    if not validate_save_path(output_path, config):
        output_path = f"data/{run_id}.md"

    duration_ms = int((time.time() - t0) * 1000)
    run_log.steps.append(
        StepLog(
            step_id=len(run_log.steps) + 1,
            name="postprocess",
            input_preview="format and save",
            output_preview=f"saved to {output_path}",
            duration_ms=duration_ms,
            status="success",
        )
    )

    # ── 第 5 步：保存结果 ──
    t0 = time.time()
    try:
        _file_save(output_path, formatted, config, overwrite=True)
        run_log.result_path = output_path
        duration_ms = int((time.time() - t0) * 1000)
        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="save_result",
                input_preview=output_path,
                output_preview=f"saved {len(formatted)} chars",
                duration_ms=duration_ms,
                status="success",
            )
        )
    except Exception as e:
        duration_ms = int((time.time() - t0) * 1000)
        run_log.steps.append(
            StepLog(
                step_id=len(run_log.steps) + 1,
                name="save_result",
                input_preview=output_path,
                output_preview=str(e),
                duration_ms=duration_ms,
                status="failed",
            )
        )

    # ── 更新汇总（计算属性，无需手动设置） ──
    # total_tokens_in、total_tokens_out、total_cost_yuan 由各步骤计算得出

    return run_log


def _call_llm(
    llm,
    system_prompt: str,
    user_query: str,
    context: str,
) -> tuple[str, bool]:
    """使用给定上下文调用 LLM。

    Args:
        llm: LLM 实例（或用于测试的 FakeLLM）。
        system_prompt: 系统提示词文本。
        user_query: 用户的自然语言查询。
        context: 文档内容（调用方已根据 upload_policy 决定脱敏/全文）。

    Returns:
        (LLM 响应文本, 是否使用了兜底模型) 元组。
    """
    import logging
    logger = logging.getLogger(__name__)
    used_fallback = False

    if llm is None:
        return "（未配置 LLM，无法生成回复）", False

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"## 用户任务\n\n{user_query}\n\n## 文档内容\n\n{context}",
        },
    ]

    # 尝试 LangChain invoke
    if hasattr(llm, "invoke"):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            lc_messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"## 用户任务\n\n{user_query}\n\n## 文档内容\n\n{context}"
                ),
            ]
            response = llm.invoke(lc_messages)
            if hasattr(llm, "used_fallback"):
                used_fallback = llm.used_fallback
            text = response.content if hasattr(response, "content") else str(response)
            return text, used_fallback
        except Exception as exc:
            logger.warning("LangChain invoke 失败，回退到直接调用: %s", exc)

    # 兜底方案：直接调用
    try:
        response = llm(messages)
        if hasattr(llm, "used_fallback"):
            used_fallback = llm.used_fallback
        text = response.content if hasattr(response, "content") else str(response)
        return text, used_fallback
    except Exception as exc:
        logger.error("LLM 调用完全失败: %s", exc)
        text = str(llm._response) if hasattr(llm, "_response") else "Error"
        return text, False


def _local_analysis(query: str, text: str) -> str:
    """在不使用云端 LLM 的情况下执行基本本地分析。

    Args:
        query: 用户查询。
        text: 文档文本。

    Returns:
        基本分析结果。
    """
    lines = text.strip().split("\n")
    word_count = len(text)
    line_count = len(lines)
    paragraph_count = len([l for l in lines if l.strip()])

    # 检测一些模式
    has_todos = "待办" in text or "TODO" in text
    has_dates = any(
        char.isdigit() for char in text if char.isdigit()
    )

    result = f"""## 本地分析结果

> ⚠️ 本地兜底模式：未使用云端 LLM，以下为基于规则的本地分析。

### 基本信息

- 总字符数：{word_count}
- 行数：{line_count}
- 段落数：{paragraph_count}

### 内容预览

{text[:300]}{'...' if len(text) > 300 else ''}

### 检测结果

- 包含待办事项：{'是' if has_todos else '未检测到'}
- 包含日期/数字：{'是' if has_dates else '未检测到'}
"""
    return result
