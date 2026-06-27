"""Agent orchestration — the main run_task entry point."""
import time
import uuid
from typing import Any, List

from src.agent.prompts import SYSTEM_PROMPT
from src.observability.run_log import RunLog, StepLog
from src.tools.cost import check_limits, estimate_cost, estimate_tokens
from src.tools.file_ops import read_file as _file_read
from src.tools.file_ops import save_file as _file_save
from src.tools.redaction import detect_sensitive, redact
from src.workflow.postprocess import format_output, validate_save_path
from src.workflow.preprocess import preprocess
from src.workflow.upload_policy import decide_upload_strategy, generate_preview


def create_agent_with_tools(llm, tools: List):
    """Create a LangChain agent with the given tools.

    Args:
        llm: A LangChain-compatible chat model.
        tools: List of LangChain Tool objects.

    Returns:
        An agent executor (or the llm with bound tools for simple cases).
    """
    # For v0.1, we use a simple approach: bind tools to LLM
    # Full LangChain create_agent requires more setup
    if hasattr(llm, "bind_tools"):
        return llm.bind_tools(tools)
    return llm


def run_task(
    query: str,
    files: List[str],
    mode: str,
    config,
    llm: Any = None,
    auto_confirm: bool = False,
) -> RunLog:
    """Execute a complete workflow: preprocess → upload_policy → LLM → postprocess.

    This is the main entry point for all user tasks.

    Args:
        query: User's natural language request.
        files: List of file paths to process.
        mode: One of 'quick', 'privacy_enhanced', 'manual_confirm', 'local_fallback'.
        config: AppConfig instance.
        llm: LangChain-compatible chat model. If None, uses a placeholder.
        auto_confirm: Skip user confirmation step (for testing/automation).

    Returns:
        RunLog with full execution trace.
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

    # ── Step 1: Preprocess files ──
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

    # ── Step 2: Upload policy decision ──
    t0 = time.time()
    combined_text = "\n\n".join(doc.raw_text for doc in documents)
    # Use the first document's sensitive matches for policy decision
    primary_doc = documents[0]
    # Merge sensitive matches from all documents
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
            output_preview=f"strategy={decision.strategy}, tokens_in_estimate={preview.tokens_in_estimate}",
            duration_ms=duration_ms,
            tokens_in=preview.tokens_in_estimate,
            cost_yuan=preview.cost_estimate,
            status="success",
        )
    )

    # ── Step 2.5: Check limits ──
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

    # ── Step 3: LLM call (or local fallback) ──
    if decision.strategy == "blocked":
        # Local fallback: no cloud call
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
        # Cloud LLM call
        t0 = time.time()
        try:
            result_text = _call_llm(
                llm=llm,
                system_prompt=SYSTEM_PROMPT,
                user_query=query,
                context=combined_text,
                strategy=decision.strategy,
                redact_map=decision.redact_map,
            )
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
        # Estimate output tokens
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

    # ── Step 4: Postprocess ──
    t0 = time.time()
    formatted = format_output(result_text)
    output_path = f"output/{run_id}.md"
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

    # ── Step 5: Save result ──
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

    # ── Update totals (computed properties, no need to set) ──
    # total_tokens_in, total_tokens_out, total_cost_yuan are computed from steps

    return run_log


def _call_llm(
    llm,
    system_prompt: str,
    user_query: str,
    context: str,
    strategy: str,
    redact_map: dict,
) -> str:
    """Call the LLM with the given context.

    Args:
        llm: The LLM instance (or FakeLLM for testing).
        system_prompt: System prompt text.
        user_query: User's natural language query.
        context: Document content.
        strategy: Upload strategy used.
        redact_map: Redaction mapping.

    Returns:
        LLM response text.
    """
    if llm is None:
        return "（未配置 LLM，无法生成回复）"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"## 用户任务\n\n{user_query}\n\n## 文档内容\n\n{context}",
        },
    ]

    # Try LangChain invoke
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
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception:
            pass

    # Fallback: call directly
    try:
        response = llm(messages)
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception:
        return str(llm._response) if hasattr(llm, "_response") else "Error"


def _local_analysis(query: str, text: str) -> str:
    """Perform basic local analysis without cloud LLM.

    Args:
        query: User query.
        text: Document text.

    Returns:
        Basic analysis result.
    """
    lines = text.strip().split("\n")
    word_count = len(text)
    line_count = len(lines)
    paragraph_count = len([l for l in lines if l.strip()])

    # Detect some patterns
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
