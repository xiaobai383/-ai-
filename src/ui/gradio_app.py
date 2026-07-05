"""个人 AI 工作流助手 Gradio 界面 — 豆包式会话应用。

布局：左侧会话列表 + 右侧对话区（模式/文件/Chatbot/输入）+ 洞察 + 设置。
会话消息用 JSON 存储（SessionStore），上传文件走 ChromaDB 向量检索（RAG 注入）。
"""
import logging
import tempfile
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import List

import gradio as gr

from src.config import AppConfig
from src.knowledge.session_store import SessionStore

logger = logging.getLogger(__name__)


# ── 模式常量 ──
_DEFAULT_MODE = "privacy_enhanced"


def build_ui(config: AppConfig):
    """构建并返回豆包式会话 Gradio 界面。"""
    css = """
    .stat-card { background: #f0f0ff; padding: 12px; border-radius: 8px; text-align: center; }
    .stat-value { font-size: 24px; font-weight: bold; color: #6366f1; }
    .stat-label { font-size: 12px; color: #666; }
    .session-item { padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    .status-bar { background: #f0f0ff; padding: 8px 16px; border-radius: 8px; font-size: 13px; }
    """

    with gr.Blocks(title="个人 AI 工作流助手", css=css) as demo:
        gr.Markdown("# 个人 AI 工作流助手")
        # 顶部费用/余额状态条
        status_bar = gr.HTML(_render_status_bar(config))

        with gr.Tabs():
            # ============================================================
            # Tab 1: 对话 — 豆包式会话（左侧列表 + 右侧对话区）
            # ============================================================
            with gr.TabItem("💬 对话"):
                with gr.Row():
                    # ── 左侧：会话列表 ──
                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("### 会话列表")
                        new_btn = gr.Button("➕ 新建会话", variant="primary", size="sm")
                        session_radio = gr.Radio(
                            choices=_session_choices(config),
                            label="",
                            container=False,
                        )
                        with gr.Row():
                            rename_btn = gr.Button("✏️ 重命名", variant="secondary", size="sm")
                            delete_btn = gr.Button("🗑️ 删除", variant="secondary", size="sm")
                        rename_input = gr.Textbox(
                            label="新名称",
                            visible=False,
                            lines=1,
                        )

                    # ── 右侧：对话区 ──
                    with gr.Column(scale=4):
                        session_header = gr.HTML(
                            '<div class="status-bar">请在左侧选择或新建会话</div>'
                        )
                        # 文件上传（当前会话级）
                        with gr.Accordion("📎 上传文件（可选，对话时自动检索）", open=False):
                            file_input = gr.File(
                                label="选择文件（.txt/.md）",
                                file_types=[".txt", ".md"],
                                file_count="multiple",
                            )
                            upload_status = gr.Textbox(
                                label="上传状态",
                                interactive=False,
                                lines=2,
                            )

                        # 对话气泡
                        chatbot = gr.Chatbot(
                            label="对话",
                            height=420,
                        )

                        with gr.Row():
                            chat_input = gr.Textbox(
                                label="输入消息",
                                placeholder="输入问题，回车或点击发送...",
                                lines=2,
                                scale=4,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)

                    # ── 隐藏状态 ──
                    current_session = gr.State(None)  # 当前 session_id

                # ════════════ 事件处理 ════════════

                def _on_new_session():
                    """新建会话：创建 → 刷新列表 → 自动选中 → 清空对话。"""
                    store = SessionStore()
                    s = store.create_session(mode=_DEFAULT_MODE)
                    choices = _session_choices(config)
                    header = _render_session_header(s)
                    return (
                        gr.update(choices=choices, value=s["id"]),  # session_radio
                        header,  # session_header
                        [],  # chatbot 清空
                        s["id"],  # current_session
                        "",  # chat_input
                        "新会话已创建",  # upload_status
                        _render_status_bar(config, session_id=s["id"]),  # status_bar
                    )

                def _on_switch_session(session_id):
                    """切换会话：加载消息渲染到 Chatbot + 刷新当前会话费用。"""
                    if not session_id:
                        return [], '<div class="status-bar">请选择会话</div>', session_id, "", _render_status_bar(config)
                    store = SessionStore()
                    s = store.get_session(session_id)
                    if not s:
                        return [], '<div class="status-bar">会话不存在</div>', session_id, "", _render_status_bar(config)
                    msgs = store.get_messages(session_id)
                    # 渲染为 Chatbot messages 格式（展示时还原脱敏占位符）
                    from src.workflow.postprocess import restore_redactions
                    display = []
                    for m in msgs:
                        content = m["content"]
                        rm = m.get("redact_map")
                        if rm:
                            content = restore_redactions(content, rm)
                        display.append({"role": m["role"], "content": content})
                    header = _render_session_header(s)
                    return display, header, session_id, "", _render_status_bar(config, session_id=session_id)

                def _on_upload(files, session_id):
                    """文件上传后向量化存 ChromaDB（带 session_id，供 RAG 检索）。"""
                    if not session_id:
                        return "⚠️ 请先在左侧新建或选择一个会话，再上传文件"
                    if not files:
                        return "未选择文件"

                    # Gradio 临时目录加入白名单
                    tmp_dir = str(Path(tempfile.gettempdir()))
                    if tmp_dir not in config.allowed_paths:
                        config.allowed_paths.append(tmp_dir)

                    try:
                        added = _index_uploaded_files(files, session_id, config, mode=_DEFAULT_MODE)
                        if added == 0:
                            return "⚠️ 索引了 0 个文件块，请检查文件内容是否为空或 Ollama 是否正常运行"
                        return f"✅ 已索引 {added} 个文件块（已脱敏）到当前会话，对话时会自动检索相关内容"
                    except Exception as exc:
                        return f"⚠️ 索引失败：{exc}"

                def _on_send(message, session_id, files):
                    """发送消息：历史拼接 + 文件RAG注入 → 流式生成 → 持久化 → 清空输入。

                    outputs: [chatbot, session_radio, current_session, chat_input, upload_status, status_bar]
                    前几次 yield 中 status_bar 用 gr.update() 不更新，最后一次刷新费用。
                    """
                    message = (message or "").strip()
                    if not session_id:
                        yield [], '<div class="status-bar">请先新建或选择会话</div>', session_id, "", "", gr.update()
                        return
                    if not message:
                        store = SessionStore()
                        msgs = store.get_messages(session_id)
                        from src.workflow.postprocess import restore_redactions
                        display = []
                        for m in msgs:
                            content = m["content"]
                            rm = m.get("redact_map")
                            if rm:
                                content = restore_redactions(content, rm)
                            display.append({"role": m["role"], "content": content})
                        yield display, gr.update(), session_id, "", "", gr.update()
                        return

                    store = SessionStore()
                    # 取历史消息（不含当前）
                    history = store.get_messages(session_id)
                    from src.workflow.postprocess import restore_redactions
                    history_display = []
                    for m in history:
                        content = m["content"]
                        rm = m.get("redact_map")
                        if rm:
                            content = restore_redactions(content, rm)
                        history_display.append({"role": m["role"], "content": content})

                    # 渲染：历史 + 当前用户消息 + 空 assistant 占位
                    history_display.append({"role": "user", "content": message})
                    history_display.append({"role": "assistant", "content": ""})
                    yield history_display, gr.update(), session_id, "", "", gr.update()

                    # ── 脱敏用户输入（默认隐私增强）──
                    from src.tools.redaction import detect_sensitive, redact
                    user_redact_map = {}
                    if config.redaction_enabled:
                        matches = detect_sensitive(message, config.redaction_rules)
                        if matches:
                            message, user_redact_map = redact(message, matches)

                    # 中间状态：告知用户正在处理
                    history_display[-1] = {"role": "assistant", "content": "⏳ **正在检索和分析，请稍候...**"}
                    yield history_display, gr.update(), session_id, "", "", gr.update()
                    yield history_display, gr.update(), session_id, "", "", gr.update()

                    # 意图感知检索（含 query 改写、意图路由、长期记忆）
                    from src.agent.chat import chat_stream, smart_retrieve
                    llm = _build_chat_llm(config)

                    file_context, long_term_memory, file_redact_map = smart_retrieve(
                        session_id, message, config, llm, history
                    )
                    merged_redact_map = {**user_redact_map, **file_redact_map}

                    # 流式生成（usage 收集 token 用量供费用记录）
                    accumulated = ""
                    usage = {}
                    used_fallback = False
                    try:
                        for chunk in chat_stream(
                            history, message, llm, file_context, long_term_memory, usage_out=usage
                        ):
                            accumulated += chunk
                            # 展示时还原脱敏占位符
                            display_text = restore_redactions(accumulated, merged_redact_map)
                            history_display[-1] = {"role": "assistant", "content": display_text}
                            yield history_display, gr.update(), session_id, "", "", gr.update()
                        used_fallback = usage.get("used_fallback", False)
                    except Exception as exc:
                        accumulated = f"⚠️ 生成失败：{exc}"
                        history_display[-1] = {"role": "assistant", "content": accumulated}
                        yield history_display, gr.update(), session_id, "", "", gr.update()

                    # API 调用失败、自动切换本地模型时，给用户提示
                    if used_fallback:
                        fallback_notice = "\n\n> ⚠️ **API 调用失败，已自动切换为本地模型**（回答质量可能降低）"
                        display_text = restore_redactions(accumulated, merged_redact_map) + fallback_notice
                        history_display[-1] = {"role": "assistant", "content": display_text}
                        yield history_display, gr.update(), session_id, "", "", gr.update()

                    # 持久化：存储脱敏后内容 + redact_map（敏感信息不落库）
                    store.add_message(session_id, "user", message, redact_map=user_redact_map or None)
                    store.add_message(session_id, "assistant", accumulated, redact_map=merged_redact_map or None)

                    # 持久化到 conversation_memory 向量集合
                    if getattr(config, 'conversation_memory_enabled', True):
                        try:
                            from src.agent.memory import store_conversation_turn
                            from src.knowledge.embedder import OllamaEmbedder
                            from src.knowledge.store import KnowledgeStore
                            store_conv = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
                            embedder = OllamaEmbedder(
                                base_url=config.knowledge_embed_base_url,
                                model=config.knowledge_embed_model,
                            )
                            store_conversation_turn(store_conv, embedder, session_id, message, accumulated)
                        except Exception as exc:
                            logger.warning("对话记忆持久化失败: %s", exc)

                    # 记录本轮费用到 data/logs/chat_costs.jsonl（供 _compute_total_spent 实时统计）
                    _log_chat_cost(session_id, message, accumulated, usage, config)

                    # 更新会话列表（标题可能变了）+ 刷新当前会话费用状态条
                    new_choices = _session_choices(config)
                    yield history_display, gr.update(choices=new_choices, value=session_id), session_id, "", "", _render_status_bar(config, session_id=session_id)

                def _on_delete(session_id):
                    """删除当前会话。"""
                    if not session_id:
                        return gr.update(), '<div class="status-bar">请选择会话</div>', None, [], "未选择会话"
                    store = SessionStore()
                    store.delete_session(session_id)
                    choices = _session_choices(config)
                    return (
                        gr.update(choices=choices, value=None),
                        '<div class="status-bar">会话已删除，请选择或新建</div>',
                        None,
                        [],
                        "会话已删除",
                    )

                def _on_rename(session_id):
                    """显示重命名输入框。"""
                    if not session_id:
                        return gr.update(visible=False, value="")
                    store = SessionStore()
                    s = store.get_session(session_id)
                    return gr.update(visible=True, value=s["title"] if s else "")

                def _do_rename(session_id, new_title, _visibility):
                    """执行重命名。"""
                    if not session_id or not new_title.strip():
                        return gr.update(visible=False, value=""), gr.update(), gr.update()
                    store = SessionStore()
                    store.rename_session(session_id, new_title.strip())
                    s = store.get_session(session_id)
                    choices = _session_choices(config)
                    return (
                        gr.update(visible=False, value=""),
                        gr.update(choices=choices, value=session_id),
                        _render_session_header(s) if s else gr.update(),
                    )

                # ── 绑定事件 ──
                new_btn.click(
                    fn=_on_new_session,
                    inputs=[],
                    outputs=[session_radio, session_header, chatbot, current_session, chat_input, upload_status, status_bar],
                )
                session_radio.change(
                    fn=_on_switch_session,
                    inputs=[session_radio],
                    outputs=[chatbot, session_header, current_session, chat_input, status_bar],
                )
                file_input.change(
                    fn=_on_upload,
                    inputs=[file_input, current_session],
                    outputs=[upload_status],
                )
                send_btn.click(
                    fn=_on_send,
                    inputs=[chat_input, current_session, file_input],
                    outputs=[chatbot, session_radio, current_session, chat_input, upload_status, status_bar],
                )
                chat_input.submit(
                    fn=_on_send,
                    inputs=[chat_input, current_session, file_input],
                    outputs=[chatbot, session_radio, current_session, chat_input, upload_status, status_bar],
                )
                delete_btn.click(
                    fn=_on_delete,
                    inputs=[current_session],
                    outputs=[session_radio, session_header, current_session, chatbot, upload_status],
                )
                rename_btn.click(
                    fn=_on_rename,
                    inputs=[current_session],
                    outputs=[rename_input],
                )
                rename_input.submit(
                    fn=_do_rename,
                    inputs=[current_session, rename_input, rename_input],
                    outputs=[rename_input, session_radio, session_header],
                )

                # 页面初始化：如有会话自动选中第一个
                def _init_chat():
                    from src.workflow.postprocess import restore_redactions
                    choices = _session_choices(config)
                    if choices:
                        first_id = choices[0][1]
                        store = SessionStore()
                        s = store.get_session(first_id)
                        msgs = store.get_messages(first_id)
                        display = []
                        for m in msgs:
                            content = m["content"]
                            rm = m.get("redact_map")
                            if rm:
                                content = restore_redactions(content, rm)
                            display.append({"role": m["role"], "content": content})
                        return (
                            gr.update(choices=choices, value=first_id),
                            _render_session_header(s) if s else '<div class="status-bar">请选择会话</div>',
                            display,
                            first_id,
                        )
                    return (
                        gr.update(choices=choices),
                        '<div class="status-bar">请点击「新建会话」开始</div>',
                        [],
                        None,
                    )

                demo.load(
                    fn=_init_chat,
                    inputs=[],
                    outputs=[session_radio, session_header, chatbot, current_session],
                )

            # ============================================================
            # Tab 2: 洞察 — 仪表盘（优化：删 Token 趋势图，修 Token 显示）
            # ============================================================
            with gr.TabItem("📊 洞察"):
                gr.Markdown("### 📈 运行仪表盘")

                with gr.Row():
                    dash_days = gr.Dropdown(
                        choices=[("今天", 1), ("近7天", 7), ("近30天", 30), ("全部", 0)],
                        value=7,
                        label="时间范围",
                        scale=1,
                    )
                    dash_refresh = gr.Button("🔄 刷新", variant="secondary", scale=1)

                with gr.Row():
                    dash_total = gr.Textbox(label="📊 总任务", value="...", interactive=False)
                    dash_cost = gr.Textbox(label="💰 总费用", value="...", interactive=False)
                    dash_tokens = gr.Textbox(label="🔤 Token", value="...", interactive=False)
                    dash_fallback = gr.Textbox(label="⚠️ 兜底次数", value="...", interactive=False)

                with gr.Row():
                    dash_cost_chart = gr.Plot(label="费用趋势")
                with gr.Row():
                    dash_mode_pie = gr.Plot(label="模式分布", scale=1)
                    dash_model_pie = gr.Plot(label="模型分布", scale=1)

                gr.Markdown("#### 最近任务")
                dash_recent = gr.HTML(label="最近执行")

                def _refresh_dashboard(days):
                    from src.dashboard.aggregator import DashboardAggregator
                    from src.dashboard.charts import (
                        cost_trend_chart,
                        mode_pie_chart,
                        model_pie_chart,
                    )

                    agg = DashboardAggregator()
                    stats = agg.aggregate(days=int(days))

                    total = str(stats.total_tasks)
                    cost = f"¥{stats.total_cost_yuan:.4f}"
                    # 修复：原来是 "{in}→{out}"（显示 144018→5200），改为清晰的「输入/输出」格式
                    tokens = f"输入 {stats.total_tokens_in:,} / 输出 {stats.total_tokens_out:,}"
                    fb = str(stats.fallback_count)

                    html = '<div style="max-height:200px;overflow-y:auto">'
                    for t in stats.recent_tasks[:5]:
                        fb_b = " ⚠️兜底" if t.get("fallback") else ""
                        s = "✅" if t.get("status") == "success" else "❌"
                        html += f'<div style="font-size:12px;padding:4px 0;border-bottom:1px solid #f0f0f0">{s} {t.get("user_query","")} — ¥{t.get("cost_yuan",0)}{fb_b}</div>'
                    html += "</div>"

                    return (
                        total, cost, tokens, fb,
                        cost_trend_chart(stats.daily),
                        mode_pie_chart(stats.mode_distribution),
                        model_pie_chart(stats.model_distribution),
                        html,
                    )

                dash_refresh.click(
                    fn=_refresh_dashboard,
                    inputs=[dash_days],
                    outputs=[
                        dash_total, dash_cost, dash_tokens, dash_fallback,
                        dash_cost_chart, dash_mode_pie, dash_model_pie, dash_recent,
                    ],
                )

                gr.Markdown("---")
                gr.Markdown("### 🔔 通知历史")
                notify_output = gr.HTML(label="最近通知")

                def _load_notifications():
                    log_path = Path(config.notifications_log_file)
                    if not log_path.exists():
                        return "<p style='color:#999'>暂无通知记录</p>"
                    try:
                        import json
                        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
                        recent = lines[-20:]
                        html = '<div style="max-height:300px;overflow-y:auto">'
                        for line in reversed(recent):
                            if line.strip():
                                entry = json.loads(line)
                                ts = entry.get("timestamp", "")
                                html += f'<div style="font-size:12px;padding:4px 0;border-bottom:1px solid #f0f0f0">🔔 <strong>{entry.get("title","")}</strong>: {entry.get("message","")} <span style="color:#999">({ts})</span></div>'
                        html += "</div>"
                        return html
                    except Exception:
                        return "<p style='color:#999'>无法加载通知记录</p>"

                notify_button = gr.Button("🔄 刷新通知", variant="secondary")
                notify_button.click(fn=_load_notifications, inputs=[], outputs=[notify_output])

            # ============================================================
            # Tab 3: 设置 — 可编辑配置（删模板/偏好，配置前端可改同步后端）
            # ============================================================
            with gr.TabItem("⚙️ 设置"):
                gr.Markdown("### 配置管理（修改后点击保存生效）")

                with gr.Row():
                    with gr.Column():
                        cfg_model = gr.Textbox(label="模型名称", value=config.model_name)
                        cfg_base_url = gr.Textbox(label="API 地址", value=config.model_base_url)
                        cfg_api_key = gr.Textbox(
                            label="API Key",
                            value=config.api_key,
                            type="password",
                        )
                        cfg_budget = gr.Number(label="预算 (¥)", value=config.budget_yuan)

                with gr.Row():
                    with gr.Column():
                        cfg_max_tokens = gr.Number(label="单次 Token 上限", value=config.max_tokens_per_request)
                        cfg_max_cost = gr.Number(label="单次费用上限 (¥)", value=config.max_cost_per_request_yuan)
                        cfg_redaction = gr.Checkbox(label="启用脱敏", value=config.redaction_enabled)

                with gr.Row():
                    with gr.Column():
                        cfg_fb_enabled = gr.Checkbox(label="启用本地兜底", value=config.fallback_enabled)
                        cfg_fb_model = gr.Textbox(label="兜底模型", value=config.fallback_ollama_model)
                        cfg_fb_timeout = gr.Number(label="兜底超时(秒)", value=config.fallback_timeout_seconds)

                save_btn = gr.Button("💾 保存配置", variant="primary")
                save_status = gr.Textbox(label="保存状态", interactive=False, lines=2)

                def _save_config(
                    model, base_url, api_key, budget,
                    max_tokens, max_cost, redaction,
                    fb_enabled, fb_model, fb_timeout,
                ):
                    """保存配置到 YAML + 更新内存。"""
                    try:
                        config.model_name = model
                        config.model_base_url = base_url
                        # api_key 从 .env 管理，这里只在用户输入非空时更新
                        if api_key and "sk-fake" not in api_key:
                            config.api_key = api_key
                        config.budget_yuan = float(budget)
                        config.max_tokens_per_request = int(max_tokens)
                        config.max_cost_per_request_yuan = float(max_cost)
                        config.redaction_enabled = redaction
                        config.fallback_enabled = fb_enabled
                        config.fallback_ollama_model = fb_model
                        config.fallback_timeout_seconds = int(fb_timeout)

                        config.save_to_yaml()
                        return "✅ 配置已保存到 config.yaml 并已生效"
                    except Exception as exc:
                        return f"❌ 保存失败：{exc}"

                save_btn.click(
                    fn=_save_config,
                    inputs=[
                        cfg_model, cfg_base_url, cfg_api_key, cfg_budget,
                        cfg_max_tokens, cfg_max_cost, cfg_redaction,
                        cfg_fb_enabled, cfg_fb_model, cfg_fb_timeout,
                    ],
                    outputs=[save_status],
                )

    return demo


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════

def _session_choices(config: AppConfig) -> list:
    """返回会话列表 Radio choices [(显示文本, session_id), ...]。"""
    store = SessionStore()
    sessions = store.list_sessions()
    choices = []
    for s in sessions:
        title = s["title"]
        count = s.get("message_count", 0)
        label = f"{title} ({count}条)"
        choices.append((label, s["id"]))
    return choices


def _render_session_header(session: dict) -> str:
    """渲染会话头 HTML（标题 + 消息数 + 隐私状态）。"""
    if not session:
        return '<div class="status-bar">请选择会话</div>'
    count = session.get("message_count", 0)
    title = session["title"]
    return (
        f'<div class="status-bar">'
        f'<strong>{title}</strong> &nbsp;|&nbsp; '
        f'🔒 隐私增强 &nbsp;|&nbsp; '
        f'消息数: <strong>{count}</strong>'
        f'</div>'
    )


def _render_status_bar(config: AppConfig, session_id: str | None = None) -> str:
    """渲染顶部费用/余额状态条。"""
    total_spent = _compute_total_spent(config, session_id=session_id)
    balance, balance_src = _fetch_balance(config, session_id=session_id)
    balance_color = "#16a34a" if balance > 0 else "#dc2626"
    label = "会话费用" if session_id else "累计费用"
    return (
        f'<div class="status-bar">'
        f'💰 {label}: <strong>¥{total_spent:.4f}</strong> &nbsp;|&nbsp; '
        f'🪙 余额: <strong style="color:{balance_color}">¥{balance:.4f}</strong>'
        f'（{balance_src}）'
        f'</div>'
    )


def _index_uploaded_files(files, session_id: str, config: AppConfig, mode: str = "quick") -> int:
    """上传文件 → 解析 → 分块 → embed → 存 ChromaDB（带 session_id，供 RAG 检索）。

    根据 mode 处理敏感信息：
      - quick: 原始文本（当前行为）
      - privacy_enhanced / manual_confirm: 先脱敏再分块入库
      - local_fallback: 跳过，不索引

    Returns:
        索引的文件块数。
    """
    from src.knowledge.embedder import OllamaEmbedder
    from src.knowledge.indexer import Indexer
    from src.knowledge.store import COLLECTION_SESSION_FILES, KnowledgeStore

    if mode == "local_fallback":
        return 0

    store = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
    embedder = OllamaEmbedder(
        base_url=config.knowledge_embed_base_url,
        model=config.knowledge_embed_model,
    )
    col = store.get_or_create(COLLECTION_SESSION_FILES)

    added = 0
    for f in files:
        path = getattr(f, "path", None) or getattr(f, "name", None) or str(f)
        try:
            text = Path(path).read_text(encoding="utf-8")
        except Exception:
            continue
        if not text.strip():
            continue

        # privacy_enhanced / manual_confirm: 脱敏后入库，持久化 redact_map
        redact_map_json = None
        if mode in ("privacy_enhanced", "manual_confirm"):
            from src.tools.redaction import detect_sensitive, redact
            matches = detect_sensitive(text, config.redaction_rules)
            if matches:
                text, rm = redact(text, matches)
                import json as _json
                redact_map_json = _json.dumps(rm, ensure_ascii=False)

        # 父子分块：子块用于 embedding，父块用于召回时返回完整上下文
        child_size = getattr(config, 'hierarchical_chunk_child_size', 4000)
        parent_size = getattr(config, 'hierarchical_chunk_parent_size', 12000)
        hierarchical = Indexer._chunk_text_hierarchical(text, child_size, parent_size)
        for i, chunk_info in enumerate(hierarchical):
            chunk = chunk_info["child"]
            emb = embedder.embed(chunk)
            if emb is None:
                continue
            doc_id = f"{session_id}:{Path(path).name}:{i}"
            store.add(
                col,
                [chunk],
                [{"session_id": session_id, "source": Path(path).name,
                  "chunk_index": i,
                  "parent_content": chunk_info["parent"][:5000],
                  "redact_map": redact_map_json,
                  "chunk_version": Indexer.CHUNK_VERSION}],
                [doc_id],
                [emb],
            )
            added += 1
    return added


def _build_chat_llm(config: AppConfig):
    """构建对话用 LLM 实例（与工作台共用同一套构建逻辑）。

    优先 FallbackChatModel（云端优先 + Ollama 兜底）；
    未配置 api_key 时返回 None，对话路径会给出提示文本。
    """
    if not config.api_key or "sk-fake" in config.api_key:
        return None
    if config.fallback_enabled:
        from src.fallback.provider import FallbackChatModel

        return FallbackChatModel(
            primary_model=config.model_name,
            primary_base_url=config.model_base_url,
            api_key=config.api_key,
            fallback_base_url=config.fallback_ollama_base_url,
            fallback_model=config.fallback_ollama_model,
            timeout=config.fallback_timeout_seconds,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.model_name,
        api_key=config.api_key,
        base_url=config.model_base_url,
        temperature=0.3,
    )


def _log_chat_cost(session_id, query, response, usage, config: AppConfig):
    """记录对话费用到 data/logs/chat_costs.jsonl（供 _compute_total_spent 实时统计）。

    多轮对话路径不走 run_task（不写 RunLog），所以单独记录到 chat_costs.jsonl。
    格式与 RunLog 的 total_cost_yuan 字段对齐，_compute_total_spent 同时扫描两个来源。
    """
    import json as _json
    import time as _time

    from src.tools.cost import estimate_cost

    tokens_in = usage.get("tokens_in", 0)
    tokens_out = usage.get("tokens_out", 0)
    # token 数为 0 时（API 未返回 usage），费用记 0
    cost = estimate_cost(tokens_in, tokens_out, config.model_name) if (tokens_in or tokens_out) else 0.0

    entry = {
        "timestamp": int(_time.time() * 1000),
        "session_id": session_id,
        "query": query[:200],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_yuan": cost,
        "model": config.model_name,
        "fallback": usage.get("used_fallback", False),
    }
    log_path = Path("data/logs/chat_costs.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")


def _compute_total_spent(config: AppConfig, session_id: str | None = None) -> float:
    """扫描 data/logs/ 下费用记录计算累计费用。

    Args:
        session_id: 若提供，只统计该会话的 chat_costs；否则统计全部。
    """
    import json as _json
    logs_dir = Path("data/logs")
    total = 0.0
    if logs_dir.exists():
        # 来源1：run_task 的 RunLog（只在无 session_id 时统计）
        if not session_id:
            for fp in logs_dir.glob("run-*.jsonl"):
                try:
                    first = fp.read_text(encoding="utf-8").split("\n")[0]
                    total += float(_json.loads(first).get("total_cost_yuan", 0))
                except Exception:
                    pass
        # 来源2：多轮对话的 chat_costs（按 session_id 过滤）
        chat_cost_path = logs_dir / "chat_costs.jsonl"
        if chat_cost_path.exists():
            try:
                for line in chat_cost_path.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        entry = _json.loads(line)
                        if session_id and entry.get("session_id") != session_id:
                            continue
                        total += float(entry.get("cost_yuan", 0))
            except Exception:
                pass
    return total


def _fetch_balance(config: AppConfig, session_id: str | None = None) -> tuple[float, str]:
    """获取余额信息。返回 (余额, 来源标签)。"""
    from src.tools.cost import fetch_real_balance

    real = fetch_real_balance(config.api_key, config.model_base_url)
    if real is not None:
        return real, "DeepSeek API"
    total_spent = _compute_total_spent(config, session_id=session_id)
    return config.budget_yuan - total_spent, "本地估算"


def launch_ui(config: AppConfig | None = None):
    """启动 Gradio 界面。"""
    if config is None:
        config = AppConfig.from_yaml_and_env()

    demo = build_ui(config)
    demo.launch(
        server_name="127.0.0.1",
        server_port=config.gradio_port,
        share=False,
        show_error=True,
    )
