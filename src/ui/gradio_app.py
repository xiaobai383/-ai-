"""Gradio UI for the Personal AI Workflow Assistant — v1.0 4-tab layout."""
from typing import List

import gradio as gr

from src.agent.app import run_task
from src.config import AppConfig


def build_ui(config: AppConfig):
    """Build and return the Gradio Blocks UI with 4 tabs.

    Args:
        config: Application configuration.

    Returns:
        A gradio.Blocks instance ready to launch.
    """
    css = """
    .upload-preview { background: #f5f5f5; padding: 12px; border-radius: 8px; font-family: monospace; }
    .sensitive-warn { color: #d32f2f; font-weight: bold; }
    .stat-card { background: #f0f0ff; padding: 16px; border-radius: 8px; text-align: center; }
    .stat-value { font-size: 28px; font-weight: bold; color: #6366f1; }
    .stat-label { font-size: 12px; color: #666; }
    """

    with gr.Blocks(title="个人 AI 工作流助手", css=css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 个人 AI 工作流助手 v1.0")
        gr.Markdown("云端智力，本地边界 — 可控上传、可审计执行 · 向量检索 · 模型兜底 · 仪表盘")

        # ============================================================
        # Tab 1: 工作台 — 单任务执行 + 执行回放
        # ============================================================
        with gr.Tabs():
            with gr.TabItem("🛠️ 工作台"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="任务描述",
                            placeholder="例如：总结这篇文档的要点，提取待办事项...",
                            lines=3,
                        )
                        mode_radio = gr.Radio(
                            choices=[
                                ("快速模式", "quick"),
                                ("隐私增强模式（推荐）", "privacy_enhanced"),
                                ("手动确认模式", "manual_confirm"),
                                ("本地兜底模式", "local_fallback"),
                            ],
                            value="privacy_enhanced",
                            label="运行模式",
                            info="隐私增强模式：本地检测敏感信息并自动脱敏后再上传",
                        )

                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="选择文件",
                            file_types=[".txt", ".md"],
                            file_count="multiple",
                        )
                        run_button = gr.Button("▶ 开始执行", variant="primary")

                with gr.Accordion("上传预览", open=True):
                    preview_panel = gr.Textbox(
                        label="即将发送的内容",
                        lines=8,
                        interactive=False,
                        elem_classes=["upload-preview"],
                    )

                with gr.Accordion("执行日志", open=True):
                    progress_output = gr.Textbox(
                        label="执行进度",
                        lines=6,
                        interactive=False,
                    )

                with gr.Accordion("结果", open=True):
                    result_output = gr.Markdown("等待执行...")

                with gr.Accordion("RunLog 摘要", open=False):
                    runlog_output = gr.Textbox(
                        label="执行摘要（token / 费用 / 耗时）",
                        lines=10,
                        interactive=False,
                    )

                # ── Event handlers ──

                def on_file_upload(files):
                    if not files:
                        return "（未选择文件）"
                    lines = []
                    for f in files:
                        lines.append(f"📄 {f.name} ({f.size} bytes)")
                    return "\n".join(lines)

                file_input.change(fn=on_file_upload, inputs=[file_input], outputs=[preview_panel])

                def on_run(query, mode, files):
                    if not query.strip():
                        yield "❌ 请输入任务描述", "", "## 错误\n\n请输入任务描述", ""
                        return
                    if not files:
                        yield "❌ 请选择至少一个文件", "", "## 错误\n\n请选择至少一个文件", ""
                        return

                    file_paths = [f.name for f in files]
                    progress_lines: List[str] = []

                    def progress_callback(msg: str):
                        progress_lines.append(msg)

                    progress_callback("⏳ 正在解析文件...")

                    try:
                        llm = None
                        if config.api_key and "sk-fake" not in config.api_key:
                            from src.fallback.provider import FallbackChatModel

                            if config.fallback_enabled:
                                llm = FallbackChatModel(
                                    primary_model=config.model_name,
                                    primary_base_url=config.model_base_url,
                                    api_key=config.api_key,
                                    fallback_base_url=config.fallback_ollama_base_url,
                                    fallback_model=config.fallback_ollama_model,
                                    timeout=config.fallback_timeout_seconds,
                                )
                            else:
                                from langchain_openai import ChatOpenAI
                                llm = ChatOpenAI(
                                    model=config.model_name,
                                    api_key=config.api_key,
                                    base_url=config.model_base_url,
                                    temperature=0.3,
                                    model_kwargs={"extra_body": {"thinking": {"type": "enabled"}}},
                                )

                        run_log = run_task(
                            query=query,
                            files=file_paths,
                            mode=mode,
                            config=config,
                            llm=llm,
                            auto_confirm=True,
                        )

                        # Persist the RunLog for replay
                        try:
                            run_log.save_to_disk("data/logs")
                        except Exception:
                            pass

                        for step in run_log.steps:
                            icon = "✅" if step.status == "success" else "❌"
                            progress_callback(
                                f"{icon} {step.name} — {step.output_preview} ({step.duration_ms}ms)"
                            )

                        result_md = "## 执行完成\n\n"
                        if run_log.result_path:
                            from pathlib import Path
                            result_file = Path(run_log.result_path)
                            if result_file.exists():
                                result_md = result_file.read_text(encoding="utf-8")
                            else:
                                result_md += f"结果已保存到 `{run_log.result_path}`"
                        else:
                            result_md += "（未生成结果文件）"

                        if run_log.fallback:
                            result_md = "⚠️ 本次执行使用了本地模型兜底（云端 API 不可用）\n\n" + result_md

                        log_summary = (
                            f"Run ID: {run_log.run_id}\n"
                            f"模式: {run_log.mode}\n"
                            f"模型: {run_log.model}\n"
                            f"步骤数: {len(run_log.steps)}\n"
                            f"输入 token: {run_log.total_tokens_in}\n"
                            f"输出 token: {run_log.total_tokens_out}\n"
                            f"预计费用: ¥{run_log.total_cost_yuan:.6f}\n"
                            f"本地兜底: {'是' if run_log.fallback else '否'}\n"
                            f"结果路径: {run_log.result_path or 'N/A'}\n"
                        )

                        yield (
                            "\n".join(progress_lines),
                            "",
                            result_md,
                            log_summary,
                        )

                    except Exception as e:
                        progress_callback(f"❌ 错误: {str(e)}")
                        yield (
                            "\n".join(progress_lines),
                            "",
                            f"## 执行失败\n\n```\n{str(e)}\n```",
                            f"错误: {str(e)}",
                        )

                run_button.click(
                    fn=on_run,
                    inputs=[query_input, mode_radio, file_input],
                    outputs=[progress_output, preview_panel, result_output, runlog_output],
                )

                # ── 执行回放（内嵌在工作台）──
                gr.Markdown("---")
                gr.Markdown("### 📜 执行回放")

                with gr.Row():
                    replay_search = gr.Textbox(label="搜索任务", placeholder="输入关键词筛选...", scale=2)
                    replay_mode_filter = gr.Dropdown(
                        choices=["全部", "quick", "privacy_enhanced", "manual_confirm", "local_fallback"],
                        value="全部",
                        label="模式筛选",
                        scale=1,
                    )
                    replay_status_filter = gr.Dropdown(
                        choices=["全部", "success", "failed"],
                        value="全部",
                        label="状态筛选",
                        scale=1,
                    )
                    replay_refresh = gr.Button("🔍 刷新", variant="secondary", scale=1)

                replay_output = gr.HTML(label="执行历史")

                def _render_replay(search, mode_filter, status_filter):
                    """Render replay timeline as HTML."""
                    from src.replay.loader import RunLogLoader

                    loader = RunLogLoader()
                    mode = None if mode_filter == "全部" else mode_filter
                    status = None if status_filter == "全部" else status_filter
                    result = loader.list_all(
                        search=search or None,
                        mode_filter=mode,
                        status_filter=status,
                        limit=20,
                    )

                    if not result.items:
                        return "<p style='color:#999'>暂无执行记录</p>"

                    html = '<div style="max-height:500px;overflow-y:auto">'
                    for item in result.items:
                        fb_badge = " ⚠️ 本地兜底" if item.fallback else ""
                        status_icon = "✅" if item.status == "success" else "❌"

                        cost_str = f"¥{item.total_cost_yuan:.4f}" if item.total_cost_yuan > 0 else "¥0"
                        html += f"""
                        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px;margin:6px 0;background:#fff">
                            <div style="display:flex;justify-content:space-between;align-items:center">
                                <strong>{status_icon} {item.user_query[:50]}</strong>
                                <span style="font-size:12px;color:#999">{item.run_id}</span>
                            </div>
                            <div style="font-size:12px;color:#666;margin-top:4px">
                                模式: {item.mode} | 模型: {item.model} | 步骤: {item.step_count}步
                                | Token: {item.total_tokens_in}→{item.total_tokens_out}
                                | 费用: {cost_str}{fb_badge}
                            </div>
                        </div>"""
                    html += "</div>"
                    html += f"<p style='font-size:12px;color:#999;margin-top:8px'>共 {result.total} 条记录</p>"
                    return html

                replay_refresh.click(
                    fn=_render_replay,
                    inputs=[replay_search, replay_mode_filter, replay_status_filter],
                    outputs=[replay_output],
                )

                # Load replay on tab switch (initial load via page load triggers refresh)
                replay_search.change(
                    fn=_render_replay,
                    inputs=[replay_search, replay_mode_filter, replay_status_filter],
                    outputs=[replay_output],
                )

            # ============================================================
            # Tab 2: 知识库 — 向量检索 + 索引管理
            # ============================================================
            with gr.TabItem("📚 知识库"):
                gr.Markdown("### 🔍 语义检索")

                with gr.Row():
                    search_query = gr.Textbox(
                        label="搜索内容",
                        placeholder="例如：上次风险分析的结果是什么？...",
                        lines=2,
                        scale=3,
                    )
                    search_collection = gr.Dropdown(
                        choices=["全部", "runlog", "outputs"],
                        value="全部",
                        label="搜索范围",
                        scale=1,
                    )
                    search_topk = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="结果数量", scale=1)
                    search_button = gr.Button("搜索", variant="primary", scale=1)

                search_results = gr.HTML(label="检索结果")

                def _do_search(query, collection, top_k):
                    if not query.strip():
                        return "<p style='color:#999'>请输入搜索关键词</p>"

                    from src.knowledge.embedder import OllamaEmbedder
                    from src.knowledge.search import Searcher
                    from src.knowledge.store import KnowledgeStore

                    store = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
                    embedder = OllamaEmbedder(
                        base_url=config.knowledge_embed_base_url,
                        model=config.knowledge_embed_model,
                    )
                    searcher = Searcher(store, embedder)

                    colls = None if collection == "全部" else [collection]
                    resp = searcher.search(query, top_k=top_k, collections=colls)

                    if not resp.embedding_available:
                        return """
                        <div style="padding:20px;background:#fef3c7;border-radius:8px">
                            <strong>⚠️ Ollama 未连接</strong>
                            <p>请确保 Ollama 已启动且模型 <code>nomic-embed-text</code> 已加载：</p>
                            <pre>ollama pull nomic-embed-text</pre>
                        </div>"""

                    if not resp.results:
                        return "<p style='color:#999'>未找到相关结果</p>"

                    html = "<div>"
                    for r in resp.results:
                        score_pct = int(r.score * 100)
                        html += f"""
                        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px;margin:6px 0;background:#fff">
                            <div style="display:flex;justify-content:space-between">
                                <span style="font-size:12px;color:#6366f1;font-weight:bold">相关性 {score_pct}%</span>
                                <span style="font-size:11px;color:#999">{r.doc_type} · {r.source}</span>
                            </div>
                            <div style="margin-top:6px;font-size:13px;max-height:120px;overflow-y:auto;white-space:pre-wrap">{r.document[:500]}</div>
                        </div>"""
                    html += f"<p style='font-size:12px;color:#999;margin-top:8px'>共 {resp.total_hits} 条匹配</p></div>"
                    return html

                search_button.click(
                    fn=_do_search,
                    inputs=[search_query, search_collection, search_topk],
                    outputs=[search_results],
                )

                gr.Markdown("---")
                gr.Markdown("### 📥 索引管理")

                with gr.Row():
                    index_button = gr.Button("🔄 重建索引", variant="secondary")
                    index_status = gr.Textbox(label="索引状态", lines=3, interactive=False)

                def _rebuild_index():
                    from src.knowledge.embedder import OllamaEmbedder
                    from src.knowledge.indexer import Indexer
                    from src.knowledge.store import KnowledgeStore

                    store = KnowledgeStore(persist_dir=config.knowledge_chroma_dir)
                    embedder = OllamaEmbedder(
                        base_url=config.knowledge_embed_base_url,
                        model=config.knowledge_embed_model,
                    )

                    if not embedder.is_available():
                        return "❌ Ollama 不可用，无法生成 embedding\n请先启动 Ollama 并加载模型: ollama pull nomic-embed-text"

                    indexer = Indexer(store, embedder)
                    result = indexer.index_all()

                    lines = [
                        f"✅ 索引完成",
                        f"  Logs: +{result['logs'].get('added', 0)} 跳过 {result['logs'].get('skipped', 0)}",
                        f"  Outputs: +{result['outputs'].get('added', 0)} 跳过 {result['outputs'].get('skipped', 0)}",
                    ]
                    return "\n".join(lines)

                index_button.click(fn=_rebuild_index, inputs=[], outputs=[index_status])

            # ============================================================
            # Tab 3: 洞察 — 仪表盘 + 通知历史
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

                # KPI cards
                with gr.Row():
                    dash_total = gr.Textbox(label="📊 总任务", value="...", interactive=False)
                    dash_cost = gr.Textbox(label="💰 总费用", value="...", interactive=False)
                    dash_tokens = gr.Textbox(label="🔤 Token 消耗", value="...", interactive=False)
                    dash_fallback = gr.Textbox(label="⚠️ 兜底次数", value="...", interactive=False)

                # Charts
                with gr.Row():
                    dash_cost_chart = gr.Plot(label="费用趋势")
                with gr.Row():
                    dash_mode_pie = gr.Plot(label="模式分布", scale=1)
                    dash_model_pie = gr.Plot(label="模型分布", scale=1)
                with gr.Row():
                    dash_token_bar = gr.Plot(label="Token 消耗")

                # Recent tasks
                gr.Markdown("#### 最近任务")
                dash_recent = gr.HTML(label="最近执行")

                def _refresh_dashboard(days):
                    from src.dashboard.aggregator import DashboardAggregator
                    from src.dashboard.charts import (
                        cost_trend_chart,
                        mode_pie_chart,
                        model_pie_chart,
                        token_bar_chart,
                    )

                    agg = DashboardAggregator()
                    stats = agg.aggregate(days=int(days))

                    total = str(stats.total_tasks)
                    cost = f"¥{stats.total_cost_yuan:.4f}"
                    tokens = f"{stats.total_tokens_in}→{stats.total_tokens_out}"
                    fb = str(stats.fallback_count)

                    # Build recent tasks HTML
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
                        token_bar_chart(stats.daily),
                        html,
                    )

                dash_refresh.click(
                    fn=_refresh_dashboard,
                    inputs=[dash_days],
                    outputs=[
                        dash_total, dash_cost, dash_tokens, dash_fallback,
                        dash_cost_chart, dash_mode_pie, dash_model_pie, dash_token_bar,
                        dash_recent,
                    ],
                )

                # ── 通知历史 ──
                gr.Markdown("---")
                gr.Markdown("### 🔔 通知历史")
                notify_output = gr.HTML(label="最近通知")

                def _load_notifications():
                    from pathlib import Path
                    import json

                    log_path = Path(config.notifications_log_file)
                    if not log_path.exists():
                        return "<p style='color:#999'>暂无通知记录</p>"

                    try:
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
            # Tab 4: 设置 — 配置 + 模板 + 偏好
            # ============================================================
            with gr.TabItem("⚙️ 设置"):
                gr.Markdown("### 当前配置")

                config_output = gr.Textbox(
                    label="config.yaml 摘要",
                    value=f"模型: {config.model_name}\n"
                          f"API: {config.model_base_url}\n"
                          f"文件上限: {config.max_file_size_mb}MB\n"
                          f"Token 上限: {config.max_tokens_per_request}\n"
                          f"费用上限: ¥{config.max_cost_per_request_yuan}\n"
                          f"脱敏: {'启用' if config.redaction_enabled else '禁用'}\n"
                          f"兜底模型: {config.fallback_ollama_model if config.fallback_enabled else '禁用'}\n"
                          f"ChromaDB: {config.knowledge_chroma_dir}\n"
                          f"Embedding: {config.knowledge_embed_model}",
                    lines=12,
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("### 📋 工作流模板")

                template_list = gr.Textbox(
                    label="可用模板",
                    value=_load_template_list(config),
                    lines=6,
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("### 🎨 用户偏好")

                pref_output = gr.Textbox(
                    label="偏好设置",
                    value=_load_preferences(config),
                    lines=6,
                    interactive=False,
                )

    return demo


def _load_template_list(config: AppConfig) -> str:
    """Load and format available workflow templates."""
    try:
        from src.workflow.templates import list_templates
        templates = list_templates()
        lines = []
        for t in templates:
            lines.append(f"  • {t.get('name', '?')} — {t.get('description', '')[:60]}")
        return "\n".join(lines) if lines else "  无可用模板"
    except Exception:
        return "  无法加载模板"


def _load_preferences(config: AppConfig) -> str:
    """Load and format user preferences."""
    try:
        from src.preferences.manager import load
        prefs = load()
        lines = [
            f"  默认模式: {prefs.get('default_mode', 'privacy_enhanced')}",
            f"  输出格式: {prefs.get('default_output_format', 'markdown')}",
            f"  自动预览: {'是' if prefs.get('auto_preview', True) else '否'}",
            f"  显示 Token: {'是' if prefs.get('show_token_count', True) else '否'}",
        ]
        return "\n".join(lines)
    except Exception:
        return "  无法加载偏好"


def launch_ui(config: AppConfig | None = None):
    """Launch the Gradio UI.

    Args:
        config: Optional AppConfig. Loads from default locations if None.
    """
    if config is None:
        config = AppConfig.from_yaml_and_env()

    demo = build_ui(config)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
