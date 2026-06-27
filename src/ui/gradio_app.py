"""Gradio UI for the Personal AI Workflow Assistant."""
from typing import List

import gradio as gr

from src.agent.app import run_task
from src.config import AppConfig


def build_ui(config: AppConfig):
    """Build and return the Gradio Blocks UI.

    Args:
        config: Application configuration.

    Returns:
        A gradio.Blocks instance ready to launch.
    """
    css = """
    .upload-preview {
        background: #f5f5f5;
        padding: 12px;
        border-radius: 8px;
        font-family: monospace;
    }
    .sensitive-warn {
        color: #d32f2f;
        font-weight: bold;
    }
    """

    with gr.Blocks(title="个人 AI 工作流助手") as demo:
        gr.Markdown("# 个人 AI 工作流助手")
        gr.Markdown("云端智力，本地边界 — 可控上传、可审计执行")

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
            """When files are uploaded, show a quick preview."""
            if not files:
                return "（未选择文件）"
            lines = []
            for f in files:
                lines.append(f"📄 {f.name} ({f.size} bytes)")
            return "\n".join(lines)

        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[preview_panel],
        )

        def on_run(query, mode, files):
            """Execute the workflow."""
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
                return "\n".join(progress_lines)

            progress_callback("⏳ 正在解析文件...")

            try:
                # Build a real LLM if api_key is configured
                llm = None
                if config.api_key and "sk-fake" not in config.api_key:
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=config.model_name,
                        api_key=config.api_key,
                        base_url=config.model_base_url,
                        temperature=0.3,
                        model_kwargs={
                            "extra_body": {"thinking": {"type": "enabled"}}
                        },
                    )

                run_log = run_task(
                    query=query,
                    files=file_paths,
                    mode=mode,
                    config=config,
                    llm=llm,
                    auto_confirm=True,
                )

                # Build progress from steps
                for step in run_log.steps:
                    icon = "✅" if step.status == "success" else "❌"
                    progress_callback(
                        f"{icon} {step.name} — {step.output_preview} "
                        f"({step.duration_ms}ms)"
                    )

                # Result
                result_md = "## 执行完成\n\n"
                if run_log.result_path:
                    # Read the saved result
                    from pathlib import Path
                    result_file = Path(run_log.result_path)
                    if result_file.exists():
                        result_md = result_file.read_text(encoding="utf-8")
                    else:
                        result_md += f"结果已保存到 `{run_log.result_path}`"
                else:
                    result_md += "（未生成结果文件）"

                # RunLog summary
                log_summary = (
                    f"Run ID: {run_log.run_id}\n"
                    f"模式: {run_log.mode}\n"
                    f"模型: {run_log.model}\n"
                    f"步骤数: {len(run_log.steps)}\n"
                    f"输入 token: {run_log.total_tokens_in}\n"
                    f"输出 token: {run_log.total_tokens_out}\n"
                    f"预计费用: ¥{run_log.total_cost_yuan:.6f}\n"
                    f"结果路径: {run_log.result_path or 'N/A'}\n"
                )

                yield (
                    "\n".join(progress_lines),
                    preview_panel.value if hasattr(preview_panel, 'value') else "",
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

    return demo


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
