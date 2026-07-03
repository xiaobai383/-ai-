"""后处理 —— 格式化 LLM 输出并验证保存路径。"""
import json
import re
from pathlib import Path
from typing import Any, Dict


def format_output(raw_output: str, fmt: str = "markdown", metadata: Dict[str, Any] | None = None) -> str:
    """清理并格式化 LLM 原始输出。

    支持多种输出格式：
    - markdown: 标准 Markdown 格式
    - plain: 无格式纯文本
    - json: 结构化 JSON 输出
    - html: 基础 HTML 格式

    参数：
        raw_output: LLM 返回的原始文本。
        fmt: 输出格式 —— 'markdown'、'plain'、'json'、'html'。
        metadata: 可选的元数据，附加到输出中。

    返回：
        格式化后的文本。
    """
    if not raw_output or not raw_output.strip():
        return ""

    if fmt == "markdown":
        return _format_markdown(raw_output, metadata)
    elif fmt == "plain":
        return _format_plain(raw_output)
    elif fmt == "json":
        return _format_json(raw_output, metadata)
    elif fmt == "html":
        return _format_html(raw_output, metadata)
    else:
        return _format_markdown(raw_output, metadata)


def _format_markdown(text: str, metadata: Dict[str, Any] | None = None) -> str:
    """格式化为标准 Markdown。"""
    lines = text.strip().split("\n")
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 空行 = 段落分隔
            if result and result[-1] != "":
                result.append("")
        else:
            result.append(stripped)

    # 拼接并规范化：确保段落之间以 \n\n 分隔
    formatted = "\n".join(result)
    # 将 3 个及以上的连续换行压缩为 2 个
    while "\n\n\n" in formatted:
        formatted = formatted.replace("\n\n\n", "\n\n")

    # 如果提供了元数据，则添加页脚
    if metadata:
        formatted += "\n\n---\n"
        formatted += f"*生成时间: {metadata.get('timestamp', 'N/A')}*\n"
        if metadata.get('model'):
            formatted += f"*模型: {metadata['model']}*\n"
        if metadata.get('tokens_in'):
            formatted += f"*Token 消耗: {metadata.get('tokens_in', 0)} 输入 / {metadata.get('tokens_out', 0)} 输出*\n"

    return formatted


def _format_plain(text: str) -> str:
    """格式化为纯文本 —— 去除 Markdown 格式标记。"""
    # 移除 Markdown 标题
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 移除粗体/斜体标记
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    # 移除链接，保留文本
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 移除代码块
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    # 移除行内代码
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # 移除水平分割线
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # 清理多余的空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _format_json(text: str, metadata: Dict[str, Any] | None = None) -> str:
    """格式化为结构化 JSON。"""
    # 尝试从 Markdown 中提取章节
    sections = []
    current_section = {"title": "Content", "content": ""}

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_section["content"].strip():
                sections.append(current_section)
            current_section = {"title": line[3:].strip(), "content": ""}
        elif line.startswith("### "):
            if current_section["content"].strip():
                sections.append(current_section)
            current_section = {"title": line[4:].strip(), "content": ""}
        else:
            current_section["content"] += line + "\n"

    if current_section["content"].strip():
        sections.append(current_section)

    result = {
        "sections": sections,
        "metadata": metadata or {},
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


def _format_html(text: str, metadata: Dict[str, Any] | None = None) -> str:
    """格式化为基础 HTML。"""
    html = text

    # 转换标题
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # 转换粗体/斜体
    html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', html)

    # 转换列表
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # 用段落包裹
    paragraphs = html.split('\n\n')
    formatted_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<h') and not p.startswith('<li'):
            p = f'<p>{p}</p>'
        formatted_paragraphs.append(p)

    html = '\n'.join(formatted_paragraphs)

    # 如果提供了元数据，则添加页脚
    if metadata:
        html += '\n<footer>\n'
        html += f'<p><em>生成时间: {metadata.get("timestamp", "N/A")}</em></p>\n'
        if metadata.get('model'):
            html += f'<p><em>模型: {metadata["model"]}</em></p>\n'
        html += '</footer>'

    return html


def restore_redactions(text: str, redact_map: Dict[str, str]) -> str:
    """将脱敏占位符还原为原始值。

    Args:
        text: 含占位符的文本（如 LLM 响应）。
        redact_map: 占位符 → 原始值的映射。

    Returns:
        还原后的文本。
    """
    if not redact_map:
        return text
    for placeholder, original in redact_map.items():
        text = text.replace(placeholder, original)
    return text


def validate_save_path(path_str: str, config) -> bool:
    """验证保存路径是否安全。

    检查项：
    - 无路径穿越（../）
    - 路径解析后在允许的目录范围内

    参数：
        path_str: 提议的文件路径。
        config: 包含 allowed_paths 的 AppConfig。

    返回：
        如果路径安全则返回 True，否则返回 False。
    """
    try:
        path = Path(path_str).resolve()
    except (OSError, ValueError):
        return False

    path_str_resolved = str(path)

    for allowed in config.allowed_paths:
        allowed_resolved = str(Path(allowed).resolve())
        if path_str_resolved.startswith(allowed_resolved):
            return True

    return False
