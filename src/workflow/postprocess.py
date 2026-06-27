"""Post-processing — format LLM output and validate save paths."""
import json
import re
from pathlib import Path
from typing import Any, Dict


def format_output(raw_output: str, fmt: str = "markdown", metadata: Dict[str, Any] | None = None) -> str:
    """Clean and format raw LLM output.

    Supports multiple output formats:
    - markdown: Standard Markdown formatting
    - plain: Plain text without formatting
    - json: Structured JSON output
    - html: Basic HTML formatting

    Args:
        raw_output: Raw text from LLM.
        fmt: Output format — 'markdown', 'plain', 'json', 'html'.
        metadata: Optional metadata to include in output.

    Returns:
        Formatted text.
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
    """Format as standard Markdown."""
    lines = text.strip().split("\n")
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Empty line = paragraph break
            if result and result[-1] != "":
                result.append("")
        else:
            result.append(stripped)

    # Join and normalize: ensure paragraphs separated by \n\n
    formatted = "\n".join(result)
    # Collapse 3+ newlines into 2
    while "\n\n\n" in formatted:
        formatted = formatted.replace("\n\n\n", "\n\n")

    # Add metadata footer if provided
    if metadata:
        formatted += "\n\n---\n"
        formatted += f"*生成时间: {metadata.get('timestamp', 'N/A')}*\n"
        if metadata.get('model'):
            formatted += f"*模型: {metadata['model']}*\n"
        if metadata.get('tokens_in'):
            formatted += f"*Token 消耗: {metadata.get('tokens_in', 0)} 输入 / {metadata.get('tokens_out', 0)} 输出*\n"

    return formatted


def _format_plain(text: str) -> str:
    """Format as plain text — strip Markdown formatting."""
    # Remove Markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    # Remove links, keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _format_json(text: str, metadata: Dict[str, Any] | None = None) -> str:
    """Format as structured JSON."""
    # Try to extract sections from Markdown
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
    """Format as basic HTML."""
    html = text

    # Convert headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Convert bold/italic
    html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', html)

    # Convert lists
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Wrap in paragraphs
    paragraphs = html.split('\n\n')
    formatted_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<h') and not p.startswith('<li'):
            p = f'<p>{p}</p>'
        formatted_paragraphs.append(p)

    html = '\n'.join(formatted_paragraphs)

    # Add metadata if provided
    if metadata:
        html += '\n<footer>\n'
        html += f'<p><em>生成时间: {metadata.get("timestamp", "N/A")}</em></p>\n'
        if metadata.get('model'):
            html += f'<p><em>模型: {metadata["model"]}</em></p>\n'
        html += '</footer>'

    return html


def validate_save_path(path_str: str, config) -> bool:
    """Validate that a save path is safe.

    Checks:
    - No path traversal (../)
    - Path resolves within allowed directories

    Args:
        path_str: Proposed file path.
        config: AppConfig with allowed_paths.

    Returns:
        True if path is safe, False otherwise.
    """
    try:
        path = Path(path_str).resolve()
    except (OSError, ValueError):
        return False

    path_str_resolved = str(path)

    # Check path traversal attempt
    if ".." in Path(path_str).parts:
        # The original path contains .. — check if resolved path is still safe
        pass

    for allowed in config.allowed_paths:
        allowed_resolved = str(Path(allowed).resolve())
        if path_str_resolved.startswith(allowed_resolved):
            return True

    return False
