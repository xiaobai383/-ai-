"""带安全边界的文件操作——读取、写入、解析、分块。"""
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.tools.cost import estimate_tokens


@dataclass
class ParsedDocument:
    """已解析文件的结构化表示。"""

    path: str
    file_type: str
    title: str = ""
    paragraphs: List[str] = field(default_factory=list)
    page_count: int = 1
    raw_text: str = ""
    chunks: List[str] = field(default_factory=list)
    sensitive_matches: list = field(default_factory=list)


def _normalize_path(file_path: str) -> Path:
    """解析为绝对路径。"""
    return Path(file_path).resolve()


def _check_path_security(file_path: str, config) -> Path:
    """根据安全策略验证文件路径。

    成功时返回解析后的 Path，违规时抛出异常。
    """
    path = _normalize_path(file_path)
    path_str = str(path)

    # 检查黑名单模式
    filename = path.name
    for pattern in config.blocked_patterns:
        if fnmatch.fnmatch(filename, pattern):
            raise PermissionError(
                f"File '{filename}' matches blocked pattern '{pattern}'"
            )

    # 检查白名单——至少有一个允许的路径必须是前缀
    allowed = False
    for allowed_path in config.allowed_paths:
        allowed_resolved = str(Path(allowed_path).resolve())
        if path_str.startswith(allowed_resolved):
            allowed = True
            break

    if not allowed:
        raise PermissionError(
            f"Path '{path_str}' is not in allowed directories: {config.allowed_paths}"
        )

    return path


def _check_file_size(file_path: Path, config) -> None:
    """如果文件超过大小限制则抛出异常。"""
    max_bytes = config.max_file_size_mb * 1024 * 1024
    size = file_path.stat().st_size
    if size > max_bytes:
        raise PermissionError(
            f"File size {size} bytes exceeds limit of {max_bytes} bytes "
            f"({config.max_file_size_mb} MB)"
        )


def read_file(file_path: str, config) -> str:
    """带安全检查的文件读取。

    Args:
        file_path: 文件路径。
        config: 包含安全设置的 AppConfig。

    Returns:
        字符串形式的文件内容。

    Raises:
        PermissionError: 如果路径被阻止、不在允许目录内或文件过大。
        FileNotFoundError: 如果文件不存在。
    """
    path = _check_path_security(file_path, config)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    _check_file_size(path, config)

    return path.read_text(encoding="utf-8")


def save_file(
    file_path: str, content: str, config, overwrite: bool = False
) -> str:
    """带安全检查的文件写入。

    Args:
        file_path: 目标路径。
        content: 要写入的内容。
        config: 包含安全设置的 AppConfig。
        overwrite: 若为 False，文件已存在时抛出 FileExistsError。

    Returns:
        解析后的路径字符串。

    Raises:
        PermissionError: 如果路径不在允许目录内。
        FileExistsError: 如果文件已存在且 overwrite=False。
    """
    path = _check_path_security(file_path, config)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {file_path}. Use overwrite=True to replace."
        )

    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(content, encoding="utf-8")
    return str(path)


def parse_file(file_path: str, config) -> ParsedDocument:
    """将文件解析为结构化的 ParsedDocument。

    支持格式：.txt、.md

    Args:
        file_path: 文件路径。
        config: AppConfig。

    Returns:
        包含原始文本和段落的 ParsedDocument。

    Raises:
        ValueError: 如果文件类型不受支持。
        PermissionError: 如果安全检查失败。
        FileNotFoundError: 如果文件不存在。
    """
    path = _normalize_path(file_path)
    suffix = path.suffix.lower()

    if suffix not in (".txt", ".md"):
        raise ValueError(
            f"Unsupported file type: '{suffix}'. Supported: .txt, .md"
        )

    content = read_file(file_path, config)
    paragraphs = _extract_paragraphs(content)
    title = _extract_title(path, content, suffix)

    doc = ParsedDocument(
        path=str(path),
        file_type=suffix.lstrip("."),
        title=title,
        paragraphs=paragraphs,
        raw_text=content,
    )

    # 对内容进行分块
    doc.chunks = chunk_text(content, max_tokens=1000)

    return doc


def _extract_paragraphs(text: str) -> List[str]:
    """将文本拆分为非空段落。"""
    parts = text.split("\n\n")
    return [p.strip() for p in parts if p.strip()]


def _extract_title(path: Path, content: str, suffix: str) -> str:
    """从文件名或内容中提取标题。"""
    if suffix == ".md":
        # 尝试查找第一个 H1 标题
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                return line[2:].strip()
    # 回退方案：使用不带扩展名的文件名
    return path.stem


def chunk_text(text: str, max_tokens: int = 1000) -> List[str]:
    """将文本拆分为尊重段落边界的块。

    每个块应约为 max_tokens 或更小。
    尽可能将段落保持在一起。如果单个段落超过 max_tokens，则进行强制拆分。

    Args:
        text: 要分块的文本。
        max_tokens: 每个块的目标最大 token 数。

    Returns:
        文本块列表。
    """
    if not text.strip():
        return []

    paragraphs = _extract_paragraphs(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para, "deepseek-v4-flash")

        # 如果单个段落过大，则强制拆分
        if para_tokens > max_tokens:
            # 先刷新当前块
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            # 按字符块强制拆分过大的段落
            chars_per_token = max(1, len(para) // max(1, para_tokens))
            chars_per_chunk = chars_per_token * max_tokens
            for i in range(0, len(para), chars_per_chunk):
                sub = para[i : i + chars_per_chunk]
                chunks.append(sub)
            continue

        # 如果添加此段落会超出限制，则刷新当前块
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    # 刷新剩余内容
    if current:
        chunks.append("\n\n".join(current))

    return chunks
