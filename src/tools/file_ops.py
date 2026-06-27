"""File operations with security boundaries — read, write, parse, chunk."""
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.tools.cost import estimate_tokens


@dataclass
class ParsedDocument:
    """Structured representation of a parsed file."""

    path: str
    file_type: str
    title: str = ""
    paragraphs: List[str] = field(default_factory=list)
    page_count: int = 1
    raw_text: str = ""
    chunks: List[str] = field(default_factory=list)
    sensitive_matches: list = field(default_factory=list)


def _normalize_path(file_path: str) -> Path:
    """Resolve to absolute path."""
    return Path(file_path).resolve()


def _check_path_security(file_path: str, config) -> Path:
    """Validate a file path against security policies.

    Returns the resolved Path on success, raises on violation.
    """
    path = _normalize_path(file_path)
    path_str = str(path)

    # Check blacklist patterns
    filename = path.name
    for pattern in config.blocked_patterns:
        if fnmatch.fnmatch(filename, pattern):
            raise PermissionError(
                f"File '{filename}' matches blocked pattern '{pattern}'"
            )

    # Check whitelist — at least one allowed path must be a prefix
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
    """Raise if file exceeds size limit."""
    max_bytes = config.max_file_size_mb * 1024 * 1024
    size = file_path.stat().st_size
    if size > max_bytes:
        raise PermissionError(
            f"File size {size} bytes exceeds limit of {max_bytes} bytes "
            f"({config.max_file_size_mb} MB)"
        )


def read_file(file_path: str, config) -> str:
    """Read a file with security checks.

    Args:
        file_path: Path to the file.
        config: AppConfig with security settings.

    Returns:
        File content as string.

    Raises:
        PermissionError: If path is blocked or outside allowed dirs, or file too large.
        FileNotFoundError: If file does not exist.
    """
    path = _check_path_security(file_path, config)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    _check_file_size(path, config)

    return path.read_text(encoding="utf-8")


def save_file(
    file_path: str, content: str, config, overwrite: bool = False
) -> str:
    """Write content to a file with security checks.

    Args:
        file_path: Target path.
        content: Content to write.
        config: AppConfig with security settings.
        overwrite: If False, raise FileExistsError when file already exists.

    Returns:
        The resolved path string.

    Raises:
        PermissionError: If path is outside allowed dirs.
        FileExistsError: If file exists and overwrite=False.
    """
    path = _check_path_security(file_path, config)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {file_path}. Use overwrite=True to replace."
        )

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(content, encoding="utf-8")
    return str(path)


def parse_file(file_path: str, config) -> ParsedDocument:
    """Parse a file into a structured ParsedDocument.

    Supports: .txt, .md

    Args:
        file_path: Path to the file.
        config: AppConfig.

    Returns:
        ParsedDocument with raw text and paragraphs.

    Raises:
        ValueError: If file type is unsupported.
        PermissionError: If security check fails.
        FileNotFoundError: If file does not exist.
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

    # Chunk the content
    doc.chunks = chunk_text(content, max_tokens=1000)

    return doc


def _extract_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraphs."""
    parts = text.split("\n\n")
    return [p.strip() for p in parts if p.strip()]


def _extract_title(path: Path, content: str, suffix: str) -> str:
    """Extract title from filename or content."""
    if suffix == ".md":
        # Try to find first H1 heading
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                return line[2:].strip()
    # Fallback: use filename without extension
    return path.stem


def chunk_text(text: str, max_tokens: int = 1000) -> List[str]:
    """Split text into chunks respecting paragraph boundaries.

    Each chunk should be approximately max_tokens or smaller.
    Paragraphs are kept together when possible. If a single paragraph
    exceeds max_tokens, it is hard-split.

    Args:
        text: The text to chunk.
        max_tokens: Target maximum tokens per chunk.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    paragraphs = _extract_paragraphs(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para, "deepseek-v4-flash")

        # If a single paragraph is too large, hard-split it
        if para_tokens > max_tokens:
            # Flush current chunk first
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            # Hard-split the huge paragraph by character chunks
            chars_per_token = max(1, len(para) // max(1, para_tokens))
            chars_per_chunk = chars_per_token * max_tokens
            for i in range(0, len(para), chars_per_chunk):
                sub = para[i : i + chars_per_chunk]
                chunks.append(sub)
            continue

        # If adding this paragraph would exceed limit, flush current chunk
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    # Flush remaining
    if current:
        chunks.append("\n\n".join(current))

    return chunks
