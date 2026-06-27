"""Preprocessing orchestration — file parsing, chunking, sensitive detection."""
from typing import List

from src.tools.file_ops import ParsedDocument, parse_file


def preprocess(file_paths: List[str], config) -> List[ParsedDocument]:
    """Parse and preprocess a list of files.

    For each file:
    1. Parse into ParsedDocument (txt/md)
    2. Chunk the content
    3. If redaction enabled, detect sensitive info (Phase 4)

    Args:
        file_paths: List of file paths to process.
        config: AppConfig instance.

    Returns:
        List of ParsedDocument objects.

    Raises:
        FileNotFoundError: If any file does not exist.
        PermissionError: If any file fails security checks.
    """
    documents: List[ParsedDocument] = []

    for path in file_paths:
        doc = parse_file(path, config)

        if config.redaction_enabled:
            from src.tools.redaction import detect_sensitive
            doc.sensitive_matches = detect_sensitive(doc.raw_text)

        documents.append(doc)

    return documents
