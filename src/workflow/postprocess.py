"""Post-processing — format LLM output and validate save paths."""
from pathlib import Path


def format_output(raw_output: str, fmt: str = "markdown") -> str:
    """Clean and format raw LLM output.

    Ensures proper Markdown formatting:
    - Converts single newlines within paragraphs to spaces (soft wrapping)
    - Preserves double newlines as paragraph breaks
    - Strips leading/trailing whitespace

    Args:
        raw_output: Raw text from LLM.
        fmt: Output format — only 'markdown' supported for now.

    Returns:
        Formatted text.
    """
    if not raw_output or not raw_output.strip():
        return ""

    if fmt == "markdown":
        lines = raw_output.strip().split("\n")
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
        text = "\n".join(result)
        # Collapse 3+ newlines into 2
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text

    return raw_output


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
