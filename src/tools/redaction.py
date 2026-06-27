"""Sensitive information detection and redaction using regex rules."""
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SensitiveMatch:
    """A detected piece of sensitive information."""

    type: str  # PHONE, EMAIL, ID_CARD, API_KEY, PASSWORD, PATH, PERSON
    value: str
    start: int
    end: int
    placeholder: str = ""


# Detection rules: (type, regex_pattern)
_DETECTION_RULES: List[Tuple[str, str]] = [
    # Chinese mobile phone: 1[3-9] + 9 digits = 11 digits total
    ("PHONE", r"1[3-9]\d{9}"),
    # Email address
    ("EMAIL", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # Chinese ID card: 17 digits + digit/X, OR exactly 18 digits
    ("ID_CARD", r"\d{17}[\dXx]"),
    # API keys with common prefixes (at least 20 chars after prefix)
    ("API_KEY", r"(?:sk|api|AKIA)[-_][a-zA-Z0-9]{8,}"),
    # Password in config-like contexts (key=value format)
    (
        "PASSWORD",
        r"(?:password|passwd|pwd|secret)\s*[=:]\s*(\S+)",
    ),
    # Absolute file paths
    ("PATH", r"(?:[A-Za-z]:\\[^\s,;]+|/(?:home|etc|usr|var|tmp|opt)/[^\s,;]+)"),
    # Chinese person names (2-3 Chinese characters, common surnames)
    (
        "PERSON",
        r"[王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩邓彭曹曾田萧潘袁蔡蒋余于杜叶程魏苏吕丁]"
        r"[\u4e00-\u9fff]{1,2}",
    ),
]


def detect_sensitive(text: str) -> List[SensitiveMatch]:
    """Detect sensitive information in text using regex rules.

    Args:
        text: The text to scan.

    Returns:
        List of SensitiveMatch objects sorted by start position.
    """
    if not text:
        return []

    matches: List[SensitiveMatch] = []

    for match_type, pattern in _DETECTION_RULES:
        for m in re.finditer(pattern, text):
            value = m.group(0)
            matches.append(
                SensitiveMatch(
                    type=match_type,
                    value=value,
                    start=m.start(),
                    end=m.end(),
                    placeholder="",
                )
            )

    # Sort by position, then deduplicate overlapping matches (keep first/longer)
    matches.sort(key=lambda x: (x.start, -x.end))
    deduped: List[SensitiveMatch] = []
    for m in matches:
        if deduped and m.start < deduped[-1].end:
            # Overlapping — keep the existing one if it covers this
            continue
        deduped.append(m)

    # Assign placeholders with sequential numbering per type
    counters: Dict[str, int] = {}
    for m in deduped:
        counters.setdefault(m.type, 0)
        counters[m.type] += 1
        m.placeholder = f"{m.type}_{counters[m.type]}"

    return deduped


def redact(
    text: str, matches: List[SensitiveMatch], strategy: str = "replace"
) -> Tuple[str, Dict[str, str]]:
    """Replace sensitive information with placeholders.

    Args:
        text: Original text.
        matches: List of detected sensitive matches (from detect_sensitive).
        strategy: Redaction strategy — currently only 'replace' is supported.

    Returns:
        Tuple of (redacted_text, redaction_map) where redaction_map is
        placeholder -> original_value.
    """
    if not matches:
        return text, {}

    mapping: Dict[str, str] = {}

    # Build redacted text by replacing from end to start (preserve indices)
    parts: List[str] = []
    last_end = 0
    for m in sorted(matches, key=lambda x: x.start):
        parts.append(text[last_end : m.start])
        parts.append(m.placeholder)
        mapping[m.placeholder] = m.value
        last_end = m.end
    parts.append(text[last_end:])

    redacted_text = "".join(parts)
    return redacted_text, mapping
