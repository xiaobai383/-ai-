"""使用正则表达式规则进行敏感信息检测与脱敏。"""
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SensitiveMatch:
    """检测到的一条敏感信息。"""

    type: str  # PHONE, EMAIL, ID_CARD, API_KEY, PASSWORD, PATH, PERSON
    value: str
    start: int
    end: int
    placeholder: str = ""


# 检测规则：(类型, 正则表达式模式)
_DETECTION_RULES: List[Tuple[str, str]] = [
    # 中国手机号：1[3-9] + 9 位数字 = 共 11 位
    ("PHONE", r"1[3-9]\d{9}"),
    # 电子邮箱地址
    ("EMAIL", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    # 中国身份证号：17 位数字 + 数字/X，或恰好 18 位数字
    ("ID_CARD", r"\d{17}[\dXx]"),
    # 常见前缀的 API 密钥（前缀后至少 8 个字符）
    ("API_KEY", r"(?:sk|api|AKIA)[-_][a-zA-Z0-9]{8,}"),
    # 类配置文件中的密码（key=value 格式）
    (
        "PASSWORD",
        r"(?:password|passwd|pwd|secret)\s*[=:]\s*(\S+)",
    ),
    # 绝对文件路径
    ("PATH", r"(?:[A-Za-z]:\\[^\s,;]+|/(?:home|etc|usr|var|tmp|opt)/[^\s,;]+)"),
    # 中国姓名（2-3 个中文字符，常见姓氏）
    (
        "PERSON",
        r"[王李张刘陈杨黄赵周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩邓彭曹曾田萧潘袁蔡蒋余于杜叶程魏苏吕丁]"
        r"[\u4e00-\u9fff]{1,2}",
    ),
]


def detect_sensitive(text: str, rules_config: Dict[str, bool] | None = None) -> List[SensitiveMatch]:
    """使用正则表达式规则检测文本中的敏感信息。

    Args:
        text: 要扫描的文本。
        rules_config: 可选的规则开关字典，如 {"PHONE": True, "EMAIL": False}。
                      未提供或为 None 时启用所有规则。

    Returns:
        按起始位置排序的 SensitiveMatch 对象列表。
    """
    if not text:
        return []

    rules = _DETECTION_RULES
    if rules_config:
        rules = [
            (t, p) for t, p in _DETECTION_RULES
            if rules_config.get(t, True)
        ]

    matches: List[SensitiveMatch] = []

    for match_type, pattern in rules:
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

    # 按位置排序，然后对重叠匹配去重（保留第一个/较长的）
    matches.sort(key=lambda x: (x.start, -x.end))
    deduped: List[SensitiveMatch] = []
    for m in matches:
        if deduped and m.start < deduped[-1].end:
            # 重叠——如果已有匹配覆盖了此匹配，则保留已有的
            continue
        deduped.append(m)

    # 按类型分配带有序号编号的占位符
    counters: Dict[str, int] = {}
    for m in deduped:
        counters.setdefault(m.type, 0)
        counters[m.type] += 1
        m.placeholder = f"{m.type}_{counters[m.type]}"

    return deduped


def redact(
    text: str, matches: List[SensitiveMatch], strategy: str = "replace"
) -> Tuple[str, Dict[str, str]]:
    """用占位符替换敏感信息。

    Args:
        text: 原始文本。
        matches: 检测到的敏感信息匹配列表（来自 detect_sensitive）。
        strategy: 脱敏策略——目前仅支持 'replace'。

    Returns:
        (脱敏后文本, 脱敏映射表) 元组，其中脱敏映射表为 占位符 -> 原始值。
    """
    if not matches:
        return text, {}

    mapping: Dict[str, str] = {}

    # 从后往前替换来构建脱敏文本（保持索引正确）
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
