"""预处理编排 —— 文件解析、分块、敏感信息检测。"""
from typing import List

from src.tools.file_ops import ParsedDocument, parse_file


def preprocess(file_paths: List[str], config) -> List[ParsedDocument]:
    """解析并预处理文件列表。

    对每个文件执行以下步骤：
    1. 解析为 ParsedDocument（txt/md）
    2. 对内容进行分块
    3. 如果启用了脱敏，检测敏感信息（阶段 4）

    参数：
        file_paths: 待处理的文件路径列表。
        config: AppConfig 实例。

    返回：
        ParsedDocument 对象列表。

    异常：
        FileNotFoundError: 如果任何文件不存在。
        PermissionError: 如果任何文件未通过安全检查。
    """
    documents: List[ParsedDocument] = []

    for path in file_paths:
        doc = parse_file(path, config)

        if config.redaction_enabled:
            from src.tools.redaction import detect_sensitive
            doc.sensitive_matches = detect_sensitive(doc.raw_text, config.redaction_rules or None)

        documents.append(doc)

    return documents
