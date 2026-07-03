"""端到端测试 —— 验证 run_task() 完整管线的接线完整性。

测试范围：
1. 脱敏映射表是否真正注入到 LLM 请求中（known-code-issues #1）
2. RunLog 字段完整性（known-code-issues #5）
3. 四种上传模式全覆盖
4. 配置字段是否被实际消费（known-code-issues #4）
5. output 文件是否正确生成

设计原则：
- 每个断言旁边标注对应的 known-code-issues 编号
- 失败的断言 = wiring bug，不是测试写错了
- FakeLLM 记录实际收到的 message，用于事后取证
"""

import os
import tempfile
from pathlib import Path

import pytest
from src.config import AppConfig
from src.agent.app import run_task
from src.observability.run_log import RunLog


# ────────────────────────────────────────────────────────────
# 扩展版 FakeLLM — 记录实际收到的 messages
# ────────────────────────────────────────────────────────────

class RecordingFakeLLM:
    """Fake LLM 扩展版：记录收到的 messages，用于事后验证脱敏是否生效。"""

    def __init__(self, response: str = "测试回复"):
        self._response = response
        self._last_invoke_messages = None   # 记录最后一次 invoke 的完整参数
        self._last_call_messages = None     # 记录最后一次 __call__ 的完整参数

    def invoke(self, messages, **kwargs):
        self._last_invoke_messages = messages
        from langchain_core.messages import AIMessage
        return AIMessage(content=self._response)

    def bind_tools(self, tools):
        return self

    def __call__(self, *args, **kwargs):
        self._last_call_messages = args[0] if args else kwargs
        return self

    @property
    def received_content(self) -> str:
        """提取最后一次收到的**用户消息**文本内容（非 system prompt）。"""
        messages = self._last_invoke_messages or self._last_call_messages
        if messages is None:
            return ""
        import langchain_core.messages as lc
        # 优先找 HumanMessage
        for m in messages:
            if isinstance(m, lc.HumanMessage):
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    return content
        # fallback: 找 role='user' 的 dict
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content", "")
        # last resort: 任何有内容的字符串
        for m in messages:
            content = getattr(m, "content", "")
            if isinstance(content, str) and len(content) > 0:
                return content
        return ""


# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────

def _create_temp_file(content: str, suffix: str = ".txt") -> str:
    """创建包含指定内容的临时文件（在 temp 目录下），返回路径。"""
    tmpdir = Path(tempfile.gettempdir()) / "e2e_test"
    tmpdir.mkdir(parents=True, exist_ok=True)
    fp = tmpdir / f"test_sensitive{suffix}"
    fp.write_text(content, encoding="utf-8")
    return str(fp)


SENSITIVE_CONTENT = (
    "联系人：张三\n"
    "手机号：13812345678\n"
    "邮箱：zhangsan@example.com\n"
    "身份证：110101199003071234\n"
    "公司地址：北京市海淀区中关村大街1号\n"
)

CLEAN_CONTENT = (
    "今天的会议主要讨论了以下几个议题：\n\n"
    "1. 项目进度更新\n"
    "2. 下一阶段工作计划\n"
    "3. 预算调整方案\n\n"
    "与会人员一致同意上述方案。\n"
)


@pytest.fixture
def tmp_sensitive_file():
    """创建含敏感信息的临时 txt 文件。"""
    fp = _create_temp_file(SENSITIVE_CONTENT)
    yield fp
    try:
        os.remove(fp)
    except OSError:
        pass


@pytest.fixture
def tmp_clean_file():
    """创建不含敏感信息的临时 txt 文件。"""
    fp = _create_temp_file(CLEAN_CONTENT)
    yield fp
    try:
        os.remove(fp)
    except OSError:
        pass


@pytest.fixture
def privacy_config():
    """脱敏开启的 AppConfig。"""
    return AppConfig(
        model_name="deepseek-v4-flash",
        model_base_url="https://api.deepseek.com/v1",
        api_key="sk-fake-for-testing",
        allowed_paths=[
            "data/",
            "output/",
            str(Path(tempfile.gettempdir())),
        ],
        max_file_size_mb=5,
        max_tokens_per_request=50000,
        max_cost_per_request_yuan=0.5,
        redaction_enabled=True,
    )


@pytest.fixture
def quick_config():
    """脱敏关闭的 AppConfig（quick 模式用）。"""
    return AppConfig(
        model_name="deepseek-v4-flash",
        model_base_url="https://api.deepseek.com/v1",
        api_key="sk-fake-for-testing",
        allowed_paths=[
            "data/",
            "output/",
            str(Path(tempfile.gettempdir())),
        ],
        max_file_size_mb=5,
        max_tokens_per_request=50000,
        max_cost_per_request_yuan=0.5,
        redaction_enabled=False,
    )


# ════════════════════════════════════════════════════════════
# 阶段 2：脱敏是否真正生效（known-code-issues #1）
# ════════════════════════════════════════════════════════════

class TestRedactionWiring:
    """验证脱敏在完整管线中是否真正生效。

    脱敏目前在 run_task() 内完整工作：
    1. decide_upload_strategy → 返回 redacted text 到 selected_chunks
    2. run_task 用 text_to_send = selected_chunks（脱敏文本）代替 raw text
    3. LLM 收到的是脱敏后的占位符文本
    4. restore_redactions() 在 LLM 返回后将占位符还原为原始值
    """

    def test_privacy_enhanced_strategy_is_redacted(self, tmp_sensitive_file, privacy_config):
        """有敏感文件时，策略应为 'redacted'。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结文档",
            files=[tmp_sensitive_file],
            mode="privacy_enhanced",
            config=privacy_config,
            llm=llm,
            auto_confirm=True,
        )
        upload_step = _find_step(result, "upload_policy")
        assert upload_step is not None, "应存在 upload_policy 步骤"
        assert "redacted" in upload_step.output_preview, (
            f"策略应为 redacted，实际 output_preview={upload_step.output_preview}"
        )

    def test_redacted_text_not_contain_phone(self, tmp_sensitive_file, privacy_config):
        """脱敏后发往 LLM 的文本中不应含原始手机号。"""
        llm = RecordingFakeLLM(response="完成")
        run_task(
            query="总结文档",
            files=[tmp_sensitive_file],
            mode="privacy_enhanced",
            config=privacy_config,
            llm=llm,
            auto_confirm=True,
        )
        sent_text = llm.received_content
        assert sent_text, "LLM 应该收到消息但 received_content 为空"
        assert "13812345678" not in sent_text, (
            f"known-code-issues #1: 脱敏未生效！LLM 收到的文本中仍含原始手机号。\n"
            f"收到的内容前 200 字符: {sent_text[:200]}"
        )
        assert "zhangsan@example.com" not in sent_text, (
            f"known-code-issues #1: 脱敏未生效！LLM 收到的文本中仍含原始邮箱。\n"
            f"收到的内容前 200 字符: {sent_text[:200]}"
        )

    def test_redacted_text_contains_placeholders(self, tmp_sensitive_file, privacy_config):
        """脱敏后发往 LLM 的文本中应含 PHONE_1、EMAIL_1 等占位符。"""
        llm = RecordingFakeLLM(response="完成")
        run_task(
            query="总结文档",
            files=[tmp_sensitive_file],
            mode="privacy_enhanced",
            config=privacy_config,
            llm=llm,
            auto_confirm=True,
        )
        sent_text = llm.received_content
        assert "PHONE_1" in sent_text or "PHONE_" in sent_text, (
            f"脱敏后文本应包含 PHONE_ 占位符，实际内容前 200 字符: {sent_text[:200]}"
        )
        assert "EMAIL_1" in sent_text or "EMAIL_" in sent_text, (
            f"脱敏后文本应包含 EMAIL_ 占位符，实际内容前 200 字符: {sent_text[:200]}"
        )

    def test_sensitive_matches_detected(self, tmp_sensitive_file, privacy_config):
        """敏感文件应该被检测到敏感信息。"""
        from src.workflow.preprocess import preprocess
        docs = preprocess([tmp_sensitive_file], privacy_config)
        assert len(docs[0].sensitive_matches) > 0, (
            "redaction_enabled=True 但 sensitive_matches 为空——"
            "预处理阶段敏感检测未生效"
        )


# ════════════════════════════════════════════════════════════
# 阶段 3：RunLog 完整性验证
# ════════════════════════════════════════════════════════════

class TestRunLogCompleteness:
    """验证 RunLog 每个字段、步骤都有值。

    known-code-issues #5: RunLog.fallback 设为 True 的死代码。
    """

    def test_runlog_total_tokens_positive(self, tmp_clean_file, quick_config):
        """输入 token 应为正数。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        assert result.total_tokens_in > 0, "输入 token 应为正数"

    def test_runlog_total_tokens_out_positive(self, tmp_clean_file, quick_config):
        """输出 token 应为正数（FakeLLM 返回了 "完成"）。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        assert result.total_tokens_out > 0, (
            f"输出 token 应为正数，实际={result.total_tokens_out}"
        )

    def test_runlog_five_steps_present(self, tmp_clean_file, quick_config):
        """应有 5 个核心步骤。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        step_names = {s.name for s in result.steps}
        expected = {"preprocess", "upload_policy", "llm_call", "postprocess", "save_result"}
        missing = expected - step_names
        assert not missing, f"缺少步骤: {missing}"

    def test_runlog_all_steps_success(self, tmp_clean_file, quick_config):
        """所有步骤状态应为 success。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        for step in result.steps:
            assert step.status == "success", (
                f"步骤 {step.name} 状态={step.status}，应为 success\n"
                f"output_preview={step.output_preview}"
            )

    def test_runlog_user_query_preserved(self, tmp_clean_file, quick_config):
        """user_query 应被正确记录。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="请分析这份文件的性能瓶颈",
            files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        assert result.user_query == "请分析这份文件的性能瓶颈"

    def test_runlog_result_path_not_none(self, tmp_clean_file, quick_config):
        """result_path 应被设置。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        assert result.result_path is not None, "result_path 不应为 None"
        assert len(result.result_path) > 0, "result_path 不应为空字符串"


# ════════════════════════════════════════════════════════════
# 阶段 4：四模式全覆盖
# ════════════════════════════════════════════════════════════

class TestFourModes:
    """覆盖四种上传模式的完整链路。"""

    def test_quick_mode_sends_full_text(self, tmp_sensitive_file, quick_config):
        """quick 模式：LLM 收到原始全文（含手机号）。"""
        llm = RecordingFakeLLM(response="完成")
        run_task(
            query="总结", files=[tmp_sensitive_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        sent = llm.received_content
        assert "13812345678" in sent, (
            f"quick 模式应发送全文（含敏感信息），但未在消息中找到手机号。\n"
            f"前 200 字符: {sent[:200]}"
        )

    def test_privacy_enhanced_no_sensitive_is_full(self, tmp_clean_file, privacy_config):
        """privacy_enhanced + 无敏感文件 → strategy 应为 full。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="privacy_enhanced",
            config=privacy_config, llm=llm, auto_confirm=True,
        )
        upload_step = _find_step(result, "upload_policy")
        assert "full" in upload_step.output_preview, (
            f"无敏感文件时策略应为 full，实际={upload_step.output_preview}"
        )

    def test_privacy_enhanced_with_sensitive_is_redacted(self, tmp_sensitive_file, privacy_config):
        """privacy_enhanced + 有敏感文件 → strategy 应为 redacted。"""
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_sensitive_file], mode="privacy_enhanced",
            config=privacy_config, llm=llm, auto_confirm=True,
        )
        upload_step = _find_step(result, "upload_policy")
        assert "redacted" in upload_step.output_preview, (
            f"有敏感文件时策略应为 redacted，实际={upload_step.output_preview}"
        )

    def test_local_fallback_no_llm_call(self, tmp_clean_file, quick_config):
        """local_fallback 模式：不应有 llm_call 步骤，应有 local_analysis。"""
        llm = RecordingFakeLLM(response="不应被调用")
        result = run_task(
            query="本地分析", files=[tmp_clean_file], mode="local_fallback",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        step_names = {s.name for s in result.steps}
        assert "llm_call" not in step_names, (
            f"local_fallback 不应有 llm_call，实际步骤: {step_names}"
        )
        assert "local_analysis" in step_names, (
            f"local_fallback 应有 local_analysis 步骤，实际步骤: {step_names}"
        )

    def test_local_fallback_result_contains_local(self, tmp_clean_file, quick_config):
        """local_fallback 返回内容应含"本地分析结果"。"""
        llm = RecordingFakeLLM(response="不应被调用")
        result = run_task(
            query="分析", files=[tmp_clean_file], mode="local_fallback",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        output_path = Path(result.result_path)
        if output_path.exists():
            content = output_path.read_text(encoding="utf-8")
            assert "本地分析结果" in content, (
                f"本地兜底输出应含'本地分析结果'，实际内容前 200 字符: {content[:200]}"
            )


# ════════════════════════════════════════════════════════════
# 阶段 5：配置消费验证
# ════════════════════════════════════════════════════════════

class TestConfigConsumption:
    """验证 config.yaml 中的字段是否被 run_task() 实际消费。

    known-code-issues #4: 多个配置字段被加载但从未被 run_task 读取。
    """

    def test_redaction_disabled_no_detection(self, tmp_sensitive_file, quick_config):
        """redaction_enabled=False 时不应检测敏感信息。"""
        from src.workflow.preprocess import preprocess
        docs = preprocess([tmp_sensitive_file], quick_config)
        assert docs[0].sensitive_matches == [], (
            "redaction_enabled=False 但 sensitive_matches 非空——"
            "预处理阶段忽略了 redaction_enabled 配置"
        )

    def test_redaction_enabled_detects_sensitive(self, tmp_sensitive_file, privacy_config):
        """redaction_enabled=True 时应检测到敏感信息。"""
        from src.workflow.preprocess import preprocess
        docs = preprocess([tmp_sensitive_file], privacy_config)
        phone_matches = [m for m in docs[0].sensitive_matches if m.type == "PHONE"]
        assert len(phone_matches) >= 1, (
            "redaction_enabled=True 但未检测到手机号——"
            "预处理阶段敏感检测可能有问题"
        )

    def test_max_cost_limit_blocks(self, tmp_clean_file, quick_config):
        """极低费用上限应触发熔断。"""
        tight_config = AppConfig(
            model_name="deepseek-v4-flash",
            model_base_url="https://api.deepseek.com/v1",
            api_key="sk-fake-for-testing",
            allowed_paths=[
                "data/", "output/",
                str(Path(tempfile.gettempdir())),
            ],
            max_file_size_mb=5,
            max_tokens_per_request=50000,
            max_cost_per_request_yuan=0.0000001,  # 几乎为 0，必触发
            redaction_enabled=False,
        )
        llm = RecordingFakeLLM(response="完成")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=tight_config, llm=llm, auto_confirm=True,
        )
        limit_step = _find_step(result, "limit_check")
        assert limit_step is not None, (
            "费用超限时应有 limit_check 步骤"
        )
        assert limit_step.status == "failed", (
            f"limit_check 状态应为 failed，实际={limit_step.status}"
        )


# ════════════════════════════════════════════════════════════
# 阶段 6：output 文件验证
# ════════════════════════════════════════════════════════════

class TestOutputFile:
    """验证 output/*.md 文件正确生成。"""

    def test_output_file_exists(self, tmp_clean_file, quick_config):
        """执行后应生成 output 文件。"""
        llm = RecordingFakeLLM(response="测试回复内容")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        output_path = Path(result.result_path)
        assert output_path.exists(), f"output 文件不存在: {output_path}"

    def test_output_file_content_matches_llm_response(self, tmp_clean_file, quick_config):
        """output 文件内容应为 LLM 返回的内容。"""
        expected = "这是一段测试回复，用于验证输出文件内容。"
        llm = RecordingFakeLLM(response=expected)
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        output_path = Path(result.result_path)
        content = output_path.read_text(encoding="utf-8")
        assert expected in content, (
            f"output 文件应包含 LLM 回复，期望={expected!r}，\n实际={content!r}"
        )

    def test_output_file_not_just_error(self, tmp_clean_file, quick_config):
        """output 文件内容不应是 'Error' 字符串。

        known-code-issues #2: _call_llm 静默吞异常后返回 "Error"，
        导致下游以为成功但文件内容错误。
        """
        llm = RecordingFakeLLM(response="正常回复")
        result = run_task(
            query="总结", files=[tmp_clean_file], mode="quick",
            config=quick_config, llm=llm, auto_confirm=True,
        )
        output_path = Path(result.result_path)
        content = output_path.read_text(encoding="utf-8")
        assert content.strip() != "Error", (
            "known-code-issues #2: output 文件内容为 'Error'，"
            "说明 _call_llm 静默吞掉了异常"
        )


# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────

def _find_step(run_log: RunLog, name: str):
    """在 RunLog 的 steps 列表中按名称查找步骤。"""
    for s in run_log.steps:
        if s.name == name:
            return s
    return None
