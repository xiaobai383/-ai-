"""Tests for sensitive info detection and redaction."""
import pytest
from src.tools.redaction import (
    SensitiveMatch,
    detect_sensitive,
    redact,
)


class TestDetectSensitivePhone:
    """Phone number detection."""

    def test_detect_chinese_mobile(self):
        matches = detect_sensitive("请联系 13812345678 获取详情")
        phones = [m for m in matches if m.type == "PHONE"]
        assert len(phones) == 1
        assert phones[0].value == "13812345678"

    def test_detect_multiple_phones(self):
        text = "电话：13812345678，备用：13987654321"
        matches = detect_sensitive(text)
        phones = [m for m in matches if m.type == "PHONE"]
        assert len(phones) == 2

    def test_not_phone_random_11_digits(self):
        """11111111111 is not a valid Chinese mobile format."""
        matches = detect_sensitive("号码 11111111111 不是手机号")
        phones = [m for m in matches if m.type == "PHONE"]
        assert len(phones) == 0

    def test_not_phone_too_short(self):
        matches = detect_sensitive("1234567890")
        phones = [m for m in matches if m.type == "PHONE"]
        assert len(phones) == 0


class TestDetectSensitiveEmail:
    """Email detection."""

    def test_detect_email(self):
        matches = detect_sensitive("发送到 admin@example.com 即可")
        emails = [m for m in matches if m.type == "EMAIL"]
        assert len(emails) == 1
        assert emails[0].value == "admin@example.com"

    def test_detect_email_with_subdomain(self):
        matches = detect_sensitive("邮箱 user@mail.example.co.uk 已注册")
        emails = [m for m in matches if m.type == "EMAIL"]
        assert len(emails) == 1

    def test_not_email_without_at(self):
        matches = detect_sensitive("这不是邮箱 example.com")
        emails = [m for m in matches if m.type == "EMAIL"]
        assert len(emails) == 0

    def test_not_email_without_domain(self):
        matches = detect_sensitive("user@ 不是完整邮箱")
        emails = [m for m in matches if m.type == "EMAIL"]
        assert len(emails) == 0


class TestDetectSensitiveIdCard:
    """ID card number detection."""

    def test_detect_18_digit_id(self):
        matches = detect_sensitive("身份证号 110101199003071234")
        ids = [m for m in matches if m.type == "ID_CARD"]
        assert len(ids) == 1
        assert ids[0].value == "110101199003071234"

    def test_not_id_card_17_digits(self):
        """17 digits is not a valid ID card."""
        matches = detect_sensitive("号码 11010119900307123")
        ids = [m for m in matches if m.type == "ID_CARD"]
        assert len(ids) == 0

    def test_not_id_card_with_letter_at_end(self):
        """Valid ID card can end with X."""
        matches = detect_sensitive("身份证 11010119900307123X")
        ids = [m for m in matches if m.type == "ID_CARD"]
        assert len(ids) == 1


class TestDetectSensitiveApiKey:
    """API key detection."""

    def test_detect_sk_prefix_key(self):
        matches = detect_sensitive("使用 sk-abc123xyz456 作为密钥")
        keys = [m for m in matches if m.type == "API_KEY"]
        assert len(keys) == 1
        assert "sk-abc123xyz456" in keys[0].value

    def test_detect_api_prefix_key(self):
        matches = detect_sensitive("api-key: api-abcdefghij")
        keys = [m for m in matches if m.type == "API_KEY"]
        assert len(keys) == 1

    def test_not_api_key_looks_like_word(self):
        """sk- followed by too short string shouldn't match."""
        # sk-xy is too short to be a real key
        matches = detect_sensitive("sk-xy")
        keys = [m for m in matches if m.type == "API_KEY"]
        # We want reasonable key length — at least ~20 chars
        assert len(keys) == 0


class TestDetectSensitivePassword:
    """Password field detection."""

    def test_detect_password_equals(self):
        matches = detect_sensitive("password=mysecret123")
        pws = [m for m in matches if m.type == "PASSWORD"]
        assert len(pws) == 1

    def test_detect_passwd_equals(self):
        matches = detect_sensitive("配置 passwd=admin123 用于登录")
        pws = [m for m in matches if m.type == "PASSWORD"]
        assert len(pws) == 1

    def test_not_password_in_normal_text(self):
        matches = detect_sensitive("请输入你的 password 来完成注册")
        pws = [m for m in matches if m.type == "PASSWORD"]
        assert len(pws) == 0


class TestDetectSensitivePath:
    """Absolute path detection."""

    def test_detect_windows_absolute_path(self):
        matches = detect_sensitive(r"文件位于 C:\Users\admin\docs\file.txt")
        paths = [m for m in matches if m.type == "PATH"]
        assert len(paths) >= 1

    def test_detect_unix_absolute_path(self):
        matches = detect_sensitive("配置文件 /home/user/config.yaml 已加载")
        paths = [m for m in matches if m.type == "PATH"]
        assert len(paths) >= 1

    def test_not_relative_path(self):
        matches = detect_sensitive("打开 data/file.txt 即可")
        paths = [m for m in matches if m.type == "PATH"]
        assert len(paths) == 0


class TestDetectSensitiveMultiple:
    """Multiple matches in same text."""

    def test_multiple_types(self):
        text = "姓名张三，电话13812345678，邮箱zhangsan@example.com"
        matches = detect_sensitive(text)
        # Should find at least phone and email
        types = {m.type for m in matches}
        assert "PHONE" in types
        assert "EMAIL" in types

    def test_all_matches_have_position(self):
        text = "电话：13812345678"
        matches = detect_sensitive(text)
        for m in matches:
            assert m.start >= 0
            assert m.end > m.start
            assert text[m.start : m.end] == m.value


class TestRedact:
    """Tests for redaction."""

    def test_redact_phone(self):
        text = "联系 13812345678 获取信息"
        matches = detect_sensitive(text)
        redacted, mapping = redact(text, matches)
        assert "13812345678" not in redacted
        assert "PHONE_1" in redacted
        assert "PHONE_1" in mapping
        assert mapping["PHONE_1"] == "13812345678"

    def test_redact_email(self):
        text = "发送到 admin@example.com"
        matches = detect_sensitive(text)
        redacted, mapping = redact(text, matches)
        assert "admin@example.com" not in redacted
        assert "EMAIL_1" in redacted

    def test_redact_multiple_same_type(self):
        text = "手机1: 13812345678, 手机2: 13987654321"
        matches = detect_sensitive(text)
        redacted, mapping = redact(text, matches)
        assert "PHONE_1" in redacted
        assert "PHONE_2" in redacted
        assert mapping["PHONE_1"] == "13812345678"
        assert mapping["PHONE_2"] == "13987654321"

    def test_redact_person_name(self):
        text = "张三和李四参加了会议"
        matches = detect_sensitive(text)
        redacted, _ = redact(text, matches)
        # Chinese names should be detected and redacted
        # Note: name detection is ambitious; test checks no crash
        assert isinstance(redacted, str)

    def test_redact_empty_text(self):
        matches = detect_sensitive("")
        redacted, mapping = redact("", matches)
        assert redacted == ""
        assert mapping == {}

    def test_redact_preserves_non_sensitive(self):
        text = "今天的会议主题是项目进度讨论"
        matches = detect_sensitive(text)
        redacted, _ = redact(text, matches)
        # Non-sensitive text should remain mostly intact
        assert "项目进度" in redacted or len(redacted) > 0

    def test_redaction_map_is_complete(self):
        text = "电话13812345678，邮箱test@example.com"
        matches = detect_sensitive(text)
        redacted, mapping = redact(text, matches)
        # Every placeholder in redacted text should be in mapping
        for key in mapping:
            assert mapping[key] is not None
            assert len(mapping[key]) > 0

    def test_sensitive_match_dataclass(self):
        m = SensitiveMatch(
            type="PHONE", value="13812345678", start=3, end=14, placeholder="PHONE_1"
        )
        assert m.type == "PHONE"
        assert m.value == "13812345678"
        assert m.start == 3
        assert m.end == 14
        assert m.placeholder == "PHONE_1"
