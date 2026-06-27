"""Tests for upload policy decision and preview generation."""
import pytest
from src.config import AppConfig
from src.tools.file_ops import ParsedDocument
from src.tools.redaction import SensitiveMatch
from src.workflow.upload_policy import (
    UploadDecision,
    UploadPreview,
    decide_upload_strategy,
    generate_preview,
)


@pytest.fixture
def config():
    return AppConfig(
        model_name="deepseek-v4-flash",
        model_base_url="https://api.deepseek.com/v1",
        allowed_paths=["data/", "output/"],
        max_file_size_mb=5,
        max_tokens_per_request=50000,
        max_cost_per_request_yuan=0.5,
    )


def make_doc(sensitive_matches=None):
    """Helper to create a ParsedDocument for testing."""
    return ParsedDocument(
        path="data/test.txt",
        file_type="txt",
        title="Test Document",
        paragraphs=["段落一内容。", "段落二内容。"],
        raw_text="段落一内容。\n\n段落二内容。",
        chunks=["段落一内容。", "段落二内容。"],
        sensitive_matches=sensitive_matches or [],
    )


def make_phone_match():
    return SensitiveMatch(
        type="PHONE", value="13812345678", start=0, end=11, placeholder="PHONE_1"
    )


class TestDecideUploadStrategy:
    """Tests for upload strategy decision logic."""

    # --- Quick mode ---

    def test_quick_mode_no_sensitive(self, config):
        doc = make_doc()
        decision = decide_upload_strategy(doc, "quick", config)
        assert decision.strategy == "full"

    def test_quick_mode_with_sensitive(self, config):
        """Quick mode ignores sensitive info — still full send."""
        doc = make_doc([make_phone_match()])
        decision = decide_upload_strategy(doc, "quick", config)
        assert decision.strategy == "full"

    # --- Privacy enhanced mode ---

    def test_privacy_enhanced_no_sensitive(self, config):
        doc = make_doc()
        decision = decide_upload_strategy(doc, "privacy_enhanced", config)
        assert decision.strategy == "full"

    def test_privacy_enhanced_with_phone(self, config):
        doc = make_doc([make_phone_match()])
        decision = decide_upload_strategy(doc, "privacy_enhanced", config)
        assert decision.strategy == "redacted"

    def test_privacy_enhanced_with_api_key(self, config):
        match = SensitiveMatch(
            type="API_KEY", value="sk-abc123xyz4567890",
            start=0, end=22, placeholder="API_KEY_1"
        )
        doc = make_doc([match])
        decision = decide_upload_strategy(doc, "privacy_enhanced", config)
        assert decision.strategy == "redacted"

    # --- Manual confirmation mode ---

    def test_manual_confirm_no_sensitive(self, config):
        doc = make_doc()
        decision = decide_upload_strategy(doc, "manual_confirm", config)
        assert decision.strategy == "chunks"
        assert decision.needs_confirmation is True

    def test_manual_confirm_with_sensitive(self, config):
        doc = make_doc([make_phone_match()])
        decision = decide_upload_strategy(doc, "manual_confirm", config)
        assert decision.strategy == "chunks"
        assert decision.needs_confirmation is True

    # --- Local fallback mode ---

    def test_local_fallback(self, config):
        doc = make_doc()
        decision = decide_upload_strategy(doc, "local_fallback", config)
        assert decision.strategy == "blocked"

    def test_local_fallback_with_sensitive(self, config):
        doc = make_doc([make_phone_match()])
        decision = decide_upload_strategy(doc, "local_fallback", config)
        assert decision.strategy == "blocked"

    # --- Unknown mode ---

    def test_unknown_mode_defaults_to_manual_confirm(self, config):
        doc = make_doc()
        decision = decide_upload_strategy(doc, "unknown_mode", config)
        # Should fall back to safest default
        assert decision.strategy in ("chunks", "blocked")


class TestUploadDecision:
    """Tests for UploadDecision dataclass."""

    def test_default_fields(self):
        d = UploadDecision(strategy="full")
        assert d.strategy == "full"
        assert d.selected_chunks == []
        assert d.redact_map == {}
        assert d.needs_confirmation is False

    def test_with_redact_map(self):
        d = UploadDecision(
            strategy="redacted",
            redact_map={"PHONE_1": "13812345678"},
        )
        assert "PHONE_1" in d.redact_map


class TestGeneratePreview:
    """Tests for preview generation."""

    def test_preview_contains_summary(self, config):
        doc = make_doc()
        decision = UploadDecision(strategy="full")
        preview = generate_preview(doc, decision, config)
        assert "段落一" in preview.summary or len(preview.summary) > 0

    def test_preview_has_token_estimate(self, config):
        doc = make_doc()
        decision = UploadDecision(strategy="full")
        preview = generate_preview(doc, decision, config)
        assert preview.tokens_in_estimate > 0

    def test_preview_has_cost_estimate(self, config):
        doc = make_doc()
        decision = UploadDecision(strategy="full")
        preview = generate_preview(doc, decision, config)
        assert preview.cost_estimate >= 0.0

    def test_preview_has_model_name(self, config):
        doc = make_doc()
        decision = UploadDecision(strategy="full")
        preview = generate_preview(doc, decision, config)
        assert preview.model == "deepseek-v4-flash"

    def test_preview_with_sensitive_info(self, config):
        doc = make_doc([make_phone_match()])
        decision = UploadDecision(strategy="redacted")
        preview = generate_preview(doc, decision, config)
        assert preview.has_sensitive is True
        assert "PHONE" in preview.sensitive_types

    def test_preview_without_sensitive_info(self, config):
        doc = make_doc()
        decision = UploadDecision(strategy="full")
        preview = generate_preview(doc, decision, config)
        assert preview.has_sensitive is False
        assert preview.sensitive_types == []

    def test_preview_blocked_strategy(self, config):
        doc = make_doc()
        decision = UploadDecision(strategy="blocked")
        preview = generate_preview(doc, decision, config)
        assert preview.tokens_in_estimate == 0
        assert preview.cost_estimate == 0.0
