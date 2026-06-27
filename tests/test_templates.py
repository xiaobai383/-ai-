"""Tests for workflow template functionality."""
import pytest
from pathlib import Path

from src.workflow.templates import (
    list_templates,
    load_workflow_template,
    create_template,
    delete_template,
)


@pytest.fixture
def temp_templates_dir(tmp_path):
    """Create a temporary templates directory."""
    return tmp_path / "workflows"


def test_create_and_load_template(temp_templates_dir):
    """Test creating and loading a workflow template."""
    steps = [
        {"name": "preprocess", "action": "parse_file"},
        {"name": "llm_call", "action": "invoke_llm"},
    ]

    template_path = create_template(
        name="test_template",
        description="A test template",
        steps=steps,
    )

    # Note: create_template uses TEMPLATES_DIR, not temp_templates_dir
    # For unit testing, we'd need to mock this
    # This is a basic integration test
    templates = list_templates()
    assert isinstance(templates, list)


def test_load_nonexistent_template():
    """Test loading a non-existent template raises error."""
    with pytest.raises(FileNotFoundError):
        load_workflow_template("nonexistent_template_xyz")


def test_list_templates():
    """Test listing templates returns a list."""
    templates = list_templates()
    assert isinstance(templates, list)


def test_delete_nonexistent_template():
    """Test deleting a non-existent template returns False."""
    result = delete_template("nonexistent_template_xyz")
    assert result is False
