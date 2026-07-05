"""Tests for workflow template functionality."""
import pytest
from pathlib import Path

from src.workflow.templates import (
    list_templates,
    load_workflow_template,
)


def test_load_nonexistent_template():
    """Test loading a non-existent template raises error."""
    with pytest.raises(FileNotFoundError):
        load_workflow_template("nonexistent_template_xyz")


def test_list_templates():
    """Test listing templates returns a list."""
    templates = list_templates()
    assert isinstance(templates, list)
