"""Tests for user preferences functionality."""
import pytest
from pathlib import Path

from src.preferences.manager import PreferenceManager, UserPreferences


@pytest.fixture
def temp_preferences_dir(tmp_path):
    """Create a temporary preferences directory."""
    return tmp_path / "preferences"


@pytest.fixture
def preference_manager(temp_preferences_dir):
    """Create a PreferenceManager with temporary directory."""
    return PreferenceManager(preferences_dir=temp_preferences_dir)


def test_default_preferences(preference_manager):
    """Test loading default preferences."""
    prefs = preference_manager.load()

    assert prefs.default_mode == "privacy_enhanced"
    assert prefs.default_output_format == "markdown"
    assert prefs.include_metadata is True
    assert prefs.show_token_count is True
    assert prefs.auto_preview is True


def test_save_and_load_preferences(preference_manager):
    """Test saving and loading preferences."""
    prefs = UserPreferences(
        default_mode="quick",
        default_output_format="json",
        show_token_count=False,
    )

    preference_manager.save(prefs)
    loaded = preference_manager.load()

    assert loaded.default_mode == "quick"
    assert loaded.default_output_format == "json"
    assert loaded.show_token_count is False


def test_update_preferences(preference_manager):
    """Test updating specific preference fields."""
    preference_manager.update(default_mode="manual_confirm")

    prefs = preference_manager.load()
    assert prefs.default_mode == "manual_confirm"


def test_add_recent_template(preference_manager):
    """Test adding recent templates."""
    preference_manager.add_recent_template("summarize")
    preference_manager.add_recent_template("extract_todos")

    prefs = preference_manager.load()
    assert "summarize" in prefs.recent_templates
    assert "extract_todos" in prefs.recent_templates


def test_toggle_favorite_template(preference_manager):
    """Test toggling favorite templates."""
    # Add to favorites
    result = preference_manager.toggle_favorite_template("summarize")
    assert result is True

    prefs = preference_manager.load()
    assert "summarize" in prefs.favorite_templates

    # Remove from favorites
    result = preference_manager.toggle_favorite_template("summarize")
    assert result is False

    prefs = preference_manager.load()
    assert "summarize" not in prefs.favorite_templates


def test_reset_preferences(preference_manager):
    """Test resetting preferences to defaults."""
    # Modify some settings
    preference_manager.update(default_mode="quick", show_token_count=False)

    # Reset
    prefs = preference_manager.reset()

    assert prefs.default_mode == "privacy_enhanced"
    assert prefs.show_token_count is True


def test_preferences_to_dict(preference_manager):
    """Test converting preferences to dictionary."""
    prefs = preference_manager.load()
    prefs_dict = prefs.to_dict()

    assert isinstance(prefs_dict, dict)
    assert "default_mode" in prefs_dict
    assert "default_output_format" in prefs_dict


def test_preferences_from_dict():
    """Test creating preferences from dictionary."""
    data = {
        "default_mode": "quick",
        "default_output_format": "html",
        "unknown_key": "should_be_ignored",
    }

    prefs = UserPreferences.from_dict(data)

    assert prefs.default_mode == "quick"
    assert prefs.default_output_format == "html"
    assert not hasattr(prefs, "unknown_key")
