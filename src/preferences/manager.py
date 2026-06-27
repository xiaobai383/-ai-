"""User preferences management — store and retrieve user settings."""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class UserPreferences:
    """User preference settings."""

    # Default processing mode
    default_mode: str = "privacy_enhanced"

    # Output format preferences
    default_output_format: str = "markdown"
    include_metadata: bool = True
    include_timestamp: bool = True

    # UI preferences
    show_token_count: bool = True
    show_cost_estimate: bool = True
    auto_preview: bool = True

    # Workflow preferences
    favorite_templates: list = field(default_factory=list)
    recent_templates: list = field(default_factory=list)

    # Notification preferences
    notify_on_completion: bool = True
    notify_on_cost_threshold: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class PreferenceManager:
    """Manages user preferences with file-based persistence."""

    def __init__(self, preferences_dir: str | Path | None = None):
        """Initialize preference manager.

        Args:
            preferences_dir: Directory to store preferences. Defaults to 'data/preferences'.
        """
        if preferences_dir is None:
            preferences_dir = Path("data/preferences")

        self.preferences_dir = Path(preferences_dir)
        self.preferences_dir.mkdir(parents=True, exist_ok=True)

        self._preferences_file = self.preferences_dir / "user_preferences.json"
        self._cache: Optional[UserPreferences] = None

    def load(self) -> UserPreferences:
        """Load user preferences from file.

        Returns:
            UserPreferences instance.
        """
        if self._cache is not None:
            return self._cache

        if self._preferences_file.exists():
            try:
                with open(self._preferences_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = UserPreferences.from_dict(data)
            except Exception:
                self._cache = UserPreferences()
        else:
            self._cache = UserPreferences()

        return self._cache

    def save(self, preferences: UserPreferences) -> None:
        """Save user preferences to file.

        Args:
            preferences: UserPreferences to save.
        """
        self._cache = preferences

        with open(self._preferences_file, "w", encoding="utf-8") as f:
            json.dump(preferences.to_dict(), f, ensure_ascii=False, indent=2)

    def update(self, **kwargs) -> UserPreferences:
        """Update specific preference fields.

        Args:
            **kwargs: Fields to update.

        Returns:
            Updated UserPreferences.
        """
        preferences = self.load()

        for key, value in kwargs.items():
            if hasattr(preferences, key):
                setattr(preferences, key, value)

        self.save(preferences)
        return preferences

    def add_recent_template(self, template_name: str, max_recent: int = 10) -> None:
        """Add a template to recent templates list.

        Args:
            template_name: Name of the template.
            max_recent: Maximum number of recent templates to keep.
        """
        preferences = self.load()

        # Remove if already exists
        if template_name in preferences.recent_templates:
            preferences.recent_templates.remove(template_name)

        # Add to front
        preferences.recent_templates.insert(0, template_name)

        # Trim to max
        preferences.recent_templates = preferences.recent_templates[:max_recent]

        self.save(preferences)

    def toggle_favorite_template(self, template_name: str) -> bool:
        """Toggle a template as favorite.

        Args:
            template_name: Name of the template.

        Returns:
            True if added to favorites, False if removed.
        """
        preferences = self.load()

        if template_name in preferences.favorite_templates:
            preferences.favorite_templates.remove(template_name)
            self.save(preferences)
            return False
        else:
            preferences.favorite_templates.append(template_name)
            self.save(preferences)
            return True

    def reset(self) -> UserPreferences:
        """Reset preferences to defaults.

        Returns:
            Default UserPreferences.
        """
        default = UserPreferences()
        self.save(default)
        return default
